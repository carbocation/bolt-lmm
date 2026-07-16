/*
   This file is part of the BOLT-LMM linear mixed model software package
   developed by Po-Ru Loh.  Copyright (C) 2014-2025 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "CudaStep1.hpp"

namespace LMM {

  namespace {

    struct SharedPackedGenotypeCache {
      const uchar *hostGenotypes;
      uchar *deviceGenotypes;
      uint64 M, bytesPerSnp, cachedSnps, references;

      SharedPackedGenotypeCache() : hostGenotypes(nullptr), deviceGenotypes(nullptr),
        M(0), bytesPerSnp(0), cachedSnps(0), references(0) {}
    };

    SharedPackedGenotypeCache sharedPackedCache;

    bool sharedCacheMatches(const uchar *hostGenotypes, uint64 M,
                            uint64 bytesPerSnp) {
      return sharedPackedCache.deviceGenotypes != nullptr &&
             sharedPackedCache.hostGenotypes == hostGenotypes &&
             sharedPackedCache.M == M &&
             sharedPackedCache.bytesPerSnp == bytesPerSnp;
    }

    void checkCuda(cudaError_t status, const char *operation) {
      if (status != cudaSuccess) {
        std::ostringstream msg;
        msg << operation << ": " << cudaGetErrorString(status);
        throw std::runtime_error(msg.str());
      }
    }

    void checkCublas(cublasStatus_t status, const char *operation) {
      if (status != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream msg;
        msg << operation << ": cuBLAS status " << static_cast<int>(status);
        throw std::runtime_error(msg.str());
      }
    }

    __global__ void decodeSnpBlock(double snpBlock[], const uchar packedBlock[],
                                   const double maskIndivs[], const double lookup[],
                                   const double negCovComps[], const uchar projMaskSnps[],
                                   const uchar activeMaskSnps[],
                                   uint64 m0, uint64 blockSize, uint64 Nstride,
                                   uint64 Cstride) {
      const uint64 NCstride = Nstride + Cstride;
      const uint64 index = static_cast<uint64>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (index >= blockSize * NCstride)
        return;

      const uint64 mPlus = index / NCstride;
      const uint64 nc = index - mPlus * NCstride;
      const uint64 m = m0 + mPlus;
      double value = 0;
      if (projMaskSnps[m] && (!activeMaskSnps || activeMaskSnps[m])) {
        if (nc < Nstride) {
          const uint64 bytesPerSnp = Nstride >> 2;
          const uchar packed = packedBlock[mPlus * bytesPerSnp + (nc >> 2)];
          const uchar bedCode = (packed >> (2 * (nc & 3))) & 3;
          const uchar lookupIndex = bedCode == 0 ? 2 : bedCode == 1 ? 3 :
                                    bedCode == 2 ? 1 : 0;
          value = lookup[m * 4 + lookupIndex] * (maskIndivs ? maskIndivs[nc] : 1.0);
        }
        else {
          value = negCovComps[m * Cstride + (nc - Nstride)];
        }
      }
      snpBlock[index] = value;
    }

    __global__ void computeMarkerLookup(const uchar packedBlock[], const double maskIndivs[],
                                        double lookup[], uchar projMaskSnps[], uint64 m0,
                                        uint64 blockSize, uint64 Nstride, uint64 Nused) {
      const uint64 mPlus = blockIdx.x;
      if (mPlus >= blockSize)
        return;
      const uint64 m = m0 + mPlus;
      if (!projMaskSnps[m])
        return;

      const uint64 bytesPerSnp = Nstride >> 2;
      double sum = 0;
      uint64 count = 0;
      for (uint64 byte = threadIdx.x; byte < bytesPerSnp; byte += blockDim.x) {
        const uchar packed = packedBlock[mPlus * bytesPerSnp + byte];
        const uint64 n0 = byte << 2;
        for (uint64 offset = 0; offset < 4; offset++) {
          const uchar bedCode = (packed >> (2 * offset)) & 3;
          if (maskIndivs[n0 + offset] && bedCode != 1) {
            sum += bedCode == 0 ? 2 : bedCode == 2 ? 1 : 0;
            count++;
          }
        }
      }

      __shared__ double sums[256];
      __shared__ uint64 counts[256];
      __shared__ double mean;
      sums[threadIdx.x] = sum;
      counts[threadIdx.x] = count;
      __syncthreads();
      for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
          sums[threadIdx.x] += sums[threadIdx.x + stride];
          counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0)
        mean = counts[0] ? sums[0] / counts[0] : 0;
      __syncthreads();

      double meanCenterNorm2 = 0;
      for (uint64 byte = threadIdx.x; byte < bytesPerSnp; byte += blockDim.x) {
        const uchar packed = packedBlock[mPlus * bytesPerSnp + byte];
        const uint64 n0 = byte << 2;
        for (uint64 offset = 0; offset < 4; offset++) {
          const uchar bedCode = (packed >> (2 * offset)) & 3;
          if (maskIndivs[n0 + offset] && bedCode != 1) {
            const double genotype = bedCode == 0 ? 2 : bedCode == 2 ? 1 : 0;
            const double centered = genotype - mean;
            meanCenterNorm2 += centered * centered;
          }
        }
      }
      sums[threadIdx.x] = meanCenterNorm2;
      __syncthreads();
      for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride)
          sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
      }

      if (threadIdx.x == 0) {
        if (!counts[0] || sums[0] <= 0) {
          projMaskSnps[m] = 0;
          return;
        }
        const double invMeanCenterNorm = sqrt((Nused - 1) / sums[0]);
        lookup[m * 4] = -mean * invMeanCenterNorm;
        lookup[m * 4 + 1] = (1 - mean) * invMeanCenterNorm;
        lookup[m * 4 + 2] = (2 - mean) * invMeanCenterNorm;
        lookup[m * 4 + 3] = 0;
      }
    }

    __global__ void finalizeMarkerInitialization(double lookup[], double negCovComps[],
                                                  double Xnorm2s[], uchar projMaskSnps[],
                                                  uint64 m0, uint64 blockSize,
                                                  uint64 Cindep, uint64 Cstride,
                                                  uint64 Nused) {
      const uint64 mPlus = static_cast<uint64>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (mPlus >= blockSize)
        return;
      const uint64 m = m0 + mPlus;
      if (!projMaskSnps[m])
        return;
      double projNorm2 = Nused - 1;
      for (uint64 c = 0; c < Cindep; c++) {
        const double component = negCovComps[m * Cstride + c];
        projNorm2 -= component * component;
      }
      if (projNorm2 < 0.1) {
        projMaskSnps[m] = 0;
        lookup[m * 4] = lookup[m * 4 + 1] = 0;
        lookup[m * 4 + 2] = lookup[m * 4 + 3] = 0;
        return;
      }
      for (uint64 c = 0; c < Cindep; c++)
        negCovComps[m * Cstride + c] *= -1;
      Xnorm2s[m] = projNorm2;
    }

    __global__ void applyBatchMask(double Xtrans[], const uchar batchMaskSnps[],
                                   uint64 m0, uint64 blockSize, uint64 B) {
      const uint64 index = static_cast<uint64>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (index < blockSize * B) {
        const uint64 mPlus = index / B;
        const uint64 b = index - mPlus * B;
        Xtrans[index] *= batchMaskSnps[(m0 + mPlus) * B + b];
      }
    }

    __global__ void flipCovariateComponents(double snpBlock[], uint64 blockSize,
                                             uint64 Nstride, uint64 Cstride) {
      const uint64 index = static_cast<uint64>(blockIdx.x) * blockDim.x + threadIdx.x;
      if (index < blockSize * Cstride) {
        const uint64 mPlus = index / Cstride;
        const uint64 c = index - mPlus * Cstride;
        snpBlock[mPlus * (Nstride + Cstride) + Nstride + c] *= -1;
      }
    }

    unsigned int numBlocks(uint64 count, unsigned int threads) {
      return static_cast<unsigned int>((count + threads - 1) / threads);
    }

    bool allocateSharedPackedCache(const uchar *hostGenotypes, uint64 M,
                                   uint64 bytesPerSnp, uint64 snpsPerBlock,
                                   uint64 NCstride) {
      if (sharedCacheMatches(hostGenotypes, M, bytesPerSnp))
        return false;
      if (sharedPackedCache.deviceGenotypes != nullptr) {
        if (sharedPackedCache.references)
          return false;
        cudaFree(sharedPackedCache.deviceGenotypes);
        sharedPackedCache = SharedPackedGenotypeCache();
      }

      size_t freeBytes = 0, totalBytes = 0;
      if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess) {
        cudaGetLastError();
        return false;
      }
      const uint64 decodedBlockBytes = snpsPerBlock * NCstride * sizeof(double);
      const uint64 reserveBytes = decodedBlockBytes + (4ULL << 30);
      if (freeBytes <= reserveBytes)
        return false;

      uint64 candidateSnps = std::min<uint64>(M, (freeBytes - reserveBytes) /
                                                  bytesPerSnp);
      if (candidateSnps < M)
        candidateSnps -= candidateSnps % snpsPerBlock;
      if (!candidateSnps)
        return false;

      uchar *deviceCache = nullptr;
      const uint64 cacheBytes = candidateSnps * bytesPerSnp;
      cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&deviceCache), cacheBytes);
      if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
      }

      sharedPackedCache.hostGenotypes = hostGenotypes;
      sharedPackedCache.deviceGenotypes = deviceCache;
      sharedPackedCache.M = M;
      sharedPackedCache.bytesPerSnp = bytesPerSnp;
      sharedPackedCache.cachedSnps = candidateSnps;
      sharedPackedCache.references = 0;
      return true;
    }

  }

  struct CudaStep1::Impl {
    const uchar *hostGenotypes;
    uint64 M, Nstride, Cstride, NCstride, bytesPerSnp, snpsPerBlock;
    uint64 batchCapacity;

    cublasHandle_t cublas;
    uchar *packedBlock;
    double *snpBlock;
    double *maskIndivs;
    double *lookup;
    double *negCovComps;
    uchar *projMaskSnps;
    uchar *activeMaskSnps;
    uchar *batchMaskSnps;
    double *inCovCompVecs;
    double *outCovCompVecs;
    double *Xtrans;
    double *bayesGram;
    uint64 gramCapacity;
    const uchar *cachedPackedGenotypes;
    uint64 cachedSnps;
    bool packedCacheAttached;

    Impl(const uchar hostGenotypesIn[], const double maskIndivsIn[],
         const double (*lookupIn)[4], const double negCovCompsIn[],
         const uchar projMaskSnpsIn[], uint64 MIn, uint64 NstrideIn, uint64 CstrideIn) :
      hostGenotypes(hostGenotypesIn), M(MIn), Nstride(NstrideIn), Cstride(CstrideIn),
      NCstride(NstrideIn + CstrideIn), bytesPerSnp(NstrideIn >> 2),
      snpsPerBlock(std::min<uint64>(MIn, 1024)), batchCapacity(0), cublas(nullptr),
      packedBlock(nullptr), snpBlock(nullptr), maskIndivs(nullptr), lookup(nullptr),
      negCovComps(nullptr), projMaskSnps(nullptr), activeMaskSnps(nullptr),
      batchMaskSnps(nullptr), inCovCompVecs(nullptr), outCovCompVecs(nullptr),
      Xtrans(nullptr), bayesGram(nullptr), gramCapacity(0),
      cachedPackedGenotypes(nullptr), cachedSnps(0), packedCacheAttached(false) {

      if (!hostGenotypes || !M || (Nstride & 3))
        throw std::runtime_error("Invalid packed genotype dimensions for CUDA Step 1");

      int device = 0;
      cudaDeviceProp properties;
      checkCuda(cudaGetDevice(&device), "cudaGetDevice");
      checkCuda(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
      checkCublas(cublasCreate(&cublas), "cublasCreate");

      checkCuda(cudaMalloc(reinterpret_cast<void **>(&packedBlock),
                           snpsPerBlock * bytesPerSnp * sizeof(*packedBlock)),
                "cudaMalloc packed genotype block");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&snpBlock),
                           snpsPerBlock * NCstride * sizeof(*snpBlock)),
                "cudaMalloc decoded SNP block");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&maskIndivs),
                           Nstride * sizeof(*maskIndivs)), "cudaMalloc sample mask");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&lookup), M * 4 * sizeof(*lookup)),
                "cudaMalloc genotype lookup");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&negCovComps),
                           M * Cstride * sizeof(*negCovComps)),
                "cudaMalloc covariate components");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&projMaskSnps),
                           M * sizeof(*projMaskSnps)), "cudaMalloc SNP mask");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&activeMaskSnps),
                           M * sizeof(*activeMaskSnps)), "cudaMalloc active SNP mask");

      checkCuda(cudaMemcpy(maskIndivs, maskIndivsIn, Nstride * sizeof(*maskIndivs),
                           cudaMemcpyHostToDevice), "copy sample mask to CUDA");
      checkCuda(cudaMemcpy(lookup, lookupIn, M * 4 * sizeof(*lookup),
                           cudaMemcpyHostToDevice), "copy genotype lookup to CUDA");
      checkCuda(cudaMemcpy(negCovComps, negCovCompsIn,
                           M * Cstride * sizeof(*negCovComps), cudaMemcpyHostToDevice),
                "copy covariate components to CUDA");
      checkCuda(cudaMemcpy(projMaskSnps, projMaskSnpsIn, M * sizeof(*projMaskSnps),
                           cudaMemcpyHostToDevice), "copy SNP mask to CUDA");

      attachPackedCache();

      std::cout << "CUDA Step 1 enabled on " << properties.name
                << " (compute capability " << properties.major << "." << properties.minor
                << ", " << snpsPerBlock << " SNPs per GPU block)" << std::endl;
    }

    ~Impl() {
      cudaFree(bayesGram);
      cudaFree(Xtrans);
      cudaFree(outCovCompVecs);
      cudaFree(inCovCompVecs);
      cudaFree(batchMaskSnps);
      cudaFree(activeMaskSnps);
      cudaFree(projMaskSnps);
      cudaFree(negCovComps);
      cudaFree(lookup);
      cudaFree(maskIndivs);
      cudaFree(snpBlock);
      cudaFree(packedBlock);
      releasePackedCache();
      if (cublas)
        cublasDestroy(cublas);
    }

    void attachPackedCache() {
      if (sharedCacheMatches(hostGenotypes, M, bytesPerSnp)) {
        cachedPackedGenotypes = sharedPackedCache.deviceGenotypes;
        cachedSnps = sharedPackedCache.cachedSnps;
        sharedPackedCache.references++;
        packedCacheAttached = true;
        std::cout << "Reusing CUDA packed genotype cache (" << cachedSnps << "/" << M
                  << " SNPs)" << std::endl;
        return;
      }
      if (sharedPackedCache.deviceGenotypes != nullptr)
        return;

      if (!allocateSharedPackedCache(hostGenotypes, M, bytesPerSnp, snpsPerBlock,
                                    NCstride))
        return;

      const uint64 cacheBytes = sharedPackedCache.cachedSnps * bytesPerSnp;
      cudaError_t status = cudaMemcpy(sharedPackedCache.deviceGenotypes, hostGenotypes,
                                      cacheBytes, cudaMemcpyHostToDevice);
      if (status != cudaSuccess) {
        cudaFree(sharedPackedCache.deviceGenotypes);
        sharedPackedCache = SharedPackedGenotypeCache();
        cudaGetLastError();
        return;
      }

      sharedPackedCache.references = 1;
      cachedPackedGenotypes = sharedPackedCache.deviceGenotypes;
      cachedSnps = sharedPackedCache.cachedSnps;
      packedCacheAttached = true;
      std::cout << "CUDA packed genotype cache: " << cachedSnps << "/" << M
                << " SNPs (" << cacheBytes / static_cast<double>(1ULL << 30)
                << " GiB)" << std::endl;
    }

    void releasePackedCache() {
      if (!packedCacheAttached)
        return;
      if (--sharedPackedCache.references == 0) {
        cudaFree(sharedPackedCache.deviceGenotypes);
        sharedPackedCache = SharedPackedGenotypeCache();
      }
      cachedPackedGenotypes = nullptr;
      cachedSnps = 0;
      packedCacheAttached = false;
    }

    const uchar *preparePackedBlock(uint64 m0, uint64 blockSize) {
      if (m0 + blockSize <= cachedSnps)
        return cachedPackedGenotypes + m0 * bytesPerSnp;
      checkCuda(cudaMemcpy(packedBlock, hostGenotypes + m0 * bytesPerSnp,
                           blockSize * bytesPerSnp * sizeof(*packedBlock),
                           cudaMemcpyHostToDevice), "copy packed genotype block to CUDA");
      return packedBlock;
    }

    void ensureBatchCapacity(uint64 B) {
      if (B <= batchCapacity)
        return;
      cudaFree(Xtrans);
      cudaFree(outCovCompVecs);
      cudaFree(inCovCompVecs);
      cudaFree(batchMaskSnps);
      Xtrans = outCovCompVecs = inCovCompVecs = nullptr;
      batchMaskSnps = nullptr;

      checkCuda(cudaMalloc(reinterpret_cast<void **>(&batchMaskSnps),
                           M * B * sizeof(*batchMaskSnps)), "cudaMalloc batch SNP mask");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&inCovCompVecs),
                           B * NCstride * sizeof(*inCovCompVecs)),
                "cudaMalloc input covariate vectors");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&outCovCompVecs),
                           B * NCstride * sizeof(*outCovCompVecs)),
                "cudaMalloc output covariate vectors");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&Xtrans),
                           snpsPerBlock * B * sizeof(*Xtrans)),
                "cudaMalloc transposed SNP product");
      batchCapacity = B;
    }

    void multiply(double out[], const double in[], const uchar batchMask[], uint64 B) {
      ensureBatchCapacity(B);
      const size_t covCompBytes = B * NCstride * sizeof(*inCovCompVecs);
      checkCuda(cudaMemcpy(inCovCompVecs, in, covCompBytes, cudaMemcpyHostToDevice),
                "copy input vectors to CUDA");
      checkCuda(cudaMemset(outCovCompVecs, 0, covCompBytes), "clear CUDA output vectors");
      checkCuda(cudaMemcpy(batchMaskSnps, batchMask, M * B * sizeof(*batchMaskSnps),
                           cudaMemcpyHostToDevice), "copy batch SNP mask to CUDA");

      const unsigned int threads = 256;
      const double one = 1.0, zero = 0.0;
      for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
        const uint64 blockSize = std::min<uint64>(snpsPerBlock, M - m0);
        const uchar *packedInput = preparePackedBlock(m0, blockSize);

        decodeSnpBlock<<<numBlocks(blockSize * NCstride, threads), threads>>>
          (snpBlock, packedInput, maskIndivs, lookup, negCovComps, projMaskSnps,
           nullptr, m0, blockSize, Nstride, Cstride);
        checkCuda(cudaGetLastError(), "launch CUDA genotype decoder");

        checkCublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                static_cast<int>(B), static_cast<int>(blockSize),
                                static_cast<int>(NCstride), &one, inCovCompVecs,
                                static_cast<int>(NCstride), snpBlock,
                                static_cast<int>(NCstride), &zero, Xtrans,
                                static_cast<int>(B)), "cuBLAS X transpose multiply");

        applyBatchMask<<<numBlocks(blockSize * B, threads), threads>>>
          (Xtrans, batchMaskSnps, m0, blockSize, B);
        checkCuda(cudaGetLastError(), "launch CUDA SNP batch mask");
        if (Cstride) {
          flipCovariateComponents<<<numBlocks(blockSize * Cstride, threads), threads>>>
            (snpBlock, blockSize, Nstride, Cstride);
          checkCuda(cudaGetLastError(), "launch CUDA covariate sign flip");
        }

        checkCublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                static_cast<int>(NCstride), static_cast<int>(B),
                                static_cast<int>(blockSize), &one, snpBlock,
                                static_cast<int>(NCstride), Xtrans, static_cast<int>(B),
                                &one, outCovCompVecs, static_cast<int>(NCstride)),
                    "cuBLAS X multiply");
      }

      checkCuda(cudaMemcpy(out, outCovCompVecs, covCompBytes, cudaMemcpyDeviceToHost),
                "copy CUDA output vectors to host");
    }

    void multiplyX(double out[], const double hostCoefficients[], uint64 B,
                   bool applyIndivMask, bool positiveCovariateComponents) {
      ensureBatchCapacity(B);
      const size_t covCompBytes = B * NCstride * sizeof(*outCovCompVecs);
      checkCuda(cudaMemset(outCovCompVecs, 0, covCompBytes),
                "clear CUDA X beta output vectors");

      const unsigned int threads = 256;
      const double one = 1.0;
      for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
        const uint64 blockSize = std::min<uint64>(snpsPerBlock, M - m0);
        const uchar *packedInput = preparePackedBlock(m0, blockSize);
        checkCuda(cudaMemcpy(Xtrans, hostCoefficients + m0 * B,
                             blockSize * B * sizeof(*Xtrans), cudaMemcpyHostToDevice),
                  "copy X beta coefficients to CUDA");

        decodeSnpBlock<<<numBlocks(blockSize * NCstride, threads), threads>>>
          (snpBlock, packedInput, applyIndivMask ? maskIndivs : nullptr, lookup,
           negCovComps, projMaskSnps, nullptr, m0, blockSize, Nstride, Cstride);
        checkCuda(cudaGetLastError(), "launch CUDA X beta genotype decoder");
        if (positiveCovariateComponents && Cstride) {
          flipCovariateComponents<<<numBlocks(blockSize * Cstride, threads), threads>>>
            (snpBlock, blockSize, Nstride, Cstride);
          checkCuda(cudaGetLastError(), "launch CUDA X beta covariate sign flip");
        }

        checkCublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                                static_cast<int>(NCstride), static_cast<int>(B),
                                static_cast<int>(blockSize), &one, snpBlock,
                                static_cast<int>(NCstride), Xtrans, static_cast<int>(B),
                                &one, outCovCompVecs, static_cast<int>(NCstride)),
                    "cuBLAS X beta multiply");
      }

      checkCuda(cudaMemcpy(out, outCovCompVecs, covCompBytes, cudaMemcpyDeviceToHost),
                "copy CUDA X beta output vectors to host");
    }

    void multiplyXtrans(double hostOut[], const double hostIn[], uint64 B) {
      ensureBatchCapacity(B);
      checkCuda(cudaMemcpy(inCovCompVecs, hostIn,
                           B * NCstride * sizeof(*inCovCompVecs), cudaMemcpyHostToDevice),
                "copy X transpose input vectors to CUDA");

      const unsigned int threads = 256;
      const double one = 1.0, zero = 0.0;
      for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
        const uint64 blockSize = std::min<uint64>(snpsPerBlock, M - m0);
        const uchar *packedInput = preparePackedBlock(m0, blockSize);

        decodeSnpBlock<<<numBlocks(blockSize * NCstride, threads), threads>>>
          (snpBlock, packedInput, maskIndivs, lookup, negCovComps, projMaskSnps,
           nullptr, m0, blockSize, Nstride, Cstride);
        checkCuda(cudaGetLastError(), "launch CUDA X transpose genotype decoder");

        checkCublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                static_cast<int>(B), static_cast<int>(blockSize),
                                static_cast<int>(NCstride), &one, inCovCompVecs,
                                static_cast<int>(NCstride), snpBlock,
                                static_cast<int>(NCstride), &zero, Xtrans,
                                static_cast<int>(B)), "cuBLAS X transpose multiply");
        checkCuda(cudaMemcpy(hostOut + m0 * B, Xtrans,
                             blockSize * B * sizeof(*Xtrans), cudaMemcpyDeviceToHost),
                  "copy CUDA X transpose products to host");
      }
    }

    void beginBayes(const double yResid[], const uchar activeMask[], uint64 B) {
      ensureBatchCapacity(B);
      checkCuda(cudaMemcpy(inCovCompVecs, yResid, B * NCstride * sizeof(*inCovCompVecs),
                           cudaMemcpyHostToDevice), "copy Bayesian residuals to CUDA");
      checkCuda(cudaMemcpy(activeMaskSnps, activeMask, M * sizeof(*activeMaskSnps),
                           cudaMemcpyHostToDevice), "copy active SNP mask to CUDA");
    }

    void computeBayes(double hostXtrans[], double hostSnpDots[], uint64 snpDotsStride,
                      uint64 m0, uint64 blockSize, uint64 B, uint64 Bleft,
                      bool computeSnpDots) {
      if (blockSize > snpsPerBlock)
        throw std::runtime_error("CUDA Bayesian block exceeds allocated SNP block");

      const uchar *packedInput = preparePackedBlock(m0, blockSize);
      const unsigned int threads = 256;
      decodeSnpBlock<<<numBlocks(blockSize * NCstride, threads), threads>>>
        (snpBlock, packedInput, maskIndivs, lookup, negCovComps, projMaskSnps,
         activeMaskSnps, m0, blockSize, Nstride, Cstride);
      checkCuda(cudaGetLastError(), "launch Bayesian CUDA genotype decoder");

      const double one = 1.0, zero = 0.0, minusOne = -1.0;
      checkCublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                              static_cast<int>(Bleft), static_cast<int>(blockSize),
                              static_cast<int>(NCstride), &one, inCovCompVecs,
                              static_cast<int>(NCstride), snpBlock,
                              static_cast<int>(NCstride), &zero, Xtrans,
                              static_cast<int>(B)), "cuBLAS Bayesian X transpose multiply");
      checkCuda(cudaMemcpy(hostXtrans, Xtrans, blockSize * B * sizeof(*Xtrans),
                           cudaMemcpyDeviceToHost), "copy Bayesian X transpose product to host");

      if (computeSnpDots) {
        if (snpDotsStride > gramCapacity) {
          cudaFree(bayesGram);
          bayesGram = nullptr;
          checkCuda(cudaMalloc(reinterpret_cast<void **>(&bayesGram),
                               snpDotsStride * snpDotsStride * sizeof(*bayesGram)),
                    "cudaMalloc Bayesian SNP Gram matrix");
          gramCapacity = snpDotsStride;
        }
        checkCublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                static_cast<int>(blockSize), static_cast<int>(blockSize),
                                static_cast<int>(Nstride), &one, snpBlock,
                                static_cast<int>(NCstride), snpBlock,
                                static_cast<int>(NCstride), &zero, bayesGram,
                                static_cast<int>(snpDotsStride)),
                    "cuBLAS Bayesian SNP Gram matrix");
        if (Cstride)
          checkCublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                  static_cast<int>(blockSize), static_cast<int>(blockSize),
                                  static_cast<int>(Cstride), &minusOne, snpBlock + Nstride,
                                  static_cast<int>(NCstride), snpBlock + Nstride,
                                  static_cast<int>(NCstride), &one, bayesGram,
                                  static_cast<int>(snpDotsStride)),
                      "cuBLAS Bayesian covariate Gram correction");
        checkCuda(cudaMemcpy2D(hostSnpDots, snpDotsStride * sizeof(*hostSnpDots),
                               bayesGram, snpDotsStride * sizeof(*bayesGram),
                               blockSize * sizeof(*hostSnpDots), blockSize,
                               cudaMemcpyDeviceToHost), "copy Bayesian SNP Gram matrix to host");
      }
    }

    void updateBayes(const double hostBetaUpdates[], uint64 blockSize,
                     uint64 B, uint64 Bleft) {
      checkCuda(cudaMemcpy(Xtrans, hostBetaUpdates, blockSize * B * sizeof(*Xtrans),
                           cudaMemcpyHostToDevice), "copy Bayesian beta updates to CUDA");
      const unsigned int threads = 256;
      if (Cstride) {
        flipCovariateComponents<<<numBlocks(blockSize * Cstride, threads), threads>>>
          (snpBlock, blockSize, Nstride, Cstride);
        checkCuda(cudaGetLastError(), "launch Bayesian CUDA covariate sign flip");
      }
      const double one = 1.0;
      checkCublas(cublasDgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                              static_cast<int>(NCstride), static_cast<int>(Bleft),
                              static_cast<int>(blockSize), &one, snpBlock,
                              static_cast<int>(NCstride), Xtrans, static_cast<int>(B),
                              &one, inCovCompVecs, static_cast<int>(NCstride)),
                  "cuBLAS Bayesian residual update");
    }

    void endBayes(double hostYResid[], uint64 B) {
      checkCuda(cudaMemcpy(hostYResid, inCovCompVecs,
                           B * NCstride * sizeof(*inCovCompVecs), cudaMemcpyDeviceToHost),
                "copy Bayesian residuals from CUDA");
    }
  };

  void CudaStep1::initializeMarkers
  (const uchar hostGenotypes[], const double hostMaskIndivs[], const double hostCovBasis[],
   uint64 Cindep, double (*hostLookup)[4], double hostNegCovComps[], double hostXnorm2s[],
   uchar hostProjMaskSnps[], uint64 Nused, uint64 M, uint64 Nstride, uint64 Cstride) {
    if (!hostGenotypes || !hostMaskIndivs || (Cindep && !hostCovBasis) || !hostLookup ||
        !hostNegCovComps || !hostXnorm2s || !hostProjMaskSnps || Nused < 2 || !M ||
        !Cstride || (Nstride & 3) || Cindep > Cstride)
      throw std::runtime_error("Invalid CUDA marker initialization dimensions");

    const uint64 bytesPerSnp = Nstride >> 2;
    const uint64 NCstride = Nstride + Cstride;
    const uint64 snpsPerBlock = std::min<uint64>(M, 1024);
    uchar *packedBlock = nullptr, *projMaskSnps = nullptr;
    double *snpBlock = nullptr, *maskIndivs = nullptr, *covBasis = nullptr;
    double *lookup = nullptr, *negCovComps = nullptr, *Xnorm2s = nullptr;
    cublasHandle_t cublas = nullptr;

    checkCublas(cublasCreate(&cublas), "cublasCreate for marker initialization");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&packedBlock),
                         snpsPerBlock * bytesPerSnp * sizeof(*packedBlock)),
              "cudaMalloc marker initialization packed block");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&snpBlock),
                         snpsPerBlock * NCstride * sizeof(*snpBlock)),
              "cudaMalloc marker initialization decoded block");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&maskIndivs),
                         Nstride * sizeof(*maskIndivs)),
              "cudaMalloc marker initialization sample mask");
    if (Cindep)
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&covBasis),
                           Cindep * Nstride * sizeof(*covBasis)),
                "cudaMalloc marker initialization covariate basis");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&lookup), M * 4 * sizeof(*lookup)),
              "cudaMalloc marker initialization lookup");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&negCovComps),
                         M * Cstride * sizeof(*negCovComps)),
              "cudaMalloc marker initialization covariate components");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&Xnorm2s), M * sizeof(*Xnorm2s)),
              "cudaMalloc marker initialization SNP norms");
    checkCuda(cudaMalloc(reinterpret_cast<void **>(&projMaskSnps),
                         M * sizeof(*projMaskSnps)),
              "cudaMalloc marker initialization SNP mask");

    checkCuda(cudaMemcpy(maskIndivs, hostMaskIndivs, Nstride * sizeof(*maskIndivs),
                         cudaMemcpyHostToDevice),
              "copy marker initialization sample mask");
    if (Cindep)
      checkCuda(cudaMemcpy(covBasis, hostCovBasis, Cindep * Nstride * sizeof(*covBasis),
                           cudaMemcpyHostToDevice),
                "copy marker initialization covariate basis");
    checkCuda(cudaMemcpy(projMaskSnps, hostProjMaskSnps, M * sizeof(*projMaskSnps),
                         cudaMemcpyHostToDevice),
              "copy marker initialization SNP mask");
    checkCuda(cudaMemset(lookup, 0, M * 4 * sizeof(*lookup)),
              "clear marker initialization lookup");
    checkCuda(cudaMemset(negCovComps, 0, M * Cstride * sizeof(*negCovComps)),
              "clear marker initialization covariate components");
    checkCuda(cudaMemset(Xnorm2s, 0, M * sizeof(*Xnorm2s)),
              "clear marker initialization SNP norms");

    const bool populateSharedCache =
      allocateSharedPackedCache(hostGenotypes, M, bytesPerSnp, snpsPerBlock, NCstride);
    const bool useSharedCache = sharedCacheMatches(hostGenotypes, M, bytesPerSnp);
    if (populateSharedCache) {
      const uint64 cacheBytes = sharedPackedCache.cachedSnps * bytesPerSnp;
      std::cout << "CUDA packed genotype cache: " << sharedPackedCache.cachedSnps
                << "/" << M << " SNPs ("
                << cacheBytes / static_cast<double>(1ULL << 30)
                << " GiB), populated during marker initialization" << std::endl;
    }

    const unsigned int threads = 256;
    const double one = 1.0, zero = 0.0;
    for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
      const uint64 blockSize = std::min<uint64>(snpsPerBlock, M - m0);
      const uint64 blockBytes = blockSize * bytesPerSnp * sizeof(*packedBlock);
      const bool blockCached = useSharedCache &&
                               m0 + blockSize <= sharedPackedCache.cachedSnps;
      uchar *packedInput = packedBlock;
      if (blockCached) {
        packedInput = sharedPackedCache.deviceGenotypes + m0 * bytesPerSnp;
        if (populateSharedCache)
          checkCuda(cudaMemcpy(packedInput, hostGenotypes + m0 * bytesPerSnp,
                               blockBytes, cudaMemcpyHostToDevice),
                    "populate CUDA packed genotype cache during marker initialization");
      }
      else
        checkCuda(cudaMemcpy(packedBlock, hostGenotypes + m0 * bytesPerSnp,
                             blockBytes, cudaMemcpyHostToDevice),
                  "copy packed genotypes for marker initialization");
      computeMarkerLookup<<<static_cast<unsigned int>(blockSize), threads>>>
        (packedInput, maskIndivs, lookup, projMaskSnps, m0, blockSize, Nstride, Nused);
      checkCuda(cudaGetLastError(), "launch marker lookup initialization");
      decodeSnpBlock<<<numBlocks(blockSize * NCstride, threads), threads>>>
        (snpBlock, packedInput, maskIndivs, lookup, negCovComps, projMaskSnps,
         nullptr, m0, blockSize, Nstride, Cstride);
      checkCuda(cudaGetLastError(), "launch marker initialization decoder");
      if (Cindep)
        checkCublas(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                static_cast<int>(Cindep), static_cast<int>(blockSize),
                                static_cast<int>(Nstride), &one, covBasis,
                                static_cast<int>(Nstride), snpBlock,
                                static_cast<int>(NCstride), &zero,
                                negCovComps + m0 * Cstride, static_cast<int>(Cstride)),
                    "cuBLAS marker covariate projection");
      finalizeMarkerInitialization<<<numBlocks(blockSize, threads), threads>>>
        (lookup, negCovComps, Xnorm2s, projMaskSnps, m0, blockSize,
         Cindep, Cstride, Nused);
      checkCuda(cudaGetLastError(), "launch marker initialization finalizer");
    }

    checkCuda(cudaMemcpy(hostLookup, lookup, M * 4 * sizeof(*lookup),
                         cudaMemcpyDeviceToHost),
              "copy initialized marker lookup from CUDA");
    checkCuda(cudaMemcpy(hostNegCovComps, negCovComps,
                         M * Cstride * sizeof(*negCovComps), cudaMemcpyDeviceToHost),
              "copy initialized marker covariate components from CUDA");
    checkCuda(cudaMemcpy(hostXnorm2s, Xnorm2s, M * sizeof(*Xnorm2s),
                         cudaMemcpyDeviceToHost),
              "copy initialized marker norms from CUDA");
    checkCuda(cudaMemcpy(hostProjMaskSnps, projMaskSnps, M * sizeof(*projMaskSnps),
                         cudaMemcpyDeviceToHost),
              "copy initialized marker mask from CUDA");

    cudaFree(projMaskSnps);
    cudaFree(Xnorm2s);
    cudaFree(negCovComps);
    cudaFree(lookup);
    cudaFree(covBasis);
    cudaFree(maskIndivs);
    cudaFree(snpBlock);
    cudaFree(packedBlock);
    cublasDestroy(cublas);
  }

  CudaStep1::CudaStep1(const uchar genotypes[], const double maskIndivs[],
                       const double (*snpValueLookup)[4],
                       const double snpCovBasisNegComps[], const uchar projMaskSnps[],
                       uint64 M, uint64 Nstride, uint64 Cstride) :
    impl(new Impl(genotypes, maskIndivs, snpValueLookup, snpCovBasisNegComps,
                  projMaskSnps, M, Nstride, Cstride)) {
  }

  CudaStep1::~CudaStep1() { delete impl; }

  void CudaStep1::multXXtransMask(double outCovCompVecs[], const double inCovCompVecs[],
                                  const uchar batchMaskSnps[], uint64 B) {
    impl->multiply(outCovCompVecs, inCovCompVecs, batchMaskSnps, B);
  }

  void CudaStep1::multX(double outCovCompVecs[], const double coefficients[], uint64 B,
                        bool applyIndivMask, bool positiveCovariateComponents) {
    impl->multiplyX(outCovCompVecs, coefficients, B, applyIndivMask,
                    positiveCovariateComponents);
  }

  void CudaStep1::multXtrans(double outSnpProducts[], const double inCovCompVecs[],
                             uint64 B) {
    impl->multiplyXtrans(outSnpProducts, inCovCompVecs, B);
  }

  void CudaStep1::beginBayesIteration(const double yResidCovCompVecs[],
                                      const uchar activeMaskSnps[], uint64 B) {
    impl->beginBayes(yResidCovCompVecs, activeMaskSnps, B);
  }

  void CudaStep1::computeBayesBlock(double XtransResids[], double snpDots[],
                                    uint64 snpDotsStride, uint64 m0, uint64 blockSize,
                                    uint64 B, uint64 Bleft, bool computeSnpDots) {
    impl->computeBayes(XtransResids, snpDots, snpDotsStride, m0, blockSize, B, Bleft,
                       computeSnpDots);
  }

  void CudaStep1::updateBayesResidual(const double betaBlockUpdates[], uint64 blockSize,
                                      uint64 B, uint64 Bleft) {
    impl->updateBayes(betaBlockUpdates, blockSize, B, Bleft);
  }

  void CudaStep1::endBayesIteration(double yResidCovCompVecs[], uint64 B) {
    impl->endBayes(yResidCovCompVecs, B);
  }

}
