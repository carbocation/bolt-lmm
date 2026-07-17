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
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "CudaStage2.hpp"

namespace LMM {

  namespace {

    const uint64 CUDA_STAGE2_MAX_SCORE_VECTORS = 32;

    void checkCuda(cudaError_t status, const char *operation) {
      if (status != cudaSuccess) {
        std::ostringstream msg;
        msg << operation << ": " << cudaGetErrorString(status);
        throw std::runtime_error(msg.str());
      }
    }

    __device__ inline void decodeHardcall(uchar packed, uint64 offset,
                                          bool pgenEncoding, double *genotype,
                                          bool *missing) {
      const uchar code = (packed >> (2 * offset)) & 3;
      if (pgenEncoding) {
        *missing = code == 3;
        *genotype = code == 0 ? 0.0 : code == 1 ? 1.0 : code == 2 ? 2.0 : 0.0;
      }
      else {
        *missing = code == 1;
        *genotype = code == 0 ? 2.0 : code == 2 ? 1.0 : 0.0;
      }
    }

    __global__ void computePackedStats(const uchar packedGenotypes[],
                                       const int inputToModel[], uint64 inputSamples,
                                       uint64 includedSamples, uint64 bytesPerVariant,
                                       bool pgenEncoding, bool identityMapping,
                                       double alleleFreqs[], double missingRates[]) {
      const uint64 variant = blockIdx.x;
      const uchar *packed = packedGenotypes + variant * bytesPerVariant;
      double alleleSum = 0;
      uint64 missingCount = 0;
      for (uint64 input = threadIdx.x; input < inputSamples; input += blockDim.x) {
        const int model = identityMapping ? static_cast<int>(input) : inputToModel[input];
        if (model < 0) continue;
        double genotype;
        bool missing;
        decodeHardcall(packed[input >> 2], input & 3, pgenEncoding, &genotype, &missing);
        if (missing) missingCount++;
        else alleleSum += genotype;
      }

      __shared__ double alleleParts[256];
      __shared__ uint64 missingParts[256];
      alleleParts[threadIdx.x] = alleleSum;
      missingParts[threadIdx.x] = missingCount;
      __syncthreads();
      for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
          alleleParts[threadIdx.x] += alleleParts[threadIdx.x + stride];
          missingParts[threadIdx.x] += missingParts[threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        const uint64 observed = includedSamples - missingParts[0];
        alleleFreqs[variant] = observed ? alleleParts[0] / (2.0 * observed) : 0;
        missingRates[variant] = includedSamples ?
          static_cast<double>(missingParts[0]) / includedSamples : 1;
      }
    }

    template<unsigned int K, int MAPPING_MODE>
    __global__ void scorePackedKernel(const uchar packedGenotypes[],
                                      const int inputToModel[], uint64 inputSamples,
                                      uint64 bytesPerVariant, bool pgenEncoding,
                                      bool identityMapping, const double alleleFreqs[],
                                      const double scoreVectors[], uint64 Nstride,
                                      double output[]) {
      const uint64 variant = blockIdx.x;
      const uchar *packed = packedGenotypes + variant * bytesPerVariant;
      const double mean = 2 * alleleFreqs[variant];
      double accum[K+1];
#pragma unroll
      for (unsigned int k = 0; k <= K; k++) accum[k] = 0;

      for (uint64 input = threadIdx.x; input < inputSamples; input += blockDim.x) {
        const int model = MAPPING_MODE == 1 ? static_cast<int>(input) :
          MAPPING_MODE == 0 ? inputToModel[input] :
          identityMapping ? static_cast<int>(input) : inputToModel[input];
        if (model < 0) continue;
        double genotype;
        bool missing;
        decodeHardcall(packed[input >> 2], input & 3, pgenEncoding, &genotype, &missing);
        const double centered = missing ? 0 : genotype - mean;
#pragma unroll
        for (unsigned int k = 0; k < K; k++)
          accum[k] += centered * scoreVectors[static_cast<uint64>(k) * Nstride + model];
        accum[K] += centered * centered;
      }

      extern __shared__ double partial[];
#pragma unroll
      for (unsigned int k = 0; k <= K; k++)
        partial[static_cast<uint64>(k) * blockDim.x + threadIdx.x] = accum[k];
      __syncthreads();
      for (unsigned int stride = blockDim.x >> 1; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
          for (unsigned int k = 0; k <= K; k++)
            partial[static_cast<uint64>(k) * blockDim.x + threadIdx.x] +=
              partial[static_cast<uint64>(k) * blockDim.x + threadIdx.x + stride];
        }
        __syncthreads();
      }
      if (threadIdx.x == 0)
#pragma unroll
        for (unsigned int k = 0; k <= K; k++)
          output[variant * (K+1) + k] = partial[static_cast<uint64>(k) * blockDim.x];
    }

    template<unsigned int K, int MAPPING_MODE>
    void launchScoreKernel(const uchar packedGenotypes[], const int inputToModel[],
                           uint64 inputSamples, uint64 bytesPerVariant,
                           bool pgenEncoding, bool identityMapping,
                           const double alleleFreqs[], const double scoreVectors[],
                           uint64 Nstride, double output[], uint64 variants,
                           cudaStream_t stream) {
      const unsigned int threads = K <= 8 ? 256 : 128;
      scorePackedKernel<K, MAPPING_MODE>
        <<<static_cast<unsigned int>(variants), threads,
           (K+1) * threads * sizeof(double), stream>>>
        (packedGenotypes, inputToModel, inputSamples, bytesPerVariant, pgenEncoding,
         identityMapping, alleleFreqs, scoreVectors, Nstride, output);
    }

    template<bool IDENTITY_MAPPING>
    void launchScoreMapping(uint64 K, const uchar packedGenotypes[],
                            const int inputToModel[], uint64 inputSamples,
                            uint64 bytesPerVariant, bool pgenEncoding,
                            const double alleleFreqs[], const double scoreVectors[],
                            uint64 Nstride, double output[], uint64 variants,
                            cudaStream_t stream) {
#define BOLT_CUDA_STAGE2_SCORE_CASE(K_) \
      case K_: launchScoreKernel<K_, IDENTITY_MAPPING ? 1 : 0>(packedGenotypes, \
        inputToModel, inputSamples, bytesPerVariant, pgenEncoding, IDENTITY_MAPPING, \
        alleleFreqs, scoreVectors, \
        Nstride, output, variants, stream); break
      switch (K) {
        BOLT_CUDA_STAGE2_SCORE_CASE(9);
        BOLT_CUDA_STAGE2_SCORE_CASE(10);
        BOLT_CUDA_STAGE2_SCORE_CASE(11);
        BOLT_CUDA_STAGE2_SCORE_CASE(12);
        BOLT_CUDA_STAGE2_SCORE_CASE(13);
        BOLT_CUDA_STAGE2_SCORE_CASE(14);
        BOLT_CUDA_STAGE2_SCORE_CASE(15);
        BOLT_CUDA_STAGE2_SCORE_CASE(16);
        BOLT_CUDA_STAGE2_SCORE_CASE(17);
        BOLT_CUDA_STAGE2_SCORE_CASE(18);
        BOLT_CUDA_STAGE2_SCORE_CASE(19);
        BOLT_CUDA_STAGE2_SCORE_CASE(20);
        BOLT_CUDA_STAGE2_SCORE_CASE(21);
        BOLT_CUDA_STAGE2_SCORE_CASE(22);
        BOLT_CUDA_STAGE2_SCORE_CASE(23);
        BOLT_CUDA_STAGE2_SCORE_CASE(24);
        BOLT_CUDA_STAGE2_SCORE_CASE(25);
        BOLT_CUDA_STAGE2_SCORE_CASE(26);
        BOLT_CUDA_STAGE2_SCORE_CASE(27);
        BOLT_CUDA_STAGE2_SCORE_CASE(28);
        BOLT_CUDA_STAGE2_SCORE_CASE(29);
        BOLT_CUDA_STAGE2_SCORE_CASE(30);
        BOLT_CUDA_STAGE2_SCORE_CASE(31);
        BOLT_CUDA_STAGE2_SCORE_CASE(32);
        default: throw std::runtime_error("Unsupported CUDA Stage 2 score-vector count");
      }
#undef BOLT_CUDA_STAGE2_SCORE_CASE
    }

    void launchScore(uint64 K, const uchar packedGenotypes[], const int inputToModel[],
                     uint64 inputSamples, uint64 bytesPerVariant, bool pgenEncoding,
                     bool identityMapping, const double alleleFreqs[],
                     const double scoreVectors[], uint64 Nstride, double output[],
                     uint64 variants, cudaStream_t stream) {
      if (K <= 8) {
#define BOLT_CUDA_STAGE2_RUNTIME_CASE(K_) \
        case K_: launchScoreKernel<K_, -1>(packedGenotypes, inputToModel, inputSamples, \
          bytesPerVariant, pgenEncoding, identityMapping, alleleFreqs, scoreVectors, \
          Nstride, output, variants, stream); break
        switch (K) {
          BOLT_CUDA_STAGE2_RUNTIME_CASE(1);
          BOLT_CUDA_STAGE2_RUNTIME_CASE(2);
          BOLT_CUDA_STAGE2_RUNTIME_CASE(3);
          BOLT_CUDA_STAGE2_RUNTIME_CASE(4);
          BOLT_CUDA_STAGE2_RUNTIME_CASE(5);
          BOLT_CUDA_STAGE2_RUNTIME_CASE(6);
          BOLT_CUDA_STAGE2_RUNTIME_CASE(7);
          BOLT_CUDA_STAGE2_RUNTIME_CASE(8);
          default: throw std::runtime_error("Unsupported CUDA Stage 2 score-vector count");
        }
#undef BOLT_CUDA_STAGE2_RUNTIME_CASE
        return;
      }
      if (identityMapping)
        launchScoreMapping<true>(K, packedGenotypes, inputToModel, inputSamples,
          bytesPerVariant, pgenEncoding, alleleFreqs, scoreVectors, Nstride,
          output, variants, stream);
      else
        launchScoreMapping<false>(K, packedGenotypes, inputToModel, inputSamples,
          bytesPerVariant, pgenEncoding, alleleFreqs, scoreVectors, Nstride,
          output, variants, stream);
    }

  }

  struct CudaStage2::Impl {
    uint64 Nstride, maxVariants, inputSamples, includedSamples, bytesPerVariant;
    uint64 scoreVectorCapacity, numScoreVectors;
    bool identityMapping;
    cudaStream_t stream;
    int *deviceInputToModel;
    uchar *hostPackedGenotypes, *devicePackedGenotypes;
    double *deviceScoreVectors, *deviceOutput, *deviceAlleleFreqs, *deviceMissingRates;
    double *hostOutput, *hostAlleleFreqs, *hostMissingRates;

    Impl(uint64 NstrideIn, uint64 maxVariantsIn) :
      Nstride(NstrideIn), maxVariants(maxVariantsIn), inputSamples(0),
      includedSamples(0), bytesPerVariant(0), scoreVectorCapacity(0),
      numScoreVectors(0), identityMapping(false), stream(nullptr),
      deviceInputToModel(nullptr), hostPackedGenotypes(nullptr),
      devicePackedGenotypes(nullptr), deviceScoreVectors(nullptr), deviceOutput(nullptr),
      deviceAlleleFreqs(nullptr), deviceMissingRates(nullptr), hostOutput(nullptr),
      hostAlleleFreqs(nullptr), hostMissingRates(nullptr) {
      if (!Nstride || !maxVariants)
        throw std::runtime_error("Invalid CUDA Stage 2 dimensions");
      checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "create CUDA Stage 2 stream");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceOutput),
                           maxVariants * (CUDA_STAGE2_MAX_SCORE_VECTORS+1) *
                           sizeof(*deviceOutput)), "cudaMalloc Stage 2 products");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceAlleleFreqs),
                           maxVariants * sizeof(*deviceAlleleFreqs)),
                "cudaMalloc Stage 2 allele frequencies");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceMissingRates),
                           maxVariants * sizeof(*deviceMissingRates)),
                "cudaMalloc Stage 2 missing rates");
      checkCuda(cudaMallocHost(reinterpret_cast<void **>(&hostOutput),
                               maxVariants * (CUDA_STAGE2_MAX_SCORE_VECTORS+1) *
                               sizeof(*hostOutput)), "cudaMallocHost Stage 2 products");
      checkCuda(cudaMallocHost(reinterpret_cast<void **>(&hostAlleleFreqs),
                               maxVariants * sizeof(*hostAlleleFreqs)),
                "cudaMallocHost Stage 2 allele frequencies");
      checkCuda(cudaMallocHost(reinterpret_cast<void **>(&hostMissingRates),
                               maxVariants * sizeof(*hostMissingRates)),
                "cudaMallocHost Stage 2 missing rates");

      int device = 0;
      cudaDeviceProp properties;
      checkCuda(cudaGetDevice(&device), "cudaGetDevice for Stage 2");
      checkCuda(cudaGetDeviceProperties(&properties, device),
                "cudaGetDeviceProperties for Stage 2");
      std::cout << "CUDA Stage 2 enabled on " << properties.name
                << " (up to " << maxVariants << " packed variants per batch)"
                << std::endl;
    }

    ~Impl() {
      if (stream) cudaStreamSynchronize(stream);
      cudaFreeHost(hostMissingRates);
      cudaFreeHost(hostAlleleFreqs);
      cudaFreeHost(hostOutput);
      cudaFreeHost(hostPackedGenotypes);
      cudaFree(deviceMissingRates);
      cudaFree(deviceAlleleFreqs);
      cudaFree(deviceOutput);
      cudaFree(deviceScoreVectors);
      cudaFree(devicePackedGenotypes);
      cudaFree(deviceInputToModel);
      if (stream) cudaStreamDestroy(stream);
    }

    void configureSamples(const int inputToModel[], uint64 count) {
      if (!count || count > static_cast<uint64>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Invalid CUDA Stage 2 input sample count");
      cudaFreeHost(hostPackedGenotypes);
      cudaFree(devicePackedGenotypes);
      cudaFree(deviceInputToModel);
      hostPackedGenotypes = nullptr;
      devicePackedGenotypes = nullptr;
      deviceInputToModel = nullptr;

      inputSamples = count;
      bytesPerVariant = (count+3) >> 2;
      includedSamples = 0;
      identityMapping = inputToModel == nullptr;
      if (inputToModel) {
        identityMapping = count <= Nstride;
        for (uint64 input = 0; input < count; input++) {
          if (inputToModel[input] >= 0) includedSamples++;
          if (inputToModel[input] != static_cast<int>(input)) identityMapping = false;
        }
      }
      else {
        if (count > Nstride)
          throw std::runtime_error("CUDA Stage 2 identity sample mapping exceeds Nstride");
        includedSamples = count;
      }
      const uint64 packedBytes = maxVariants * bytesPerVariant;
      checkCuda(cudaMallocHost(reinterpret_cast<void **>(&hostPackedGenotypes), packedBytes),
                "cudaMallocHost Stage 2 packed genotypes");
      checkCuda(cudaMalloc(reinterpret_cast<void **>(&devicePackedGenotypes), packedBytes),
                "cudaMalloc Stage 2 packed genotypes");
      if (!identityMapping) {
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceInputToModel),
                             count * sizeof(*deviceInputToModel)),
                  "cudaMalloc Stage 2 sample mapping");
        checkCuda(cudaMemcpy(deviceInputToModel, inputToModel,
                             count * sizeof(*deviceInputToModel), cudaMemcpyHostToDevice),
                  "copy Stage 2 sample mapping to CUDA");
      }
      std::cout << "CUDA Stage 2 sample mapping: " << includedSamples << "/"
                << inputSamples << " input samples included"
                << (identityMapping ? " (identity order)" : "") << std::endl;
    }

    void setScoreVectors(const double scoreVectors[], uint64 K) {
      if (!CudaStage2::supportsScoreVectors(K))
        throw std::runtime_error("CUDA Stage 2 supports 1-32 score vectors");
      const uint64 elements = K * Nstride;
      if (elements > scoreVectorCapacity) {
        cudaFree(deviceScoreVectors);
        deviceScoreVectors = nullptr;
        checkCuda(cudaMalloc(reinterpret_cast<void **>(&deviceScoreVectors),
                             elements * sizeof(*deviceScoreVectors)),
                  "cudaMalloc Stage 2 score vectors");
        scoreVectorCapacity = elements;
      }
      checkCuda(cudaMemcpyAsync(deviceScoreVectors, scoreVectors,
                                elements * sizeof(*deviceScoreVectors),
                                cudaMemcpyHostToDevice, stream),
                "copy Stage 2 score vectors to CUDA");
      numScoreVectors = K;
    }

    void scorePacked(double products[], double rawNorm2s[], double alleleFreqs[],
                     double missingRates[], uint64 variants, bool pgenEncoding,
                     uint64 packedVariantOffset) {
      if (!inputSamples || !hostPackedGenotypes)
        throw std::runtime_error("CUDA Stage 2 sample mapping was not configured");
      if (!numScoreVectors || !deviceScoreVectors)
        throw std::runtime_error("CUDA Stage 2 score vectors were not configured");
      if (!variants || packedVariantOffset+variants > maxVariants)
        throw std::runtime_error("Invalid CUDA Stage 2 packed batch size");
      const uint64 packedBytes = variants * bytesPerVariant;
      const uint64 outputValues = variants * (numScoreVectors+1);
      checkCuda(cudaMemcpyAsync(devicePackedGenotypes,
                                hostPackedGenotypes + packedVariantOffset*bytesPerVariant,
                                packedBytes,
                                cudaMemcpyHostToDevice, stream),
                "copy packed Stage 2 genotypes to CUDA");
      computePackedStats<<<static_cast<unsigned int>(variants), 256, 0, stream>>>
        (devicePackedGenotypes, deviceInputToModel, inputSamples, includedSamples,
         bytesPerVariant, pgenEncoding, identityMapping, deviceAlleleFreqs,
         deviceMissingRates);
      checkCuda(cudaGetLastError(), "launch CUDA Stage 2 hardcall statistics");
      launchScore(numScoreVectors, devicePackedGenotypes, deviceInputToModel,
                  inputSamples, bytesPerVariant, pgenEncoding, identityMapping,
                  deviceAlleleFreqs, deviceScoreVectors, Nstride, deviceOutput,
                  variants, stream);
      checkCuda(cudaGetLastError(), "launch CUDA Stage 2 packed scoring");
      checkCuda(cudaMemcpyAsync(hostOutput, deviceOutput,
                                outputValues * sizeof(*hostOutput),
                                cudaMemcpyDeviceToHost, stream),
                "copy CUDA Stage 2 products to host");
      checkCuda(cudaMemcpyAsync(hostAlleleFreqs, deviceAlleleFreqs,
                                variants * sizeof(*hostAlleleFreqs),
                                cudaMemcpyDeviceToHost, stream),
                "copy CUDA Stage 2 allele frequencies to host");
      checkCuda(cudaMemcpyAsync(hostMissingRates, deviceMissingRates,
                                variants * sizeof(*hostMissingRates),
                                cudaMemcpyDeviceToHost, stream),
                "copy CUDA Stage 2 missing rates to host");
      checkCuda(cudaStreamSynchronize(stream), "finish CUDA Stage 2 packed scoring");

      for (uint64 variant = 0; variant < variants; variant++) {
        const double *source = hostOutput + variant * (numScoreVectors+1);
        std::copy(source, source+numScoreVectors,
                  products + variant*numScoreVectors);
        rawNorm2s[variant] = source[numScoreVectors];
        alleleFreqs[variant] = hostAlleleFreqs[variant];
        missingRates[variant] = hostMissingRates[variant];
      }
    }
  };

  CudaStage2::CudaStage2(uint64 Nstride, uint64 maxVariants) :
    impl(new Impl(Nstride, maxVariants)) {}

  CudaStage2::~CudaStage2() { delete impl; }

  bool CudaStage2::supportsScoreVectors(uint64 count) {
    return count >= 1 && count <= CUDA_STAGE2_MAX_SCORE_VECTORS;
  }

  void CudaStage2::configureSamples(const int inputToModel[], uint64 inputSampleCount) {
    impl->configureSamples(inputToModel, inputSampleCount);
  }

  uint64 CudaStage2::getMaxVariants(void) const { return impl->maxVariants; }

  uint64 CudaStage2::getPackedBytesPerVariant(void) const {
    return impl->bytesPerVariant;
  }

  uchar *CudaStage2::getHostPackedBuffer(void) { return impl->hostPackedGenotypes; }

  void CudaStage2::setScoreVectors(const double scoreVectors[], uint64 count) {
    impl->setScoreVectors(scoreVectors, count);
  }

  void CudaStage2::scorePacked(double products[], double rawNorm2s[],
                               double alleleFreqs[], double missingRates[],
                               uint64 variantCount, bool pgenEncoding,
                               uint64 packedVariantOffset) {
    impl->scorePacked(products, rawNorm2s, alleleFreqs, missingRates,
                      variantCount, pgenEncoding, packedVariantOffset);
  }

}
