/*
   This file is part of the BOLT-LMM linear mixed model software package
   developed by Po-Ru Loh.  Copyright (C) 2014-2025 Harvard University.
*/

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

#include "CudaStage2.hpp"

namespace {

  int fail(const std::string &message) {
    std::cerr << "CUDA Stage 2 parity failure: " << message << std::endl;
    return 1;
  }

  unsigned char hardcallCode(int genotype, bool missing, bool pgenEncoding) {
    if (pgenEncoding) return missing ? 3 : static_cast<unsigned char>(genotype);
    if (missing) return 1;
    return genotype == 0 ? 3 : genotype == 1 ? 2 : 0;
  }

  bool closeEnough(double actual, double expected) {
    return std::fabs(actual-expected) <= 2e-10 * (1+std::fabs(expected));
  }

  int runCase(LMM::CudaStage2 &cuda, const std::vector<int> &inputToModel,
              uint64 Nstride, uint64 variants, uint64 scoreCount,
              bool pgenEncoding) {
    const uint64 inputSamples = inputToModel.size();
    const uint64 packedBytes = (inputSamples+3)>>2;
    std::vector<double> scores(scoreCount*Nstride);
    for (uint64 k = 0; k < scoreCount; k++)
      for (uint64 n = 0; n < Nstride; n++)
        scores[k*Nstride+n] = std::sin(0.013*(n+3*k)) +
          0.2*std::cos(0.031*(2*n+k));
    cuda.setScoreVectors(scores.data(), scoreCount);

    unsigned char *packed = cuda.getHostPackedBuffer();
    std::fill(packed, packed+variants*packedBytes, 0);
    std::vector<double> expectedProducts(variants*scoreCount, 0);
    std::vector<double> expectedNorms(variants, 0), expectedFreqs(variants, 0),
      expectedMissing(variants, 0);
    uint64 included = 0;
    for (int model : inputToModel) included += model >= 0;

    for (uint64 variant = 0; variant < variants; variant++) {
      double alleleSum = 0;
      uint64 missingCount = 0;
      for (uint64 input = 0; input < inputSamples; input++) {
        if (inputToModel[input] < 0) continue;
        const int genotype = static_cast<int>((input*7 + variant*5 + input/11) % 3);
        const bool missing = ((input*13 + variant*17) % 97) == 0;
        const unsigned char code = hardcallCode(genotype, missing, pgenEncoding);
        packed[variant*packedBytes + (input>>2)] |= code << (2*(input&3));
        if (missing) missingCount++;
        else alleleSum += genotype;
      }
      const uint64 observed = included-missingCount;
      const double freq = observed ? alleleSum/(2*observed) : 0;
      expectedFreqs[variant] = freq;
      expectedMissing[variant] = static_cast<double>(missingCount)/included;
      const double mean = 2*freq;
      for (uint64 input = 0; input < inputSamples; input++) {
        const int model = inputToModel[input];
        if (model < 0) continue;
        const int genotype = static_cast<int>((input*7 + variant*5 + input/11) % 3);
        const bool missing = ((input*13 + variant*17) % 97) == 0;
        const double centered = missing ? 0 : genotype-mean;
        expectedNorms[variant] += centered*centered;
        for (uint64 k = 0; k < scoreCount; k++)
          expectedProducts[variant*scoreCount+k] += centered*scores[k*Nstride+model];
      }
    }

    std::vector<double> products(variants*scoreCount), norms(variants), freqs(variants),
      missing(variants);
    cuda.scorePacked(products.data(), norms.data(), freqs.data(), missing.data(),
                     variants, pgenEncoding);
    for (uint64 variant = 0; variant < variants; variant++) {
      if (!closeEnough(norms[variant], expectedNorms[variant]) ||
          !closeEnough(freqs[variant], expectedFreqs[variant]) ||
          !closeEnough(missing[variant], expectedMissing[variant])) {
        std::ostringstream message;
        message << "variant summaries differ for variant " << variant
                << ", K=" << scoreCount << ", pgen=" << pgenEncoding;
        return fail(message.str());
      }
      for (uint64 k = 0; k < scoreCount; k++)
        if (!closeEnough(products[variant*scoreCount+k],
                         expectedProducts[variant*scoreCount+k])) {
          std::ostringstream message;
          message << "score differs for variant " << variant << ", vector " << k
                  << ", K=" << scoreCount << ", pgen=" << pgenEncoding;
          return fail(message.str());
        }
    }
    return 0;
  }

}

int main() {
  const uint64 Nstride = 512, inputSamples = 523, variants = 19;
  std::vector<int> inputToModel(inputSamples, -1);
  for (uint64 model = 0; model < 497; model++)
    inputToModel[(model*193 + 17) % inputSamples] = static_cast<int>(model);
  // Model samples can remain present in the input while excluded by the
  // phenotype/covariate mask; represent those with -1 in the GPU mapping.
  for (uint64 input = 0; input < inputSamples; input++)
    if (inputToModel[input] >= 0 && inputToModel[input] % 89 == 0)
      inputToModel[input] = -1;

  LMM::CudaStage2 cuda(Nstride, 32);
  cuda.configureSamples(inputToModel.data(), inputSamples);
  for (uint64 K : std::vector<uint64>{3, 23})
    for (bool pgenEncoding : std::vector<bool>{false, true}) {
      const int result = runCase(cuda, inputToModel, Nstride, variants, K, pgenEncoding);
      if (result) return result;
    }
  std::cout << "CUDA Stage 2 packed-scoring parity checks passed" << std::endl;
  return 0;
}
