/*
   This file is part of the BOLT-LMM linear mixed model software package
   developed by Po-Ru Loh.  Copyright (C) 2014-2025 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/

#ifndef CUDASTAGE2_HPP
#define CUDASTAGE2_HPP

#include "Types.hpp"

namespace LMM {

  class CudaStage2 {
  private:
    struct Impl;
    Impl *impl;

  public:
    CudaStage2(uint64 Nstride, uint64 maxVariants=256);
    ~CudaStage2();

    static bool supportsScoreVectors(uint64 numScoreVectors);

    // inputToModel[input sample] is the corresponding Stage 1 model index,
    // or -1 when that input sample is not part of the analysis.  Passing NULL
    // selects identity mapping for inputSampleCount samples.
    void configureSamples(const int inputToModel[], uint64 inputSampleCount);
    uint64 getMaxVariants(void) const;
    uint64 getPackedBytesPerVariant(void) const;
    uchar *getHostPackedBuffer(void);

    void setScoreVectors(const double scoreVectors[], uint64 numScoreVectors);

    // Scores the first variantCount packed variants in getHostPackedBuffer().
    // products is variant-major with numScoreVectors entries per variant.
    void scorePacked(double products[], double rawNorm2s[], double alleleFreqs[],
                     double missingRates[], uint64 variantCount, bool pgenEncoding);
  };

}

#endif
