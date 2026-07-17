/*
   This file is part of the BOLT-LMM linear mixed model software package
   developed by Po-Ru Loh.  Copyright (C) 2014-2025 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/

#ifndef CUDASTEP1_HPP
#define CUDASTEP1_HPP

#include "Types.hpp"

namespace LMM {

  class CudaStep1 {
  private:
    struct Impl;
    Impl *impl;

  public:
    static void setPackedCacheLimitGiB(double limitGiB);
    static void setPackedHostCacheLimitGiB(double limitGiB);

    static void initializeMarkers(const uchar hostGenotypes[], const double maskIndivs[],
                                  const double covBasis[], uint64 Cindep,
                                  double (*snpValueLookup)[4],
                                  double snpCovBasisNegComps[], double Xnorm2s[],
                                  uchar projMaskSnps[], uint64 Nused, uint64 M,
                                  uint64 Nstride, uint64 Cstride,
                                  bool hostGenotypesFileBacked=false);

    CudaStep1(const uchar genotypes[], const double maskIndivs[],
              const double (*snpValueLookup)[4], const double snpCovBasisNegComps[],
              const uchar projMaskSnps[], uint64 M, uint64 Nstride, uint64 Cstride,
              bool hostGenotypesFileBacked=false);
    ~CudaStep1();

    void multXXtransMask(double outCovCompVecs[], const double inCovCompVecs[],
                         const uchar batchMaskSnps[], uint64 B);
    void multX(double outCovCompVecs[], const double coefficients[], uint64 B,
               bool applyIndivMask, bool positiveCovariateComponents);
    void multXtrans(double outSnpProducts[], const double inCovCompVecs[], uint64 B);

    void beginBayesIteration(const double yResidCovCompVecs[],
                             const uchar activeMaskSnps[], uint64 B, uint64 Bleft);
    void computeBayesBlock(double XtransResids[], double snpDots[], uint64 snpDotsStride,
                           uint64 m0, uint64 blockSize, uint64 B, uint64 Bleft,
                           bool computeSnpDots);
    void updateBayesResidual(const double betaBlockUpdates[], uint64 blockSize,
                             uint64 B, uint64 Bleft);
    void endBayesIteration(double yResidCovCompVecs[], uint64 Bleft);
  };

}

#endif
