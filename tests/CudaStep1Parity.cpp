#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

#include "CudaStep1.hpp"

namespace {

  uchar encodeGenotype(int genotype) {
    if (genotype == 0) return 3;
    if (genotype == 1) return 2;
    if (genotype == 2) return 0;
    return 1;
  }

}

int main() {
  const uint64 M = 5, Nstride = 8, Cstride = 4, NCstride = Nstride + Cstride, B = 3;
  const int genotypesUnpacked[M][Nstride] = {
    {0, 1, 2, 0, 1, 2, 0, 1},
    {2, 2, 1, 0, 0, 1, 2, 9},
    {1, 0, 1, 2, 1, 0, 1, 2},
    {0, 0, 0, 1, 1, 1, 2, 2},
    {2, 1, 0, 9, 2, 1, 0, 2}
  };

  std::vector<uchar> packed(M * (Nstride >> 2));
  for (uint64 m = 0; m < M; m++)
    for (uint64 n4 = 0; n4 < Nstride; n4 += 4) {
      uchar byte = 0;
      for (uint64 k = 0; k < 4; k++)
        byte |= encodeGenotype(genotypesUnpacked[m][n4+k]) << (2*k);
      packed[m*(Nstride>>2) + (n4>>2)] = byte;
    }

  const double maskIndivs[Nstride] = {1, 1, 0, 1, 1, 0, 1, 1};
  double lookup[M][4];
  double negCovComps[M*Cstride];
  for (uint64 m = 0; m < M; m++) {
    lookup[m][0] = -0.7 + 0.03*m;
    lookup[m][1] = 0.2 + 0.02*m;
    lookup[m][2] = 1.1 + 0.01*m;
    lookup[m][3] = 0;
    for (uint64 c = 0; c < Cstride; c++)
      negCovComps[m*Cstride+c] = 0.05 * (m+1) * (c+1);
  }
  const uchar projMaskSnps[M] = {1, 1, 0, 1, 1};
  const uchar batchMaskSnps[M*B] = {
    1, 1, 0,
    1, 0, 1,
    1, 1, 1,
    0, 1, 1,
    1, 0, 1
  };

  std::vector<double> input(B*NCstride);
  for (uint64 b = 0; b < B; b++)
    for (uint64 nc = 0; nc < NCstride; nc++)
      input[b*NCstride+nc] = 0.01 * (1+b) * (1+nc) - 0.08;

  std::vector<double> expected(B*NCstride, 0), xNeg(M*NCstride, 0), xtrans(M*B, 0);
  for (uint64 m = 0; m < M; m++) {
    if (!projMaskSnps[m])
      continue;
    for (uint64 n = 0; n < Nstride; n++) {
      const int genotype = genotypesUnpacked[m][n];
      xNeg[m*NCstride+n] = lookup[m][genotype == 9 ? 3 : genotype] * maskIndivs[n];
    }
    for (uint64 c = 0; c < Cstride; c++)
      xNeg[m*NCstride+Nstride+c] = negCovComps[m*Cstride+c];

    for (uint64 b = 0; b < B; b++) {
      double dot = 0;
      for (uint64 nc = 0; nc < NCstride; nc++)
        dot += input[b*NCstride+nc] * xNeg[m*NCstride+nc];
      xtrans[m*B+b] = dot * batchMaskSnps[m*B+b];
      for (uint64 nc = 0; nc < NCstride; nc++) {
        const double xPositive = nc < Nstride ? xNeg[m*NCstride+nc] :
                                 -xNeg[m*NCstride+nc];
        expected[b*NCstride+nc] += xPositive * xtrans[m*B+b];
      }
    }
  }

  std::vector<double> observed(B*NCstride);
  LMM::CudaStep1 cuda(packed.data(), maskIndivs, lookup, negCovComps, projMaskSnps,
                      M, Nstride, Cstride);
  cuda.multXXtransMask(observed.data(), input.data(), batchMaskSnps, B);

  double maxAbsError = 0;
  for (uint64 i = 0; i < observed.size(); i++)
    maxAbsError = std::max(maxAbsError, std::fabs(observed[i] - expected[i]));
  if (maxAbsError > 1e-11) {
    std::cerr << "CUDA Step 1 parity failure: max absolute error = " << maxAbsError << std::endl;
    return 1;
  }

  std::vector<double> coefficients(M*B), expectedMultX(B*NCstride, 0);
  std::vector<double> expectedMultXAll(B*NCstride, 0), observedMultX(B*NCstride);
  std::vector<double> xAllNeg(M*NCstride, 0);
  for (uint64 m = 0; m < M; m++) {
    for (uint64 b = 0; b < B; b++)
      coefficients[m*B+b] = 0.006 * (1+m) * (1+b) - 0.013;
    if (!projMaskSnps[m])
      continue;
    for (uint64 n = 0; n < Nstride; n++) {
      const int genotype = genotypesUnpacked[m][n];
      xAllNeg[m*NCstride+n] = lookup[m][genotype == 9 ? 3 : genotype];
    }
    for (uint64 c = 0; c < Cstride; c++)
      xAllNeg[m*NCstride+Nstride+c] = negCovComps[m*Cstride+c];
    for (uint64 b = 0; b < B; b++)
      for (uint64 nc = 0; nc < NCstride; nc++) {
        const double xPositive = nc < Nstride ? xNeg[m*NCstride+nc] :
                                 -xNeg[m*NCstride+nc];
        expectedMultX[b*NCstride+nc] += xPositive * coefficients[m*B+b];
        expectedMultXAll[b*NCstride+nc] +=
          xAllNeg[m*NCstride+nc] * coefficients[m*B+b];
      }
  }

  cuda.multX(observedMultX.data(), coefficients.data(), B, true, true);
  double maxMultXError = 0;
  for (uint64 i = 0; i < observedMultX.size(); i++)
    maxMultXError = std::max(maxMultXError,
                             std::fabs(observedMultX[i] - expectedMultX[i]));
  cuda.multX(observedMultX.data(), coefficients.data(), B, false, false);
  for (uint64 i = 0; i < observedMultX.size(); i++)
    maxMultXError = std::max(maxMultXError,
                             std::fabs(observedMultX[i] - expectedMultXAll[i]));
  if (maxMultXError > 1e-11) {
    std::cerr << "CUDA Step 1 X beta parity failure: max absolute error = "
              << maxMultXError << std::endl;
    return 1;
  }

  const uchar activeMaskSnps[M] = {1, 0, 1, 1, 1};
  const uint64 Bleft = 2;
  std::vector<double> residual = input, expectedResidual = input;
  std::vector<double> bayesXtrans(M*B), expectedBayesXtrans(M*B, 0);
  std::vector<double> snpDots(M*M), expectedSnpDots(M*M, 0);
  std::vector<double> betaUpdates(M*B);
  for (uint64 m = 0; m < M; m++)
    for (uint64 b = 0; b < B; b++)
      betaUpdates[m*B+b] = 0.004 * (1+m) * (1+b) - 0.01;

  for (uint64 m = 0; m < M; m++) {
    if (!activeMaskSnps[m] || !projMaskSnps[m])
      continue;
    for (uint64 b = 0; b < Bleft; b++)
      for (uint64 nc = 0; nc < NCstride; nc++)
        expectedBayesXtrans[m*B+b] += residual[b*NCstride+nc] * xNeg[m*NCstride+nc];
    for (uint64 m2 = 0; m2 < M; m2++) {
      if (!activeMaskSnps[m2] || !projMaskSnps[m2])
        continue;
      for (uint64 n = 0; n < Nstride; n++)
        expectedSnpDots[m*M+m2] += xNeg[m*NCstride+n] * xNeg[m2*NCstride+n];
      for (uint64 c = 0; c < Cstride; c++)
        expectedSnpDots[m*M+m2] -=
          xNeg[m*NCstride+Nstride+c] * xNeg[m2*NCstride+Nstride+c];
    }
    for (uint64 b = 0; b < Bleft; b++)
      for (uint64 nc = 0; nc < NCstride; nc++) {
        const double xPositive = nc < Nstride ? xNeg[m*NCstride+nc] :
                                 -xNeg[m*NCstride+nc];
        expectedResidual[b*NCstride+nc] += xPositive * betaUpdates[m*B+b];
      }
  }

  cuda.beginBayesIteration(residual.data(), activeMaskSnps, B);
  cuda.computeBayesBlock(bayesXtrans.data(), snpDots.data(), M, 0, M, B, Bleft, true);
  cuda.updateBayesResidual(betaUpdates.data(), M, B, Bleft);
  cuda.endBayesIteration(residual.data(), B);

  double maxBayesError = 0;
  for (uint64 m = 0; m < M; m++)
    for (uint64 b = 0; b < Bleft; b++)
      maxBayesError = std::max(maxBayesError,
                               std::fabs(bayesXtrans[m*B+b] - expectedBayesXtrans[m*B+b]));
  for (uint64 i = 0; i < snpDots.size(); i++)
    maxBayesError = std::max(maxBayesError, std::fabs(snpDots[i] - expectedSnpDots[i]));
  for (uint64 i = 0; i < residual.size(); i++)
    maxBayesError = std::max(maxBayesError, std::fabs(residual[i] - expectedResidual[i]));
  if (maxBayesError > 1e-11) {
    std::cerr << "CUDA Step 1 Bayesian parity failure: max absolute error = "
              << maxBayesError << std::endl;
    return 1;
  }

  std::cout << "CUDA Step 1 parity max absolute errors: projected multiply = "
            << maxAbsError << ", X beta = " << maxMultXError
            << ", Bayesian iteration = " << maxBayesError << std::endl;
  return 0;
}
