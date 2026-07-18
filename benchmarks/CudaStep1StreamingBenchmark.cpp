#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CudaStep1.hpp"

namespace {

  uint64 parsePositive(const char *text, const char *name) {
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (!text[0] || !end || *end || !value)
      throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    return static_cast<uint64>(value);
  }

}

int main(int argc, char **argv) {
  bool stream = false;
  uint64 M = 8192;
  uint64 iterations = 20;
  for (int arg = 1; arg < argc; arg++) {
    const std::string option(argv[arg]);
    if (option == "--stream")
      stream = true;
    else if (option == "--variants" && arg + 1 < argc)
      M = parsePositive(argv[++arg], "variant count");
    else if (option == "--iterations" && arg + 1 < argc)
      iterations = parsePositive(argv[++arg], "iteration count");
    else {
      std::cerr << "Usage: " << argv[0]
                << " [--stream] [--variants M] [--iterations I]" << std::endl;
      return 2;
    }
  }

  try {
    const uint64 Nstride = 500000;
    const uint64 Cstride = 8;
    const uint64 NCstride = Nstride + Cstride;
    const uint64 B = 18;
    const uint64 snpsPerBlock = 512;
    const uint64 bytesPerSnp = Nstride >> 2;

    if (stream) {
      LMM::CudaStep1::setPackedCacheLimitGiB(0);
      LMM::CudaStep1::setPackedHostCacheLimitGiB(2);
    }

    std::vector<uchar> packed(M * bytesPerSnp);
    for (uint64 i = 0; i < packed.size(); i++)
      packed[i] = static_cast<uchar>((i * 1315423911ULL + (i >> 7)) & 255);

    std::vector<double> maskIndivs(Nstride, 1.0);
    std::vector<double> lookup(M * 4);
    std::vector<double> negCovComps(M * Cstride, 0.0);
    std::vector<uchar> projMaskSnps(M, 1);
    for (uint64 m = 0; m < M; m++) {
      lookup[m * 4] = -1.0;
      lookup[m * 4 + 1] = 0.0;
      lookup[m * 4 + 2] = 1.0;
      lookup[m * 4 + 3] = 0.0;
    }

    std::vector<double> residuals(B * NCstride);
    for (uint64 i = 0; i < residuals.size(); i++)
      residuals[i] = static_cast<double>(static_cast<int>(i % 257) - 128) * 1e-5;
    std::vector<uchar> activeMask(M, 1);
    std::vector<double> xtrans(snpsPerBlock * B);
    std::vector<double> betaUpdates(snpsPerBlock * B, 0.0);

    LMM::CudaStep1 cuda(packed.data(), maskIndivs.data(),
                        reinterpret_cast<const double (*)[4]>(lookup.data()),
                        negCovComps.data(), projMaskSnps.data(), M, Nstride,
                        Cstride, stream);
    cuda.beginBayesIteration(residuals.data(), activeMask.data(), B, B);

    const auto start = std::chrono::steady_clock::now();
    for (uint64 iteration = 0; iteration < iterations; iteration++) {
      for (uint64 m0 = 0; m0 < M; m0 += snpsPerBlock) {
        const uint64 blockSize = std::min<uint64>(snpsPerBlock, M - m0);
        cuda.computeBayesBlock(xtrans.data(), nullptr, snpsPerBlock, m0,
                               blockSize, B, B, false);
        cuda.updateBayesResidual(betaUpdates.data(), blockSize, B, B);
      }
      cuda.endBayesIteration(residuals.data(), B);
    }
    const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();

    double checksum = 0;
    for (uint64 i = 0; i < residuals.size(); i += 4093)
      checksum += residuals[i];
    std::cout << std::fixed << std::setprecision(6)
              << "mode=" << (stream ? "host-streamed" : "device-cached")
              << " N=" << Nstride << " M=" << M << " B=" << B
              << " iterations=" << iterations << " total_seconds=" << seconds
              << " seconds_per_iteration=" << seconds / iterations
              << " checksum=" << checksum << std::endl;
  }
  catch (const std::exception &error) {
    std::cerr << "CUDA Step 1 streaming benchmark failed: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
