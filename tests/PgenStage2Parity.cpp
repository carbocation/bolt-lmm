#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "Bolt.hpp"
#include "FileUtils.hpp"
#include "SnpData.hpp"

namespace {

  int fail(const std::string &message) {
    std::cerr << "PGEN Stage 2 parity failure: " << message << std::endl;
    return 1;
  }

  std::string readFile(const std::string &file) {
    std::ifstream in(file.c_str(), std::ios::in | std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
  }

}

int main(int argc, char **argv) {
  if (argc != 3) return fail("expected fixture and output directory arguments");
  const std::string fixtureDir = argv[1];
  const std::string outputDir = argv[2];
  const std::string prefix = fixtureDir + "/example";

  std::vector< std::pair<std::string, std::string> > ids =
    FileUtils::readFidIids(prefix + ".fam");
  ids.erase(ids.begin() + 497);
  ids.erase(ids.begin() + 16);
  ids.erase(ids.begin());
  std::swap(ids.front(), ids.back());

  const uint64 Nstride = (ids.size()+3)&~3;
  std::vector<double> mask(Nstride, 0);
  for (size_t n = 0; n < ids.size(); n++) mask[n] = 1;
  mask[7] = 0;
  mask[123] = 0;

  LMM::SnpData bedData(ids, mask, Nstride, prefix + ".fam");
  LMM::SnpData pgenData(ids, mask, Nstride, prefix + ".psam", 22, true);

  double Nused = 0;
  for (size_t n = 0; n < mask.size(); n++) Nused += mask[n] != 0;
  std::vector<double> covBasis(Nstride, 0);
  for (size_t n = 0; n < ids.size(); n++)
    if (mask[n]) covBasis[n] = 1/std::sqrt(Nused);

  const std::unordered_set<std::string> noBgenVariants;
  LMM::Bolt bedBolt(bedData, mask, covBasis, 1, 22, noBgenVariants);
  LMM::Bolt pgenBolt(pgenData, mask, covBasis, 1, 22, noBgenVariants);

  std::vector<double> resid(Nstride, 0);
  double residMean = 0;
  for (size_t n = 0; n < ids.size(); n++)
    if (mask[n]) {
      resid[n] = std::sin(0.17*n) + 0.25*std::cos(0.03*n);
      residMean += resid[n];
    }
  residMean /= Nused;
  for (size_t n = 0; n < ids.size(); n++)
    if (mask[n]) resid[n] -= residMean;

  const std::vector<double> noStats;
  const std::vector< std::vector<double> > resids(1, resid);
  const std::vector<LMM::Bolt::SnpChunkBoundary> boundaries(
    1, LMM::Bolt::SnpChunkBoundary(1, 1, 0));
  const std::vector<double> scales(1, 1.0);
  const std::vector<LMM::Bolt::StatsDataRetroLOCO> retroData(
    1, LMM::Bolt::StatsDataRetroLOCO("LINREG", noStats, resids, boundaries, scales));

  const std::string bedOut = outputDir + "/pgen_stage2_bed.stats";
  const std::string pgenOut = outputDir + "/pgen_stage2_pgen.stats";
  const std::string bedReferenceOut = outputDir + "/pgen_stage2_bed_reference.stats";
  const std::string pgenReferenceOut = outputDir + "/pgen_stage2_pgen_reference.stats";
  const std::vector<std::string> noFiles;
  bedBolt.streamComputeRetroLOCO(bedOut, std::vector<std::string>(1, prefix + ".bim"),
                                 std::vector<std::string>(1, prefix + ".bed"), "",
                                 noFiles, 1.0, true, retroData);
  pgenBolt.streamComputeRetroLOCOPgen(pgenOut, prefix + ".pgen", prefix + ".pvar",
                                      "", noFiles, 1.0, true, retroData);
  bedBolt.streamComputeRetroLOCO(bedReferenceOut,
                                 std::vector<std::string>(1, prefix + ".bim"),
                                 std::vector<std::string>(1, prefix + ".bed"), "",
                                 noFiles, 1.0, true, retroData, false);
  pgenBolt.streamComputeRetroLOCOPgen(pgenReferenceOut, prefix + ".pgen",
                                      prefix + ".pvar", "", noFiles, 1.0, true,
                                      retroData, false);

  const std::string bedStats = readFile(bedOut);
  const std::string pgenStats = readFile(pgenOut);
  const std::string bedReferenceStats = readFile(bedReferenceOut);
  const std::string pgenReferenceStats = readFile(pgenReferenceOut);
  std::remove(bedOut.c_str());
  std::remove(pgenOut.c_str());
  std::remove(bedReferenceOut.c_str());
  std::remove(pgenReferenceOut.c_str());
  if (bedStats.empty()) return fail("BED association output is empty");
  if (bedStats != pgenStats) return fail("BED and PGEN association outputs differ");
  if (bedReferenceStats != pgenReferenceStats)
    return fail("scalar BED and PGEN association outputs differ");
  if (bedStats != bedReferenceStats)
    return fail("optimized and scalar Stage 2 association outputs differ");

  // Exercise the identity-order path separately. AVX builds select direct
  // packed scoring here; portable builds retain the dense batched fallback.
  {
    std::vector< std::pair<std::string, std::string> > identityIds =
      FileUtils::readFidIids(prefix + ".fam");
    const uint64 identityStride = (identityIds.size()+3)&~3;
    std::vector<double> identityMask(identityStride, 0);
    for (size_t n = 0; n < identityIds.size(); n++) identityMask[n] = 1;
    std::vector<double> identityBasis(identityStride, 0);
    for (size_t n = 0; n < identityIds.size(); n++)
      identityBasis[n] = 1/std::sqrt(static_cast<double>(identityIds.size()));
    LMM::SnpData identityBedData(identityIds, identityMask, identityStride,
                                  prefix + ".fam");
    LMM::SnpData identityPgenData(identityIds, identityMask, identityStride,
                                   prefix + ".psam", 22, true);
    LMM::Bolt identityBedBolt(identityBedData, identityMask, identityBasis, 1, 22,
                              noBgenVariants);
    LMM::Bolt identityPgenBolt(identityPgenData, identityMask, identityBasis, 1, 22,
                               noBgenVariants);
    std::vector<double> identityResid(identityStride, 0);
    double identityResidMean = 0;
    for (size_t n = 0; n < identityIds.size(); n++) {
      identityResid[n] = std::sin(0.17*n) + 0.25*std::cos(0.03*n);
      identityResidMean += identityResid[n];
    }
    identityResidMean /= identityIds.size();
    for (size_t n = 0; n < identityIds.size(); n++)
      identityResid[n] -= identityResidMean;
    const std::vector< std::vector<double> > identityResids(1, identityResid);
    const std::vector<LMM::Bolt::StatsDataRetroLOCO> identityRetroData(
      1, LMM::Bolt::StatsDataRetroLOCO("LINREG", noStats, identityResids,
                                      boundaries, scales));

    const std::string directBedOut = outputDir + "/pgen_stage2_direct_bed.stats";
    const std::string directPgenOut = outputDir + "/pgen_stage2_direct_pgen.stats";
    const std::string directReferenceOut = outputDir + "/pgen_stage2_direct_reference.stats";
    identityBedBolt.streamComputeRetroLOCO(
      directBedOut, std::vector<std::string>(1, prefix + ".bim"),
      std::vector<std::string>(1, prefix + ".bed"), "", noFiles, 1.0, true,
      identityRetroData);
    identityPgenBolt.streamComputeRetroLOCOPgen(
      directPgenOut, prefix + ".pgen", prefix + ".pvar", "", noFiles, 1.0,
      true, identityRetroData);
    identityBedBolt.streamComputeRetroLOCO(
      directReferenceOut, std::vector<std::string>(1, prefix + ".bim"),
      std::vector<std::string>(1, prefix + ".bed"), "", noFiles, 1.0, true,
      identityRetroData, false);
    const std::string directBedStats = readFile(directBedOut);
    const std::string directPgenStats = readFile(directPgenOut);
    const std::string directReferenceStats = readFile(directReferenceOut);
    std::remove(directBedOut.c_str());
    std::remove(directPgenOut.c_str());
    std::remove(directReferenceOut.c_str());
    if (directBedStats != directPgenStats)
      return fail("direct-packed BED and PGEN association outputs differ");
    if (directBedStats != directReferenceStats)
      return fail("direct-packed and scalar association outputs differ");
  }

  std::cout << "BED/PGEN Stage 2 association output is byte-identical after subsetting, "
            << "reordering, and masking samples" << std::endl;
  return 0;
}
