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
  const std::vector<std::string> noFiles;
  bedBolt.streamComputeRetroLOCO(bedOut, std::vector<std::string>(1, prefix + ".bim"),
                                 std::vector<std::string>(1, prefix + ".bed"), "",
                                 noFiles, 1.0, true, retroData);
  pgenBolt.streamComputeRetroLOCOPgen(pgenOut, prefix + ".pgen", prefix + ".pvar",
                                      "", noFiles, 1.0, true, retroData);

  const std::string bedStats = readFile(bedOut);
  const std::string pgenStats = readFile(pgenOut);
  std::remove(bedOut.c_str());
  std::remove(pgenOut.c_str());
  if (bedStats.empty()) return fail("BED association output is empty");
  if (bedStats != pgenStats) return fail("BED and PGEN association outputs differ");

  std::cout << "BED/PGEN Stage 2 association output is byte-identical after subsetting, "
            << "reordering, and masking samples" << std::endl;
  return 0;
}
