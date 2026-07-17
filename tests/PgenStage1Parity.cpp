#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "PgenUtils.hpp"
#include "SnpData.hpp"

namespace {

  int fail(const std::string &message) {
    std::cerr << "PGEN Stage 1 parity failure: " << message << std::endl;
    return 1;
  }

}

int main(int argc, char **argv) {
  if (argc != 3) return fail("expected fixture and build directory arguments");

  const uchar packedInput[2] = {0xe4, 0x0e};
  const uchar expectedBed[2] = {0x4b, 0x34};
  uchar convertedBed[2] = {};
  std::vector<uint32_t> missingIndices;
  const LMM::PgenUtils::PackedHardcallStats packedStats =
    LMM::PgenUtils::packedPgenToBedAndCollectMissing(
      convertedBed, packedInput, 7, 8, missingIndices);
  const std::vector<uint32_t> expectedMissing = {3, 5};
  if (std::memcmp(convertedBed, expectedBed, sizeof(expectedBed)) != 0 ||
      packedStats.alleleSum != 5 || packedStats.numMissing != 2 ||
      missingIndices != expectedMissing)
    return fail("fused packed hardcall conversion/statistics mismatch");

  uchar exhaustiveInput[256], exhaustiveBed[256] = {}, expectedExhaustiveBed[256] = {};
  uint64 exhaustiveAlleleSum = 0, exhaustiveMissingCount = 0;
  std::vector<uint32_t> expectedExhaustiveMissing;
  const uchar pgenToBed[4] = {3, 2, 0, 1};
  for (uint32_t byte = 0; byte < 256; byte++) {
    exhaustiveInput[byte] = static_cast<uchar>(byte);
    for (uint32_t offset = 0; offset < 4; offset++) {
      const uchar genotype = (byte >> (2*offset)) & 3;
      expectedExhaustiveBed[byte] |= pgenToBed[genotype] << (2*offset);
      if (genotype == 3) {
        exhaustiveMissingCount++;
        expectedExhaustiveMissing.push_back((byte << 2) + offset);
      }
      else
        exhaustiveAlleleSum += genotype;
    }
  }
  const LMM::PgenUtils::PackedHardcallStats exhaustiveStats =
    LMM::PgenUtils::packedPgenToBedAndCollectMissing(
      exhaustiveBed, exhaustiveInput, 1024, 1024, missingIndices);
  if (std::memcmp(exhaustiveBed, expectedExhaustiveBed, sizeof(exhaustiveBed)) != 0 ||
      exhaustiveStats.alleleSum != exhaustiveAlleleSum ||
      exhaustiveStats.numMissing != exhaustiveMissingCount ||
      missingIndices != expectedExhaustiveMissing)
    return fail("exhaustive packed hardcall conversion/statistics mismatch");

  const std::string dir = argv[1];
  const std::string prefix = dir + "/example";
  const std::string persistentCache = std::string(argv[2]) + "/pgen-stage0-parity.cache";
  std::remove(persistentCache.c_str());
  const std::vector<std::string> empty;
  const std::vector<std::string> removeFiles(1, dir + "/remove.txt");

  const std::string selectedPvar = std::string(argv[2]) + "/pvar-selected-fields.pvar";
  {
    std::ofstream out(selectedPvar.c_str());
    out << "#CHROM\tINFO\tALT\tID\tPOS\tREF\tCM\tFILTER\n";
    out << "1\t" << std::string(4096, 'x') << "\tG\trs1\t101\tA\t2.5\tPASS\n";
    out << "2\t" << std::string(4096, 'y') << "\tT\trs2\t202\tC\t.\tPASS\n";
  }
  const std::vector<LMM::SnpInfo> selectedVariants =
    LMM::PgenUtils::readPvarFile(selectedPvar, 22, NULL);
  std::remove(selectedPvar.c_str());
  if (selectedVariants.size() != 2 || selectedVariants[0].ID != "rs1" ||
      selectedVariants[0].chrom != 1 || selectedVariants[0].physpos != 101 ||
      selectedVariants[0].genpos != 0.025 || selectedVariants[0].allele1 != "G" ||
      selectedVariants[0].allele2 != "A" || selectedVariants[1].ID != "rs2" ||
      selectedVariants[1].chrom != 2 || selectedVariants[1].physpos != 202 ||
      selectedVariants[1].genpos != 0 || selectedVariants[1].allele1 != "T" ||
      selectedVariants[1].allele2 != "C")
    return fail("selected-field PVAR parsing mismatch");

  LMM::SnpData bed(prefix + ".fam", std::vector<std::string>(1, prefix + ".bim"),
                   std::vector<std::string>(1, prefix + ".bed"), "", empty, empty,
                   removeFiles, 1.0, 1.0, false);
  LMM::SnpData pgen(prefix + ".pgen", prefix + ".pvar", prefix + ".psam", "",
                    empty, empty, removeFiles, 1.0, 1.0, false, empty, true, 22, dir);
  {
    LMM::SnpData cacheBuilder(prefix + ".pgen", prefix + ".pvar", prefix + ".psam", "",
                              empty, empty, removeFiles, 1.0, 1.0, false, empty, true, 22,
                              "", persistentCache, true);
  }
  LMM::SnpData cached(prefix + ".pgen", prefix + ".pvar", prefix + ".psam", "",
                      empty, empty, removeFiles, 1.0, 1.0, false, empty, true, 22,
                      "", persistentCache, false);

  if (!pgen.getGenotypesFileBacked())
    return fail("explicit cache directory did not select file-backed storage");
  if (!cached.getGenotypesFileBacked())
    return fail("persistent cache did not select file-backed storage");

  if (bed.getM() != pgen.getM()) return fail("post-QC variant counts differ");
  if (bed.getNstride() != pgen.getNstride()) return fail("sample strides differ");
  if (bed.getNumIndivsQC() != pgen.getNumIndivsQC())
    return fail("post-QC sample counts differ");
  if (bed.getIndivIds() != pgen.getIndivIds()) return fail("sample IDs differ");
  if (bed.getBedSnpToGrmIndex() != pgen.getBedSnpToGrmIndex())
    return fail("variant index mappings differ");

  const size_t genotypeBytes = bed.getM() * bed.getNstride()/4;
  if (std::memcmp(bed.getGenotypes(), pgen.getGenotypes(), genotypeBytes) != 0)
    return fail("packed hardcalls differ");
  if (std::memcmp(bed.getGenotypes(), cached.getGenotypes(), genotypeBytes) != 0)
    return fail("persistent-cache packed hardcalls differ");

  std::vector<double> bedMask(bed.getNstride()), pgenMask(pgen.getNstride());
  bed.writeMaskIndivs(bedMask.data());
  pgen.writeMaskIndivs(pgenMask.data());
  if (bedMask != pgenMask) return fail("sample masks differ");
  cached.writeMaskIndivs(pgenMask.data());
  if (bedMask != pgenMask) return fail("persistent-cache sample masks differ");

  const std::vector<LMM::SnpInfo> &bedSnps = bed.getSnpInfo();
  const std::vector<LMM::SnpInfo> &pgenSnps = pgen.getSnpInfo();
  for (size_t m = 0; m < bedSnps.size(); m++) {
    const LMM::SnpInfo &lhs = bedSnps[m];
    const LMM::SnpInfo &rhs = pgenSnps[m];
    if (lhs.chrom != rhs.chrom || lhs.ID != rhs.ID || lhs.genpos != rhs.genpos ||
        lhs.physpos != rhs.physpos || lhs.allele1 != rhs.allele1 ||
        lhs.allele2 != rhs.allele2 || lhs.MAF != rhs.MAF || lhs.vcNum != rhs.vcNum)
      return fail("variant metadata differs at index " + std::to_string(m));
  }

  if (cached.getM() != pgen.getM() || cached.getNstride() != pgen.getNstride() ||
      cached.getNumIndivsQC() != pgen.getNumIndivsQC() ||
      cached.getIndivIds() != pgen.getIndivIds() ||
      cached.getBedSnpToGrmIndex() != pgen.getBedSnpToGrmIndex() ||
      cached.getSnpInfo().size() != pgenSnps.size())
    return fail("persistent-cache metadata differs");
  for (size_t m = 0; m < pgenSnps.size(); m++) {
    const LMM::SnpInfo &lhs = pgenSnps[m];
    const LMM::SnpInfo &rhs = cached.getSnpInfo()[m];
    if (lhs.chrom != rhs.chrom || lhs.ID != rhs.ID || lhs.genpos != rhs.genpos ||
        lhs.physpos != rhs.physpos || lhs.allele1 != rhs.allele1 ||
        lhs.allele2 != rhs.allele2 || lhs.MAF != rhs.MAF || lhs.vcNum != rhs.vcNum)
      return fail("persistent-cache variant metadata differs at index " + std::to_string(m));
  }

  std::remove(persistentCache.c_str());

  std::cout << "BED/PGEN/persistent-cache Stage 1 hardcalls are byte-identical after sample removal ("
            << bed.getM() << " variants)" << std::endl;
  return 0;
}
