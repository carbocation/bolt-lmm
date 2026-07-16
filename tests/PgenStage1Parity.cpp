#include <cmath>
#include <cstring>
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
  if (argc != 2) return fail("expected fixture directory argument");

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

  const std::string dir = argv[1];
  const std::string prefix = dir + "/example";
  const std::vector<std::string> empty;
  const std::vector<std::string> removeFiles(1, dir + "/remove.txt");

  LMM::SnpData bed(prefix + ".fam", std::vector<std::string>(1, prefix + ".bim"),
                   std::vector<std::string>(1, prefix + ".bed"), "", empty, empty,
                   removeFiles, 1.0, 1.0, false);
  LMM::SnpData pgen(prefix + ".pgen", prefix + ".pvar", prefix + ".psam", "",
                    empty, empty, removeFiles, 1.0, 1.0, false, empty, true, 22, dir);

  if (!pgen.getGenotypesFileBacked())
    return fail("explicit cache directory did not select file-backed storage");

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

  std::vector<double> bedMask(bed.getNstride()), pgenMask(pgen.getNstride());
  bed.writeMaskIndivs(bedMask.data());
  pgen.writeMaskIndivs(pgenMask.data());
  if (bedMask != pgenMask) return fail("sample masks differ");

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

  std::cout << "BED/PGEN Stage 1 hardcalls are byte-identical after sample removal ("
            << bed.getM() << " variants)" << std::endl;
  return 0;
}
