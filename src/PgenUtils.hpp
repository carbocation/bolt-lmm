/*
   This file is part of the BOLT-LMM linear mixed model software package
   developed by Po-Ru Loh.  Copyright (C) 2014-2026 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/

#ifndef PGENUTILS_HPP
#define PGENUTILS_HPP

#include <string>
#include <unordered_set>
#include <vector>

#include "SnpInfo.hpp"
#include "Types.hpp"

namespace LMM {
  namespace PgenUtils {

    struct FileSet {
      std::string pgenFile;
      std::string pvarFile;
      std::string psamFile;
    };

    struct SampleInfo {
      std::string famID;
      std::string indivID;
      std::string paternalID;
      std::string maternalID;
      int sex;
      double pheno;
    };

    FileSet resolvePrefix(const std::string &prefix);

    std::vector<SampleInfo> readPsamFile(const std::string &psamFile);

    std::vector<SnpInfo> readPvarFile(
      const std::string &pvarFile, int Nauto,
      const std::unordered_set<std::string> *excludeSnpsPtr = NULL);

    // Converts pgenlib allele-count codes (0, 1, 2, missing) to the PLINK 1
    // BED bit encoding used by SnpData. out has Nstride/4 bytes.
    void packedPgenToBed(uchar out[], const uchar in[], uint64 N, uint64 Nstride);

  }
}

#endif
