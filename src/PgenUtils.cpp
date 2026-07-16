/*
   This file is part of the BOLT-LMM linear mixed model software package
   developed by Po-Ru Loh.  Copyright (C) 2014-2026 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/

#include <array>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include "FileUtils.hpp"
#include "PgenUtils.hpp"
#include "SnpData.hpp"
#include "StringUtils.hpp"

namespace LMM {
  namespace PgenUtils {

    using std::cerr;
    using std::endl;
    using std::string;
    using std::vector;
    using FileUtils::getline;

    namespace {

      bool fileReadable(const string &file) {
        std::ifstream fin(file.c_str(), std::ios::in | std::ios::binary);
        return static_cast<bool>(fin);
      }

      string resolveSidecar(const string &prefix, const string &extension,
                            bool allowGzip) {
        const string plain = prefix + extension;
        if (fileReadable(plain)) return plain;
        if (allowGzip) {
          const string gzip = plain + ".gz";
          if (fileReadable(gzip)) return gzip;
        }
        cerr << "ERROR: Unable to find " << extension << " file for --pfile prefix: "
             << prefix << endl;
        cerr << "       Tried: " << plain;
        if (allowGzip) cerr << " and " << plain << ".gz";
        cerr << endl;
        exit(1);
      }

      vector<string> splitFields(string line) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        return StringUtils::tokenizeMultipleDelimiters(line, " \t");
      }

      int findColumn(const vector<string> &header, const string &name) {
        for (uint i = 0; i < header.size(); i++)
          if (header[i] == name) return static_cast<int>(i);
        return -1;
      }

      int parseInt(const string &value, const string &field, const string &file,
                   uint64 lineNum) {
        char *end = NULL;
        errno = 0;
        long parsed = strtol(value.c_str(), &end, 10);
        if (errno || end == value.c_str() || *end != '\0' ||
            parsed < INT_MIN || parsed > INT_MAX) {
          cerr << "ERROR: Invalid " << field << " at line " << lineNum
               << " of " << file << ": " << value << endl;
          exit(1);
        }
        return static_cast<int>(parsed);
      }

      double parseDouble(const string &value, const string &field,
                         const string &file, uint64 lineNum) {
        char *end = NULL;
        errno = 0;
        double parsed = strtod(value.c_str(), &end);
        if (errno || end == value.c_str() || *end != '\0') {
          cerr << "ERROR: Invalid " << field << " at line " << lineNum
               << " of " << file << ": " << value << endl;
          exit(1);
        }
        return parsed;
      }

      int parseSex(const string &value, const string &file, uint64 lineNum) {
        if (value.empty() || value == "0" || value == "NA" || value == "NaN" ||
            value == "nan" || value == ".") return 0;
        int sex = parseInt(value, "SEX", file, lineNum);
        if (sex != 1 && sex != 2) return 0;
        return sex;
      }

      double parsePheno(const string &value) {
        double pheno;
        if (value.empty() || value == "NA" || value == "NaN" || value == "nan" ||
            value == "." || sscanf(value.c_str(), "%lf", &pheno) != 1)
          return -9;
        return pheno;
      }

      SampleInfo parseFamLayoutSample(const vector<string> &fields,
                                      const string &file, uint64 lineNum) {
        if (fields.size() < 6) {
          cerr << "ERROR: Headerless psam file must have at least 6 fam-format fields"
               << " at line " << lineNum << " of " << file << endl;
          exit(1);
        }
        SampleInfo sample;
        sample.famID = fields[0];
        sample.indivID = fields[1];
        sample.paternalID = fields[2];
        sample.maternalID = fields[3];
        sample.sex = parseSex(fields[4], file, lineNum);
        sample.pheno = parsePheno(fields[5]);
        return sample;
      }

      std::array<uchar, 256> makePgenToBedLookup() {
        // pgenlib: 00 = 0 ALT, 01 = 1 ALT, 10 = 2 ALT, 11 = missing.
        // PLINK 1: 11 = 0 A1, 10 = 1 A1, 00 = 2 A1, 01 = missing.
        const uchar pgenCodeToBed[4] = {3, 2, 0, 1};
        std::array<uchar, 256> lookup = {};
        for (uint byte = 0; byte < lookup.size(); byte++)
          for (uint offset = 0; offset < 4; offset++)
            lookup[byte] |= pgenCodeToBed[(byte >> (2*offset)) & 3] << (2*offset);
        return lookup;
      }

      const std::array<uchar, 256> PGEN_TO_BED_LOOKUP = makePgenToBedLookup();

    }

    FileSet resolvePrefix(const string &prefix) {
      FileSet files;
      files.pgenFile = resolveSidecar(prefix, ".pgen", false);
      files.pvarFile = resolveSidecar(prefix, ".pvar", true);
      files.psamFile = resolveSidecar(prefix, ".psam", true);
      return files;
    }

    vector<SampleInfo> readPsamFile(const string &psamFile) {
      FileUtils::AutoGzIfstream fin;
      fin.openOrExit(psamFile);

      vector<SampleInfo> samples;
      vector<string> header;
      bool headerFound = false;
      string line;
      uint64 lineNum = 0;
      while (getline(fin, line)) {
        lineNum++;
        vector<string> fields = splitFields(line);
        if (fields.empty()) continue;

        if (fields[0] == "#FID" || fields[0] == "#IID") {
          header = fields;
          header[0].erase(0, 1);
          headerFound = true;
          break;
        }
        if (fields[0][0] == '#') continue;

        // A psam without a header follows fam column order.
        samples.push_back(parseFamLayoutSample(fields, psamFile, lineNum));
        while (getline(fin, line)) {
          lineNum++;
          fields = splitFields(line);
          if (!fields.empty())
            samples.push_back(parseFamLayoutSample(fields, psamFile, lineNum));
        }
        fin.close();
        return samples;
      }

      if (!headerFound) {
        cerr << "ERROR: No sample records or #FID/#IID header found in psam file: "
             << psamFile << endl;
        exit(1);
      }

      const int iidCol = findColumn(header, "IID");
      const int fidCol = findColumn(header, "FID");
      const int patCol = findColumn(header, "PAT");
      const int matCol = findColumn(header, "MAT");
      const int sexCol = findColumn(header, "SEX");
      int phenoCol = findColumn(header, "PHENO1");
      if (iidCol == -1) {
        cerr << "ERROR: IID column not found in psam header: " << psamFile << endl;
        exit(1);
      }
      if (phenoCol == -1) {
        for (uint col = 0; col < header.size(); col++) {
          const string &name = header[col];
          if (name != "FID" && name != "IID" && name != "SID" && name != "PAT" &&
              name != "MAT" && name != "SEX") {
            phenoCol = col;
            break;
          }
        }
      }

      while (getline(fin, line)) {
        lineNum++;
        vector<string> fields = splitFields(line);
        if (fields.empty()) continue;
        if (fields.size() < header.size()) {
          cerr << "ERROR: Too few fields at line " << lineNum << " of psam file: "
               << psamFile << endl;
          exit(1);
        }
        SampleInfo sample;
        sample.famID = fidCol == -1 ? "0" : fields[fidCol];
        sample.indivID = fields[iidCol];
        sample.paternalID = patCol == -1 ? "0" : fields[patCol];
        sample.maternalID = matCol == -1 ? "0" : fields[matCol];
        sample.sex = sexCol == -1 ? 0 : parseSex(fields[sexCol], psamFile, lineNum);
        sample.pheno = phenoCol == -1 ? -9 : parsePheno(fields[phenoCol]);
        samples.push_back(sample);
      }
      fin.close();
      return samples;
    }

    vector<SnpInfo> readPvarFile(const string &pvarFile, int Nauto,
                                 const std::unordered_set<string> *excludeSnpsPtr) {
      FileUtils::AutoGzIfstream fin;
      fin.openOrExit(pvarFile);

      vector<SnpInfo> variants;
      vector<string> header;
      bool headerFound = false;
      string line;
      uint64 lineNum = 0;
      int numOutOfOrder = 0;
      while (getline(fin, line)) {
        lineNum++;
        vector<string> fields = splitFields(line);
        if (fields.empty()) continue;
        if (fields[0] == "#CHROM") {
          header = fields;
          header[0].erase(0, 1);
          headerFound = true;
          break;
        }
        if (fields[0][0] == '#') continue;

        cerr << "ERROR: Headerless pvar files are not currently supported: "
             << pvarFile << endl;
        exit(1);
      }
      if (!headerFound) {
        cerr << "ERROR: #CHROM header not found in pvar file: " << pvarFile << endl;
        exit(1);
      }

      const int chromCol = findColumn(header, "CHROM");
      const int posCol = findColumn(header, "POS");
      const int idCol = findColumn(header, "ID");
      const int refCol = findColumn(header, "REF");
      const int altCol = findColumn(header, "ALT");
      const int cmCol = findColumn(header, "CM");
      if (chromCol == -1 || posCol == -1 || idCol == -1 || refCol == -1 || altCol == -1) {
        cerr << "ERROR: Pvar header must contain CHROM, POS, ID, REF, and ALT: "
             << pvarFile << endl;
        exit(1);
      }

      while (getline(fin, line)) {
        lineNum++;
        vector<string> fields = splitFields(line);
        if (fields.empty()) continue;
        if (fields.size() < header.size()) {
          cerr << "ERROR: Too few fields at line " << lineNum << " of pvar file: "
               << pvarFile << endl;
          exit(1);
        }

        SnpInfo snp;
        snp.ID = fields[idCol];
        snp.chrom = SnpData::chrStrToInt(fields[chromCol], Nauto);
        snp.physpos = parseInt(fields[posCol], "POS", pvarFile, lineNum);
        snp.genpos = cmCol == -1 || fields[cmCol] == "." ? 0 :
          parseDouble(fields[cmCol], "CM", pvarFile, lineNum);
        snp.allele1 = fields[altCol];
        snp.allele2 = fields[refCol];
        if (snp.allele1.find(',') != string::npos) {
          cerr << "ERROR: Multiallelic PGEN variants are not supported (variant "
               << snp.ID << " at line " << lineNum << " of " << pvarFile << ")" << endl;
          exit(1);
        }
        if (snp.chrom == -1 &&
            (excludeSnpsPtr == NULL || excludeSnpsPtr->find(snp.ID) == excludeSnpsPtr->end())) {
          cerr << "ERROR: Unknown chromosome code at line " << lineNum
               << " of pvar file: " << pvarFile << endl;
          exit(1);
        }
        if (!variants.empty() &&
            (snp.chrom < variants.back().chrom ||
             (snp.chrom == variants.back().chrom &&
              (snp.physpos <= variants.back().physpos || snp.genpos < variants.back().genpos)))) {
          if (numOutOfOrder < 5)
            cerr << "WARNING: Out-of-order variant " << snp.ID << " at line " << lineNum
                 << " of pvar file: " << pvarFile << endl;
          numOutOfOrder++;
        }
        variants.push_back(snp);
      }
      if (numOutOfOrder)
        cerr << "WARNING: Total number of out-of-order variants in pvar file: "
             << numOutOfOrder << endl;
      fin.close();
      return variants;
    }

    void packedPgenToBed(uchar out[], const uchar in[], uint64 N, uint64 Nstride) {
      const uint64 packedBytes = (N+3)>>2;
      for (uint64 byte = 0; byte < packedBytes; byte++)
        out[byte] = PGEN_TO_BED_LOOKUP[in[byte]];
      if (N&3)
        out[packedBytes-1] &= (1U << (2*(N&3))) - 1;
      memset(out + packedBytes, 0, (Nstride/4-packedBytes) * sizeof(out[0]));
    }

  }
}
