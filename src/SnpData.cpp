/*
   This file is part of the BOLT-LMM linear mixed model software package
   developed by Po-Ru Loh.  Copyright (C) 2014-2025 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstring>
#include <cmath>
#include <cassert>
#include <array>
#include <climits>
#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <limits>
#include <sstream>
#include <map>
#include <unordered_set>
#include <utility>

#include <sys/mman.h>
#include <sys/statvfs.h>
#include <unistd.h>

#ifdef USE_SSE
#include <emmintrin.h> // SSE2 for packed doubles
#endif
#ifdef USE_NEON
#include <arm_neon.h>
#endif

#include "Types.hpp"
#include "SnpData.hpp"
#include "SnpInfo.hpp"
#include "RestrictSnpSet.hpp"
#include "FileUtils.hpp"
#include "MemoryUtils.hpp"
#include "MapInterpolater.hpp"
#include "LapackConst.hpp"
#include "PgenUtils.hpp"
#include "pgenlibr.h"

namespace LMM {

  using std::vector;
  using std::string;
  using std::pair;
  using std::cout;
  using std::cerr;
  using std::endl;
  using FileUtils::getline;

  namespace {

    typedef std::array<uchar, 4> DecodedBedByte;

    struct BedByteStats {
      uchar alleleSum;
      uchar numMissing;
    };

    std::array<DecodedBedByte, 256> makeBedByteLookup() {
      const uchar bedToGeno[4] = {2, 9, 1, 0};
      std::array<DecodedBedByte, 256> lookup = {};
      for (uint byte = 0; byte < lookup.size(); byte++)
	for (uint k = 0; k < 4; k++)
	  lookup[byte][k] = bedToGeno[(byte >> (2*k)) & 3];
      return lookup;
    }

    const std::array<DecodedBedByte, 256> BED_BYTE_LOOKUP = makeBedByteLookup();

    std::array<BedByteStats, 256> makeBedByteStats() {
      std::array<BedByteStats, 256> stats = {};
      for (uint byte = 0; byte < stats.size(); byte++)
	for (uint k = 0; k < 4; k++) {
	  const uchar geno = BED_BYTE_LOOKUP[byte][k];
	  if (geno == 9)
	    stats[byte].numMissing++;
	  else
	    stats[byte].alleleSum += geno;
	}
      return stats;
    }

    const std::array<BedByteStats, 256> BED_BYTE_STATS = makeBedByteStats();

    uint64 physicalMemoryBytes() {
      const long pages = sysconf(_SC_PHYS_PAGES);
      const long pageSize = sysconf(_SC_PAGESIZE);
      if (pages <= 0 || pageSize <= 0)
	return 0;
      if (static_cast<uint64>(pages) >
	  std::numeric_limits<uint64>::max() / static_cast<uint64>(pageSize))
	return 0;
      return static_cast<uint64>(pages) * static_cast<uint64>(pageSize);
    }

    string defaultCacheDir() {
      const char *tmpDir = std::getenv("TMPDIR");
      return tmpDir && *tmpDir ? tmpDir : "/tmp";
    }

    double bytesToGiB(uint64 bytes) {
      return bytes / static_cast<double>(1ULL << 30);
    }

  }

  const uint64 SnpData::IND_MISSING = (uint64) -1;

  int SnpData::chrStrToInt(string chrom, int Nauto) {
    if (chrom.substr(0, 3) == "chr")
      chrom = chrom.substr(3);
    if (isdigit(chrom[0])) {
      int chr = atoi(chrom.c_str());
      if (chr>=1 && chr<=Nauto+1) return chr;
      return -1;
    }
    if (chrom == "X" || chrom == "XY" || chrom == "PAR1" || chrom == "PAR2") return Nauto+1;
    return -1;
  }

  /**
   * work: 256x4 aligned work array
   * lookupBedCode[4] = {value of 0, value of missing, value of 1, value of 2}
   */
  void SnpData::buildByteLookup(double (*work)[4], const double lookupBedCode[4]) const {
    for (int byte4 = 0; byte4 < 256; byte4 += 4) {
      for (int k = 0; k < 4; k++) // fill 4 values for first of 4 consecutive bytes
	work[byte4][k] = lookupBedCode[(byte4>>(k+k))&3];
      for (int k = 1; k < 4; k++) {
	memcpy(work[byte4+k], work[byte4], sizeof(work[0]));
	work[byte4+k][0] = lookupBedCode[k];
      }
    }
    /* slow? way
    for (int byte = 0; byte < 256; byte++) {
      for (int k = 0; k < 4; k++)
	work[byte][k] = lookupBedCode[(byte>>(k+k))&3];
    }
    */
  }

  void SnpData::estChipLDscoresChrom(vector <double> &chipLDscores, int chrom,
				     const vector <int> &indivInds) const {
    uint64 chrStart = 0;
    while (chrStart < M && snps[chrStart].chrom != chrom) chrStart++;
    uint64 chrEnd = chrStart;
    while (chrEnd < M && snps[chrEnd].chrom == chrom) chrEnd++;
    uint64 Mchr = chrEnd - chrStart;
    if (Mchr == 0) return;
    uint64 Nsub = indivInds.size();

    //cout << "Estimating chip LD Scores on chromosome " << chrom << endl;
    for (uint64 m = chrStart; m < chrEnd; m++) chipLDscores[m] = 1.0;

    // allocate memory
    uchar *chrMaskSnps = ALIGNED_MALLOC_UCHARS(Mchr);
    memset(chrMaskSnps, 0, Mchr * sizeof(chrMaskSnps[0]));
    float *chrNormalizedGenos = ALIGNED_MALLOC_FLOATS(Mchr * Nsub);
    memset(chrNormalizedGenos, 0, Mchr * Nsub * sizeof(chrNormalizedGenos[0]));
    const int mBlock = 64;
    float *dotProds = ALIGNED_MALLOC_FLOATS(Mchr * mBlock);

    // fill and normalize genotypes
    for (uint64 mchr = 0; mchr < Mchr; mchr++) {
      uint64 m = chrStart + mchr;
      if (maskSnps[m])
	chrMaskSnps[mchr] = fillSnpSubRowNorm1(chrNormalizedGenos + mchr*Nsub, m, indivInds);
    }

    uint64 mchrWindowStart = 0;
    for (uint64 mchr0 = 0; mchr0 < Mchr; mchr0 += mBlock) { // sgemm to compute r2s
      uint64 mBlockCrop = std::min(Mchr, mchr0+mBlock) - mchr0;
      while (!isProximal(chrStart + mchrWindowStart, chrStart + mchr0, 0.01, 1000000))
	mchrWindowStart++;
      uint64 prevWindowSize = mchr0+mBlockCrop-1 - mchrWindowStart;

      // [mchrWindowStart..mchr0+mBlockCrop-1) x [mchr0..mchr0+mBlockCrop)
      {
	char TRANSA_ = 'T';
	char TRANSB_ = 'N';
	int M_ = prevWindowSize;
	int N_ = mBlockCrop;
	int K_ = Nsub;
	float ALPHA_ = 1;
	float *A_ = chrNormalizedGenos + mchrWindowStart*Nsub;
	int LDA_ = Nsub;
	float *B_ = chrNormalizedGenos + mchr0*Nsub;
	int LDB_ = Nsub;
	float BETA_ = 0;
	float *C_ = dotProds;
	int LDC_ = prevWindowSize;
	SGEMM_MACRO(&TRANSA_, &TRANSB_, &M_, &N_, &K_, &ALPHA_, A_, &LDA_, B_, &LDB_,
		    &BETA_, C_, &LDC_);
      }
      
      for (uint64 mPlus = 0; mPlus < mBlockCrop; mPlus++) {
	uint64 m = chrStart + mchr0 + mPlus;
	if (!chrMaskSnps[m-chrStart]) continue;
	for (uint64 mPlus2 = 0; mchrWindowStart+mPlus2 < mchr0+mPlus; mPlus2++) {
	  uint64 m2 = chrStart + mchrWindowStart + mPlus2;
	  if (!chrMaskSnps[m2-chrStart]) continue;
	  float adjR2 = dotProdToAdjR2(dotProds[mPlus2 + mPlus*prevWindowSize], Nsub);
	  chipLDscores[m] += adjR2;
	  chipLDscores[m2] += adjR2;
	}
      }
    }

    ALIGNED_FREE(dotProds);
    ALIGNED_FREE(chrNormalizedGenos);
    ALIGNED_FREE(chrMaskSnps);
  }

  float SnpData::dotProdToAdjR2(float dotProd, int n) {
    float r2 = dotProd*dotProd;
    return r2 - (1-r2)/(n-2);
  }

  /**
   * fills x[] with indivInds.size() elements corresponding to chosen subset of indivInds
   * replaces missing values with average; mean-centers and normalizes vector length to 1
   * if monomorphic among non-missing, fills with all-0s
   *
   * return: true if snp is polymorphic in indivInds; false if not
   */
  bool SnpData::fillSnpSubRowNorm1(float x[], uint64 m, const vector <int> &indivInds) const {
    /* lookupBedCode[4] = {value of 0, value of missing, value of 1, value of 2} */
    const float lookupBedCode[4] = {0.0f, 9.0f, 1.0f, 2.0f};
    float sumPresent = 0; int numPresent = 0;
    for (uint64 i = 0; i < indivInds.size(); i++) {
      uint64 n = indivInds[i];
      uchar byte = genotypes[m * (Nstride>>2) + (n>>2)];
      int k = n&3;
      x[i] = lookupBedCode[(byte>>(k+k))&3];
      if (x[i] != 9.0f) {
	sumPresent += x[i];
	numPresent++;
      }
    }
    float avg = sumPresent / numPresent;
    float sum2 = 0;
    for (uint64 i = 0; i < indivInds.size(); i++) {
      if (x[i] != 9.0f) { // non-missing; mean-center
	x[i] -= avg;
	sum2 += x[i]*x[i];
      }
      else // missing; replace with mean (centered to 0)
	x[i] = 0;
    }
    if (sum2 < 0.001) { // monomorphic among non-missing
      for (uint64 i = 0; i < indivInds.size(); i++) x[i] = 0; // set to 0
      return false;
    }
    else { // polymorphic
      float invNorm = 1.0f / sqrtf(sum2);
      for (uint64 i = 0; i < indivInds.size(); i++) x[i] *= invNorm; // normalize to vector len 1
      return true;
    }
  }

  void SnpData::processIndivs(const string &sampleFile, const vector <string> &removeFiles,
                              bool psamFormat) {
    string line;

    vector <IndivInfo> bedIndivs;
    FileUtils::AutoGzIfstream fin;
    if (psamFormat) {
      vector <PgenUtils::SampleInfo> psamIndivs = PgenUtils::readPsamFile(sampleFile);
      bedIndivs.reserve(psamIndivs.size());
      for (uint64 n = 0; n < psamIndivs.size(); n++) {
	IndivInfo indiv;
	indiv.famID = psamIndivs[n].famID;
	indiv.indivID = psamIndivs[n].indivID;
	indiv.paternalID = psamIndivs[n].paternalID;
	indiv.maternalID = psamIndivs[n].maternalID;
	indiv.sex = psamIndivs[n].sex;
	indiv.pheno = psamIndivs[n].pheno;
	bedIndivs.push_back(std::move(indiv));
      }
    }
    else {
      fin.openOrExit(sampleFile);
      while (getline(fin, line)) {
	std::istringstream iss(line);
	IndivInfo indiv; string phenoStr;
	if (!(iss >> indiv.famID >> indiv.indivID >> indiv.paternalID >> indiv.maternalID
	      >> indiv.sex >> phenoStr)) {
	  cerr << "ERROR: Incorrectly formatted fam file: " << sampleFile << endl;
	  cerr << "Line " << bedIndivs.size()+1 << ":" << endl;
	  cerr << line << endl;
	  cerr << "Unable to input 6 values (4 string, 1 int, 1 double/string)" << endl;
	  exit(1);
	}
	if (sscanf(phenoStr.c_str(), "%lf", &indiv.pheno) != 1)
	  indiv.pheno = -9;
	bedIndivs.push_back(std::move(indiv));
      }
      fin.close();
    }

    FID_IID_to_ind.reserve(bedIndivs.size());
    for (uint64 n = 0; n < bedIndivs.size(); n++) {
      string combined_ID = bedIndivs[n].famID + " " + bedIndivs[n].indivID;
      if (!FID_IID_to_ind.emplace(combined_ID, n).second) {
	cerr << "ERROR: Duplicate individual in " << (psamFormat ? "psam" : "fam")
	     << " file at sample " << n+1 << endl;
	exit(1);
      }
    }
    Nbed = bedIndivs.size();

    cout << "Total indivs in PLINK data: Nbed = " << Nbed << endl;

    // process individuals to remove
    vector <bool> useIndiv(Nbed, true);
    for (uint f = 0; f < removeFiles.size(); f++) {
      const string &removeFile = removeFiles[f];
      cout << "Reading remove file (indivs to remove): " << removeFile << endl;
      fin.openOrExit(removeFile);
      int lineCtr = 0;
      int numRemoved = 0;
      int numAbsent = 0;
      while (getline(fin, line)) {
	lineCtr++;
	std::istringstream iss(line);
	string FID, IID;
	if (!(iss >> FID >> IID)) {
	  cerr << "ERROR: Incorrectly formatted remove file: " << removeFile << endl;
	  cerr << "Line " << lineCtr << ":" << endl;
	  cerr << line << endl;
	  cerr << "Unable to input FID and IID" << endl;
	  exit(1);
	}
	string combined_ID = FID + " " + IID;
	std::unordered_map <string, uint64>::const_iterator it = FID_IID_to_ind.find(combined_ID);
	if (it == FID_IID_to_ind.end()) {
	  if (numAbsent < 5)
	    cerr << "WARNING: Unable to find individual to remove: " << combined_ID << endl;
	  numAbsent++;
	}
	else if (useIndiv[it->second]) {
	  useIndiv[it->second] = false;
	  numRemoved++;
	}
      }
      fin.close();
      cout << "Removed " << numRemoved << " individual(s)" << endl;
      if (numAbsent)
	cerr << "WARNING: " << numAbsent << " individual(s) not found in data set" << endl;
    }

    // determine number of indivs remaining post-removal and set up indices
    bedIndivToRemoveIndex.resize(Nbed);
    FID_IID_to_ind.clear(); // redo FID_IID -> indiv index
    FID_IID_to_ind.reserve(Nbed);
    indivs.reserve(Nbed);
    for (uint64 nbed = 0; nbed < Nbed; nbed++) {
      if (useIndiv[nbed]) {
	bedIndivToRemoveIndex[nbed] = indivs.size();
	FID_IID_to_ind[bedIndivs[nbed].famID + " " + bedIndivs[nbed].indivID] = indivs.size();
	indivs.push_back(std::move(bedIndivs[nbed]));
      }
      else
	bedIndivToRemoveIndex[nbed] = -1;
    }
    N = indivs.size();
    bedIndivsIdentity = Nbed == N;
    for (uint64 nbed = 0; bedIndivsIdentity && nbed < Nbed; nbed++)
      bedIndivsIdentity = bedIndivToRemoveIndex[nbed] == static_cast<int>(nbed);
    cout << "Total indivs stored in memory: N = " << N << endl;

    // allocate and initialize maskIndivs to all good (aside from filler at end)
    Nstride = (N+3)&~3;
    maskIndivs = ALIGNED_MALLOC_DOUBLES(Nstride);
    for (uint64 n = 0; n < N; n++) maskIndivs[n] = 1;
    for (uint64 n = N; n < Nstride; n++) maskIndivs[n] = 0;
  }

  vector <SnpInfo> SnpData::processSnps(vector <uint64> &Mfiles,
					const vector <string> &variantFiles,
					const vector <string> &excludeFiles,
					const vector <string> &modelSnpsFiles,
					const vector <string> &vcNamesIn, bool loadNonModelSnps,
					bool pvarFormat) {
    FileUtils::AutoGzIfstream fin;
    string line;

    std::unordered_set <string> excludeSnps;
    if (!excludeFiles.empty()) {
      for (uint f = 0; f < excludeFiles.size(); f++) {
	const string &excludeFile = excludeFiles[f];
	cout << "First-pass reading exclude file (SNPs to exclude): " << excludeFile << endl;
	fin.openOrExit(excludeFile);
	string rsID;
	while (fin >> rsID) {
	  excludeSnps.insert(rsID);
	  getline(fin, line);
	}
	fin.close();
      }
    }

    vector <SnpInfo> bedSnps;
    // read variant metadata files
    for (uint i = 0; i < variantFiles.size(); i++) {
      cout << "Reading " << (pvarFormat ? "pvar" : "bim") << " file #" << (i+1)
	   << ": " << variantFiles[i] << endl;
      vector <SnpInfo> snps_i = pvarFormat ?
	PgenUtils::readPvarFile(variantFiles[i], Nautosomes, &excludeSnps) :
	readBimFile(variantFiles[i], Nautosomes, &excludeSnps);
      bedSnps.insert(bedSnps.end(), snps_i.begin(), snps_i.end());
      Mfiles.push_back(snps_i.size());
      cout << "    Read " << Mfiles.back() << " snps" << endl;
    }
    Mbed = bedSnps.size();
    excludeSnps.clear(); // free memory no longer needed

    cout << "Total snps in PLINK data: Mbed = " << Mbed << endl;

    // >=1 = in GRM (model), 0 = not in GRM, -1 = exclude
    const int excludeVCnum = -1;
    const int nonGrmVCnum = 0;
    const int firstVCnum = 1;
    // if list of GRM snps, default = no
    const int defaultVCnum = modelSnpsFiles.empty() ? firstVCnum : nonGrmVCnum;
    vector <int> snpVCnum(Mbed, (char) defaultVCnum);
    if (vcNamesIn.empty()) {
      vcNames = vector <string> (1, "env/noise"); // 0th entry of vcNames is ignored; VCs 1-indexed
      if (modelSnpsFiles.empty()) // if modelSnpsFiles are given, vcNames will be populated later
	vcNames.push_back("modelSnps"); // if not, put all non-excluded SNPs in one variance comp
    }
    else
      vcNames = vcNamesIn;

    // check for duplicate snps
    {
      std::unordered_set <string> rsIDs;
      rsIDs.reserve(Mbed);
      for (uint64 mbed = 0; mbed < Mbed; mbed++) {
	if (!rsIDs.insert(bedSnps[mbed].ID).second) {
	  cerr << "WARNING: Duplicate snp ID " << bedSnps[mbed].ID
	       << " -- masking duplicate" << endl;
	  snpVCnum[mbed] = excludeVCnum;
	}
      }
    }

    // exclude snps if list provided; restrict GRM snps if list provided
    if (!excludeFiles.empty() || !modelSnpsFiles.empty()) {

      // create dictionary rsID -> index in full bed snp list
      std::unordered_map <string, uint64> rsID_to_ind;
      rsID_to_ind.reserve(Mbed);
      for (uint64 mbed = 0; mbed < Mbed; mbed++)
	rsID_to_ind.emplace(bedSnps[mbed].ID, mbed); // only use first of dupe IDs

      // exclude snps
      for (uint f = 0; f < excludeFiles.size(); f++) {
	const string &excludeFile = excludeFiles[f];
	cout << "Reading exclude file (SNPs to exclude): " << excludeFile << endl;
	fin.openOrExit(excludeFile);
	int numExcluded = 0;
	int numAbsent = 0;
	while (getline(fin, line)) {
	  std::istringstream iss(line);
	  string rsID; iss >> rsID;
	  std::unordered_map <string, uint64>::const_iterator it = rsID_to_ind.find(rsID);
	  if (it == rsID_to_ind.end()) {
	    if (numAbsent < 5)
	      cerr << "WARNING: Unable to find SNP to exclude: " << rsID << endl;
	    numAbsent++;
	  }
	  else if (snpVCnum[it->second] != excludeVCnum) {
	    snpVCnum[it->second] = excludeVCnum;
	    numExcluded++;
	  }
	}
	fin.close();
	cout << "Excluded " << numExcluded << " SNP(s)" << endl;
	if (numAbsent)
	  cerr << "WARNING: " << numAbsent << " SNP(s) not found in data set" << endl;
      }

      // include GRM snps listed
      std::map <string, int> vcNamesToInds;
      if (!vcNamesIn.empty()) // initialize map
	for (int v = 1; v < (int) vcNamesIn.size(); v++)
	  vcNamesToInds[vcNamesIn[v]] = v;

      for (uint f = 0; f < modelSnpsFiles.size(); f++) {
	const string &modelSnpsFile = modelSnpsFiles[f];
	cout << "Reading list of SNPs to include in model (i.e., GRM): " << modelSnpsFile << endl;
	fin.openOrExit(modelSnpsFile);
	int numIncluded = 0;
	int numAlreadyExcluded = 0;
	int numAlreadyAssigned = 0;
	int numAbsent = 0;
	while (getline(fin, line)) {
	  std::istringstream iss(line);
	  string rsID; iss >> rsID;
	  std::unordered_map <string, uint64>::const_iterator it = rsID_to_ind.find(rsID);
	  if (it == rsID_to_ind.end()) {
	    if (numAbsent < 5)
	      cerr << "WARNING: Unable to find SNP to include in model: " << rsID << endl;
	    numAbsent++;
	  }
	  else if (snpVCnum[it->second] == excludeVCnum) {
	    if (numAlreadyExcluded < 5)
	      cerr << "WARNING: SNP has been excluded: " << rsID << endl;
	    numAlreadyExcluded++;
	  }
	  else if (snpVCnum[it->second] == nonGrmVCnum) {
	    string vcName; iss >> vcName; // ok if all lines have no other fields; then vcName=""
	    if (vcName.empty()) vcName = "modelSnps";
	    if (vcNamesToInds.find(vcName) == vcNamesToInds.end()) {
	      if (vcNamesIn.empty()) {
		vcNamesToInds[vcName] = vcNames.size();
		vcNames.push_back(vcName);
	      }
	      else {
		cerr << "ERROR: SNP " << rsID << " is assigned to VC '" << vcName
		     << "' not in --remlGuessStr" << endl;
		exit(1);
	      }
	    }
	    int vcNum = vcNamesToInds[vcName];
	    if (vcNum > SnpInfo::MAX_VC_NUM) {
	      cerr << "ERROR: Too many distinct variance component names (2nd column); max = "
		   << SnpInfo::MAX_VC_NUM << endl;
	      exit(1);
	    }
	    snpVCnum[it->second] = vcNum;
	    numIncluded++;
	  }
	  else {
	    if (numAlreadyAssigned < 5)
	      cerr << "WARNING: SNP was already assigned to a variance comp: " << rsID << endl;
	    numAlreadyAssigned++;
	  }
	}
	fin.close();
	cout << "Included " << numIncluded << " SNP(s) in model in "
	     << vcNames.size()-1 << " variance component(s)"<< endl;
	if (numAbsent)
	  cerr << "WARNING: " << numAbsent << " SNP(s) not found in data set" << endl;
	if (numAlreadyExcluded)
	  cerr << "WARNING: " << numAlreadyExcluded << " SNP(s) had been excluded" << endl;
	if (numAlreadyAssigned)
	  cerr << "WARNING: " << numAlreadyAssigned << " SNP(s) were multiply assigned" << endl;
      }
    }

    // determine number of snps remaining post-exclusion and set up index
    M = 0; // note: M will be updated later after further QC
    uint64 Mexclude = 0, MnonGRM = 0;
    bedSnpToGrmIndex.resize(Mbed); // note: this index will be further updated after QC
    for (uint64 mbed = 0; mbed < Mbed; mbed++) {
      int vcNum = snpVCnum[mbed];
      bedSnps[mbed].vcNum = vcNum;
      if (vcNum >= firstVCnum) {
	bedSnpToGrmIndex[mbed] = (int) M;
	M++;
      }
      else if (vcNum == nonGrmVCnum && loadNonModelSnps) {
	bedSnpToGrmIndex[mbed] = -1;
	MnonGRM++;
      }
      else if (vcNum == nonGrmVCnum || vcNum == excludeVCnum) {
	bedSnpToGrmIndex[mbed] = -2;
	Mexclude++;
      }
      else assert(false); // shouldn't be any other possibilities
    }
    cout << endl;
    cout << "Breakdown of SNP pre-filtering results:" << endl;
    cout << "  " << M << " SNPs to include in model (i.e., GRM)" << endl;
    cout << "  " << MnonGRM << " additional non-GRM SNPs loaded" << endl;
    cout << "  " << Mexclude << " excluded SNPs" << endl;
    /*    
    if (M < 10) {
      cerr << "ERROR: Very few SNPs included in model; probably an input error" << endl;
      exit(1);
    }
    */
    return bedSnps;
  }

  void SnpData::processMap(vector <SnpInfo> &bedSnps, const string &geneticMapFile,
			   bool noMapCheck) {
    // fill in map if external file provided
    if (!geneticMapFile.empty()) {
      cout << "Filling in genetic map coordinates using reference file:" << endl;
      cout << "  " << geneticMapFile << endl;
      MapInterpolater mapInterpolater(geneticMapFile);
      for (uint64 mbed = 0; mbed < Mbed; mbed++)
	if (bedSnpToGrmIndex[mbed] != -2)
	  bedSnps[mbed].genpos =
	    mapInterpolater.interp(bedSnps[mbed].chrom, bedSnps[mbed].physpos);
    }
    
    // check map and rescale if in cM units: calculate genpos/physpos for last autosomal snp
    mapAvailable = false;
    for (int mbedLast = (int) Mbed-1; mbedLast >= 0; mbedLast--)
      if (bedSnpToGrmIndex[mbedLast] != -2 && bedSnps[mbedLast].chrom <= Nautosomes+1) {
	double scale = bedSnps[mbedLast].genpos / bedSnps[mbedLast].physpos;
	if (scale == 0)
	  cerr << "WARNING: No genetic map provided; using physical positions only" << endl;
	else if (0.5e-6 < scale && scale < 2e-6) {
	  cerr << "WARNING: Genetic map appears to be in cM units; rescaling by 0.01" << endl;
	  for (uint64 mbed = 0; mbed < Mbed; mbed++)
	    bedSnps[mbed].genpos *= 0.01;
	  mapAvailable = true;
	}
	else if (0.5e-8 < scale && scale < 2e-8)
	  mapAvailable = true;
	else {
	  if (noMapCheck) {
	    cerr << "WARNING: Genetic map appears wrong based on last genpos/bp" << endl;
	    cerr << "         Proceeding anyway because --noMapCheck is set" << endl;
	    mapAvailable = true;
	  }
	  else {
	    cerr << "ERROR: Genetic map appears wrong based on last genpos/bp" << endl;
	    cerr << "       To proceed anyway, set --noMapCheck" << endl;
	    exit(1);
	  }
	}
	break;
      }
  }

  void SnpData::storeBedLineAndCountMissing(uchar bedLineOut[], const uchar genoLine[],
					     int numMissingPerIndiv[]) {
    const uchar genoToBed[10] = {3, 2, 0, 0, 0, 0, 0, 0, 0, 1};
    const uint64 fullBytes = N >> 2;
    for (uint64 byte = 0; byte < fullBytes; byte++) {
      const uchar *geno = genoLine + (byte << 2);
      bedLineOut[byte] = genoToBed[geno[0]] | (genoToBed[geno[1]] << 2) |
	(genoToBed[geno[2]] << 4) | (genoToBed[geno[3]] << 6);
      for (uint64 offset = 0; offset < 4; offset++)
	if (geno[offset] == 9)
	  numMissingPerIndiv[(byte << 2) + offset]++;
    }

    const uint64 packedBytes = (N+3) >> 2;
    if (fullBytes != packedBytes) {
      uchar packed = 0;
      for (uint64 n = fullBytes << 2; n < N; n++) {
	packed |= genoToBed[genoLine[n]] << ((n&3) << 1);
	if (genoLine[n] == 9)
	  numMissingPerIndiv[n]++;
      }
      bedLineOut[fullBytes] = packed;
    }
    memset(bedLineOut + packedBytes, 0, (Nstride/4-packedBytes) * sizeof(bedLineOut[0]));
  }

  double SnpData::computeIdentityBedAlleleFreqAndMissing(const uchar bedLine[],
						  double *missing) const {
    uint64 alleleSum = 0, numMissing = 0;
    const uint64 fullBytes = N >> 2;
    for (uint64 byte = 0; byte < fullBytes; byte++) {
      const BedByteStats &stats = BED_BYTE_STATS[bedLine[byte]];
      alleleSum += stats.alleleSum;
      numMissing += stats.numMissing;
    }
    for (uint64 n = fullBytes << 2; n < N; n++) {
      const uchar geno = BED_BYTE_LOOKUP[bedLine[n>>2]][n&3];
      if (geno == 9)
	numMissing++;
      else
	alleleSum += geno;
    }
    *missing = static_cast<double>(numMissing) / N;
    return 0.5 * alleleSum / (N-numMissing);
  }

  void SnpData::storeIdentityBedLineAndCountMissing(uchar bedLineOut[],
						     const uchar bedLineIn[],
						     int numMissingPerIndiv[]) const {
    const uint64 fullBytes = N >> 2;
    for (uint64 byte = 0; byte < fullBytes; byte++) {
      const uchar packed = bedLineIn[byte];
      bedLineOut[byte] = packed;
      for (uint64 offset = 0; offset < 4; offset++)
	if (((packed >> (offset << 1)) & 3) == 1)
	  numMissingPerIndiv[(byte << 2) + offset]++;
    }
    if (N & 3) {
      const uint64 byte = fullBytes;
      const uchar paddingMask = (1U << (2*(N&3))) - 1;
      bedLineOut[byte] = bedLineIn[byte] & paddingMask;
      for (uint64 n = byte << 2; n < N; n++)
	if (((bedLineIn[byte] >> ((n&3) << 1)) & 3) == 1)
	  numMissingPerIndiv[n]++;
    }
  }

  /**
   * assumes Nbed and bedIndivToRemoveIndex have been initialized
   * if loadGenoLine == false, just advances the file pointer
   */
  void SnpData::readBedLine(uchar genoLine[], uchar bedLineIn[], FileUtils::AutoGzIfstream &fin,
			    bool loadGenoLine) const {
    fin.read((char *) bedLineIn, (Nbed+3)>>2);
    if (loadGenoLine) {
      if (bedIndivsIdentity) {
	const uint64 fullBytes = N >> 2;
	for (uint64 byte = 0; byte < fullBytes; byte++)
	  memcpy(genoLine + (byte << 2), BED_BYTE_LOOKUP[bedLineIn[byte]].data(), 4);
	for (uint64 n = fullBytes << 2; n < N; n++)
	  genoLine[n] = BED_BYTE_LOOKUP[bedLineIn[n>>2]][n&3];
	return;
      }
      const int bedToGeno[4] = {2, 9, 1, 0};
      for (uint64 nbed = 0; nbed < Nbed; nbed++)
	if (bedIndivToRemoveIndex[nbed] != -1) {
	  int genoValue = bedToGeno[(bedLineIn[nbed>>2]>>((nbed&3)<<1))&3];
	  genoLine[bedIndivToRemoveIndex[nbed]] = (uchar) genoValue;
	}
    }
  }

  bool SnpData::dosageValid(double dosage) {
    const double eps = 1e-6;
    return -eps <= dosage && dosage <= 2+eps;
  }

  bool SnpData::allIndivsIncluded(const double subMaskIndivs[]) const {
    for (uint64 n = 0; n < N; n++)
      if (!subMaskIndivs[n]) return false;
    return true;
  }

  bool SnpData::getBedIndivsIdentity(void) const { return bedIndivsIdentity; }

  double SnpData::computeAlleleFreqAndMissing(const uchar genoLine[],
					      const double subMaskIndivs[],
					      double *missing,
					      bool allIndivsIncluded) const {
    double alleleSum = 0, missingSum = 0;
    int numObserved = 0, numMasked = 0;
    if (allIndivsIncluded) {
      for (size_t n = 0; n < N; n++) {
	const bool isMissing = genoLine[n] == 9;
	missingSum += isMissing;
	if (!isMissing) {
	  alleleSum += genoLine[n];
	  numObserved++;
	}
      }
      *missing = missingSum / N;
      return 0.5 * alleleSum / numObserved;
    }
    for (size_t n = 0; n < N; n++) {
      if (subMaskIndivs[n]) {
	const bool isMissing = genoLine[n] == 9;
	missingSum += isMissing;
	numMasked++;
	if (!isMissing) {
	  alleleSum += genoLine[n];
	  numObserved++;
	}
      }
    }
    *missing = missingSum / numMasked;
    return 0.5 * alleleSum / numObserved;
  }

  double SnpData::computeAlleleFreq(const uchar genoLine[], const double subMaskIndivs[]) const {
    double sum = 0; int num = 0;
    for (size_t n = 0; n < N; n++)
      if (subMaskIndivs[n] && genoLine[n] != 9) {
	sum += genoLine[n];
	num++;
      }
    return 0.5 * sum / num;
  }
  double SnpData::computeAlleleFreq(const double dosageLine[], const double subMaskIndivs[])
    const {
    double sum = 0; int num = 0;
    for (size_t n = 0; n < N; n++)
      if (subMaskIndivs[n] && dosageValid(dosageLine[n])) {
	sum += dosageLine[n];
	num++;
      }
    return 0.5 * sum / num;
  }

  double SnpData::computeMAF(const uchar genoLine[], const double subMaskIndivs[]) const {
    double alleleFreq = computeAlleleFreq(genoLine, subMaskIndivs);
    return std::min(alleleFreq, 1.0-alleleFreq);
  }

  double SnpData::computeSnpMissing(const uchar genoLine[], const double subMaskIndivs[]) const {
    double sum = 0; int num = 0;
    for (uint64 n = 0; n < N; n++)
      if (subMaskIndivs[n]) {
	sum += (genoLine[n] == 9);
	num++;
      }
    return sum / num;
  }
  double SnpData::computeSnpMissing(const double dosageLine[], const double subMaskIndivs[])
    const {
    double sum = 0; int num = 0;
    for (uint64 n = 0; n < N; n++)
      if (subMaskIndivs[n]) {
	sum += !dosageValid(dosageLine[n]);
	num++;
      }
    return sum / num;
  }

  // assumes maskedSnpVector has dimension Nstride; zero-fills
  // note alleleFreq != MAF: alleleFreq = (mean allele count) / 2 and has full range [0..1]!
  void SnpData::genoLineToMaskedSnpVector(double maskedSnpVector[], const uchar genoLine[],
					  const double subMaskIndivs[], double alleleFreq,
					  bool allIndivsIncluded) const {
    if (allIndivsIncluded) {
      for (size_t n = 0; n < N; n++)
	maskedSnpVector[n] = genoLine[n] != 9 ? genoLine[n] - 2*alleleFreq : 0;
      for (uint64 n = N; n < Nstride; n++) maskedSnpVector[n] = 0;
      return;
    }
    for (size_t n = 0; n < N; n++) {
      if (subMaskIndivs[n] && genoLine[n] != 9)
	maskedSnpVector[n] = genoLine[n] - 2*alleleFreq;
      else
	maskedSnpVector[n] = 0;
    }
    for (uint64 n = N; n < Nstride; n++)
      maskedSnpVector[n] = 0;
  }

  void SnpData::identityBedLineToSnpVector(double out[], const uchar bedLine[],
					    double alleleFreq, double (*work)[4]) const {
    const double mean = 2*alleleFreq;
    const double lookupBedCode[4] = {2-mean, 0, 1-mean, -mean};
    buildByteLookup(work, lookupBedCode);

    const uint64 Nfull = N & ~(uint64) 3;
    const uchar *ptr = bedLine;
    for (uint64 n4 = 0; n4 < Nfull; n4 += 4) {
#if defined(USE_SSE)
      _mm_store_pd(&out[n4], _mm_load_pd(&work[*ptr][0]));
      _mm_store_pd(&out[n4+2], _mm_load_pd(&work[*ptr][2]));
#elif defined(USE_NEON)
      vst1q_f64(&out[n4], vld1q_f64(&work[*ptr][0]));
      vst1q_f64(&out[n4+2], vld1q_f64(&work[*ptr][2]));
#else
      memcpy(out + n4, work[*ptr], sizeof(work[0]));
#endif
      ptr++;
    }
    if (Nfull != N) {
      for (uint64 n = Nfull; n < N; n++)
	out[n] = work[*ptr][n&3];
    }
    for (uint64 n = N; n < Nstride; n++)
      out[n] = 0;
  }
  // assumes maskedSnpVector has dimension Nstride; zero-fills
  // note alleleFreq != MAF: alleleFreq = (mean allele count) / 2 and has full range [0..1]!
  void SnpData::dosageLineToMaskedSnpVector(double dosageLineVec[], const double subMaskIndivs[],
					    double alleleFreq, bool allIndivsIncluded) const {
    if (allIndivsIncluded) {
      for (size_t n = 0; n < N; n++)
	dosageLineVec[n] = dosageValid(dosageLineVec[n]) ? dosageLineVec[n] - 2*alleleFreq : 0;
      for (uint64 n = N; n < Nstride; n++) dosageLineVec[n] = 0;
      return;
    }
    for (size_t n = 0; n < N; n++) {
      if (subMaskIndivs[n] && dosageValid(dosageLineVec[n]))
	dosageLineVec[n] = dosageLineVec[n] - 2*alleleFreq;
      else
	dosageLineVec[n] = 0;
    }
    for (uint64 n = N; n < Nstride; n++)
      dosageLineVec[n] = 0;
  }

  /**    
   * reads indiv info from fam file, snp info from bim file
   * allocates memory, reads genotypes, and does QC
   * assumes numbers of bim and bed files match
   */
  SnpData::SnpData(const string &famFile, const vector <string> &bimFiles,
		   const vector <string> &bedFiles, const string &geneticMapFile,
		   const vector <string> &excludeFiles, const vector <string> &modelSnpsFiles,
		   const vector <string> &removeFiles,
		   double maxMissingPerSnp, double maxMissingPerIndiv, bool noMapCheck,
		   vector <string> vcNamesIn, bool loadNonModelSnps, int _Nautosomes)
    : Mbed(0), Nbed(0), M(0), N(0), Nstride(0), genotypes(NULL),
      genotypeStorageBytes(0), genotypesFileBacked(false), maskSnps(NULL),
      maskIndivs(NULL), numIndivsQC(0), mapAvailable(false), bedIndivsIdentity(false),
      Nautosomes(_Nautosomes) {
    
    processIndivs(famFile, removeFiles, false);
    // bedSnps = all snps in PLINK data; will filter and QC to create class member 'snps'
    vector <uint64> Mfiles;
    vector <SnpInfo> bedSnps = processSnps(Mfiles, bimFiles, excludeFiles, modelSnpsFiles,
					   vcNamesIn, loadNonModelSnps, false);
    processMap(bedSnps, geneticMapFile, noMapCheck);

    // allocate genotypes
    cout << "Allocating " << M << " x " << Nstride << "/4 bytes to store genotypes" << endl;
    genotypes = ALIGNED_MALLOC_UCHARS(M * Nstride/4); // note: M will be reduced after QC
    numIndivsQC = N;

    cout << "Reading genotypes and performing QC filtering on snps and indivs..." << endl;

    // read bed files; build final vector <SnpInfo> snps
    vector <int> numMissingPerIndiv(N);
    uchar *bedLineOut = genotypes;
    uint64 mbed = 0;

    for (uint i = 0; i < bedFiles.size(); i++) {
      if (Mfiles[i] == 0) continue;
      uint64 bytesInFile = Mfiles[i] * (uint64) ((Nbed+3)>>2);
      cout << "Reading bed file #" << (i+1) << ": " << bedFiles[i] << endl;
      cout << "    Expecting " << bytesInFile << " (+3) bytes for "
		<< Nbed << " indivs, " << Mfiles[i] << " snps" << endl;
      FileUtils::AutoGzIfstream fin;
      fin.openOrExit(bedFiles[i], std::ios::in | std::ios::binary);
      uchar header[3];
      fin.read((char *) header, 3);
      if (!fin || header[0] != 0x6c || header[1] != 0x1b || header[2] != 0x01) {
	cerr << "ERROR: Incorrect first three bytes of bed file: " << bedFiles[i] << endl;
	exit(1);
      }

      // read genotypes
      uchar *genoLine = bedIndivsIdentity ? NULL : ALIGNED_MALLOC_UCHARS(N);
      uchar *bedLineIn = ALIGNED_MALLOC_UCHARS((Nbed+3)>>2);
      int numSnpsFailedQC = 0;
      for (uint64 mfile = 0; mfile < Mfiles[i]; mfile++, mbed++) {
	if (bedIndivsIdentity)
	  fin.read((char *) bedLineIn, (Nbed+3)>>2);
	else
	  readBedLine(genoLine, bedLineIn, fin, bedSnpToGrmIndex[mbed] != -2);
	if (bedSnpToGrmIndex[mbed] != -2) { // not excluded
	  double snpMissing;
	  const double alleleFreq = bedIndivsIdentity ?
	    computeIdentityBedAlleleFreqAndMissing(bedLineIn, &snpMissing) :
	    computeAlleleFreqAndMissing(genoLine, maskIndivs, &snpMissing, true);
	  bool snpPassQC = snpMissing <= maxMissingPerSnp;
	  if (snpPassQC) {
	    if (bedSnpToGrmIndex[mbed] >= 0) { // use in GRM
	      if (bedIndivsIdentity)
		storeIdentityBedLineAndCountMissing(bedLineOut, bedLineIn,
						   numMissingPerIndiv.data());
	      else
		storeBedLineAndCountMissing(bedLineOut, genoLine, numMissingPerIndiv.data());
	      bedLineOut += Nstride>>2;
	      bedSnpToGrmIndex[mbed] = snps.size(); // reassign to final value
	      snps.push_back(bedSnps[mbed]);
	      snps.back().MAF = std::min(alleleFreq, 1.0-alleleFreq);
	    }
	    // if bedSnpToGrmIndex[mbed] == -1 (don't use in GRM), leave as-is
	  }
	  else {
	    bedSnpToGrmIndex[mbed] = -2; // exclude
	    if (numSnpsFailedQC < 5)
	      cout << "Filtering snp " << bedSnps[mbed].ID << ": "
		   << snpMissing << " missing" << endl;
	    numSnpsFailedQC++;
	  }
	}
      }
      ALIGNED_FREE(bedLineIn);
      if (genoLine != NULL) ALIGNED_FREE(genoLine);

      if (numSnpsFailedQC)
	cout << "Filtered " << numSnpsFailedQC << " SNPs with > " << maxMissingPerSnp << " missing"
	     << endl;

      if (!fin || fin.get() != EOF) {
	cerr << "ERROR: Wrong file size or reading error for bed file: "
	     << bedFiles[i] << endl;
	exit(1);
      }
      fin.close();
    }

    finishGenoQc(numMissingPerIndiv, maxMissingPerIndiv);
  }

  void SnpData::finishGenoQc(const vector <int> &numMissingPerIndiv,
			     double maxMissingPerIndiv) {
    M = snps.size();

    // allocate and initialize maskSnps to all good
    maskSnps = ALIGNED_MALLOC_UCHARS(M);
    memset(maskSnps, 1, M*sizeof(maskSnps[0]));
    
    // QC indivs for missingness
    int numIndivsFailedQC = 0;
    for (uint64 n = 0; n < N; n++)
      if (maskIndivs[n] && numMissingPerIndiv[n] > maxMissingPerIndiv * M) {
	maskIndivs[n] = 0;
	numIndivsQC--;
	if (numIndivsFailedQC < 5)
	  cout << "Filtering indiv " << indivs[n].famID << " " << indivs[n].indivID << ": "
	       << numMissingPerIndiv[n] << "/" << M << " missing" << endl;
	numIndivsFailedQC++;
      }
    if (numIndivsFailedQC)
      cout << "Filtered " << numIndivsFailedQC << " indivs with > " << maxMissingPerIndiv
	   << " missing" << endl;
    
    cout << "Total indivs after QC: " << numIndivsQC << endl;
    cout << "Total post-QC SNPs: M = " << M << endl;
    //std::map <int, int> vcSnpCounts;
    vector <int> vcSnpCounts(getNumVCs()+1);
    for (uint64 m = 0; m < M; m++)
      if (snps[m].vcNum >= 1)
	vcSnpCounts[snps[m].vcNum]++;
    for (uint v = 1; v < vcSnpCounts.size(); v++) {
      cout << "  Variance component " << v << ": " << vcSnpCounts[v] << " post-QC SNPs (name: '"
	   << vcNames[v] << "')" << endl;
      if (vcSnpCounts[v] == 0) {
	cerr << "  ERROR: No post-QC SNPs left in component \"" << vcNames[v] << "\"" << endl;
	cerr << "         Remove this component from --modelSnps to perform analysis" << endl;
	cerr << "         Also remove from --remlGuessStr if providing variance estimates" << endl;
	exit(1);
      }
    }
  }

  void SnpData::allocatePgenGenotypes(uint64 bytes, const string &cacheDirIn) {
    genotypeStorageBytes = bytes;
    const uint64 memoryBytes = physicalMemoryBytes();
    const bool forceFileBacked = !cacheDirIn.empty();
    genotypesFileBacked = forceFileBacked || (memoryBytes && bytes > memoryBytes / 2);
    if (!genotypesFileBacked) {
      genotypes = ALIGNED_MALLOC_UCHARS(bytes);
      return;
    }

    const string cacheDir = forceFileBacked ? cacheDirIn : defaultCacheDir();
    struct statvfs fsInfo;
    if (statvfs(cacheDir.c_str(), &fsInfo) != 0) {
      cerr << "ERROR: Unable to inspect PGEN cache directory " << cacheDir << ": "
	   << std::strerror(errno) << endl;
      exit(1);
    }
    const uint64 freeBytes = static_cast<uint64>(fsInfo.f_bavail) * fsInfo.f_frsize;
    if (freeBytes < bytes) {
      cerr << "ERROR: PGEN Stage 1 cache requires " << std::fixed << std::setprecision(1)
	   << bytesToGiB(bytes) << " GiB, but only " << bytesToGiB(freeBytes)
	   << " GiB is available in " << cacheDir << endl;
      exit(1);
    }

    string cacheTemplate = cacheDir;
    if (cacheTemplate.empty() || cacheTemplate.back() != '/') cacheTemplate += '/';
    cacheTemplate += "bolt-pgen-stage1-XXXXXX";
    vector<char> cachePath(cacheTemplate.begin(), cacheTemplate.end());
    cachePath.push_back('\0');
    const int fd = mkstemp(cachePath.data());
    if (fd == -1) {
      cerr << "ERROR: Unable to create PGEN Stage 1 cache in " << cacheDir << ": "
	   << std::strerror(errno) << endl;
      exit(1);
    }

    cout << "Using file-backed PGEN Stage 1 cache in " << cacheDir << " ("
	 << std::fixed << std::setprecision(1) << bytesToGiB(bytes) << " GiB; "
	 << (forceFileBacked ? "requested by --pgenCacheDir" :
	     "packed genotypes exceed half of physical RAM") << ")" << endl;
    if (unlink(cachePath.data()) != 0) {
      const int savedErrno = errno;
      close(fd);
      cerr << "ERROR: Unable to unlink temporary PGEN cache " << cachePath.data()
	   << ": " << std::strerror(savedErrno) << endl;
      exit(1);
    }
    if (bytes > static_cast<uint64>(std::numeric_limits<off_t>::max()) ||
	ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
      const int savedErrno = errno;
      close(fd);
      cerr << "ERROR: Unable to size temporary PGEN cache: "
	   << std::strerror(savedErrno) << endl;
      exit(1);
    }
    void *mapping = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    const int mapErrno = errno;
    close(fd);
    if (mapping == MAP_FAILED) {
      cerr << "ERROR: Unable to map temporary PGEN cache: "
	   << std::strerror(mapErrno) << endl;
      exit(1);
    }
    genotypes = static_cast<uchar *>(mapping);
  }

  SnpData::SnpData(const string &pgenFile, const string &pvarFile,
		   const string &psamFile, const string &geneticMapFile,
		   const vector <string> &excludeFiles,
		   const vector <string> &modelSnpsFiles,
		   const vector <string> &removeFiles,
		   double maxMissingPerSnp, double maxMissingPerIndiv, bool noMapCheck,
		   vector <string> vcNamesIn, bool loadNonModelSnps, int _Nautosomes,
		   const string &pgenCacheDir)
    : Mbed(0), Nbed(0), M(0), N(0), Nstride(0), genotypes(NULL),
      genotypeStorageBytes(0), genotypesFileBacked(false), maskSnps(NULL),
      maskIndivs(NULL), numIndivsQC(0), mapAvailable(false), bedIndivsIdentity(false),
      Nautosomes(_Nautosomes) {

    processIndivs(psamFile, removeFiles, true);
    vector <uint64> Mfiles;
    vector <string> pvarFiles(1, pvarFile);
    vector <SnpInfo> bedSnps = processSnps(Mfiles, pvarFiles, excludeFiles,
					   modelSnpsFiles, vcNamesIn,
					   loadNonModelSnps, true);
    processMap(bedSnps, geneticMapFile, noMapCheck);

    if (N == 0) {
      cerr << "ERROR: No samples remain after applying remove files" << endl;
      exit(1);
    }
    if (Nbed > INT_MAX || Mbed > INT_MAX) {
      cerr << "ERROR: PGEN dimensions exceed the range supported by the bundled reader"
	   << endl;
      exit(1);
    }

    if (M > std::numeric_limits<uint64>::max() / (Nstride/4)) {
      cerr << "ERROR: Packed PGEN genotype dimensions overflow addressable storage" << endl;
      exit(1);
    }
    const uint64 genotypeBytes = M * (Nstride/4);
    cout << "Allocating " << M << " x " << Nstride << "/4 bytes to store genotypes" << endl;
    allocatePgenGenotypes(genotypeBytes, pgenCacheDir);
    numIndivsQC = N;

    vector <int> subsetIndices1based;
    if (!bedIndivsIdentity) {
      subsetIndices1based.reserve(N);
      for (uint64 nbed = 0; nbed < Nbed; nbed++)
	if (bedIndivToRemoveIndex[nbed] != -1)
	  subsetIndices1based.push_back(static_cast<int>(nbed+1));
    }

    ::PgenReader pgenReader;
    pgenReader.Load(pgenFile, static_cast<uint32_t>(Nbed), subsetIndices1based, 1);
    if (pgenReader.GetRawSampleCt() != Nbed) {
      cerr << "ERROR: Number of samples in pgen file (" << pgenReader.GetRawSampleCt()
	   << ") does not match psam file (" << Nbed << ")" << endl;
      exit(1);
    }
    if (pgenReader.GetVariantCt() != Mbed) {
      cerr << "ERROR: Number of variants in pgen file (" << pgenReader.GetVariantCt()
	   << ") does not match pvar file (" << Mbed << ")" << endl;
      exit(1);
    }
    if (pgenReader.GetMaxAlleleCt() != 2) {
      cerr << "ERROR: Only biallelic PGEN variants are supported" << endl;
      exit(1);
    }
    if (pgenReader.DosagePresent())
      cout << "NOTE: PGEN dosages are present; Stage 1 uses hardcalls and ignores dosage overrides"
	   << endl;

    cout << "Reading PGEN hardcalls and performing QC filtering on snps and indivs..." << endl;
    const uint64 packedBytes = (N+3)>>2;
    vector <uchar> pgenLine(packedBytes);
    vector <uchar> bedLine(Nstride/4);
    vector <int> numMissingPerIndiv(N);
    uchar *bedLineOut = genotypes;
    int numSnpsFailedQC = 0;
    for (uint64 mbed = 0; mbed < Mbed; mbed++) {
      if (bedSnpToGrmIndex[mbed] == -2) continue;

      pgenReader.ReadHardcallsPacked(pgenLine.data(), packedBytes, N, 0, mbed, 1);
      PgenUtils::packedPgenToBed(bedLine.data(), pgenLine.data(), N, Nstride);
      double snpMissing;
      const double alleleFreq = computeIdentityBedAlleleFreqAndMissing(bedLine.data(),
							       &snpMissing);
      if (snpMissing <= maxMissingPerSnp) {
	if (bedSnpToGrmIndex[mbed] >= 0) {
	  storeIdentityBedLineAndCountMissing(bedLineOut, bedLine.data(),
					       numMissingPerIndiv.data());
	  bedLineOut += Nstride>>2;
	  bedSnpToGrmIndex[mbed] = snps.size();
	  snps.push_back(bedSnps[mbed]);
	  snps.back().MAF = std::min(alleleFreq, 1.0-alleleFreq);
	}
      }
      else {
	bedSnpToGrmIndex[mbed] = -2;
	if (numSnpsFailedQC < 5)
	  cout << "Filtering snp " << bedSnps[mbed].ID << ": "
	       << snpMissing << " missing" << endl;
	numSnpsFailedQC++;
      }
    }
    pgenReader.Close();

    if (numSnpsFailedQC)
      cout << "Filtered " << numSnpsFailedQC << " SNPs with > " << maxMissingPerSnp
	   << " missing" << endl;
    finishGenoQc(numMissingPerIndiv, maxMissingPerIndiv);
  }

  SnpData::SnpData(const vector < pair <string, string> > &_indivIds,
		   const vector <double> &_maskIndivs, uint64 _Nstride,
		   const string &sampleFile, int _Nautosomes, bool psamFormat) :
    Mbed(0), Nbed(0), M(0), N(_indivIds.size()), Nstride(_Nstride), genotypes(NULL),
    genotypeStorageBytes(0), genotypesFileBacked(false),
    maskSnps(NULL), maskIndivs(NULL), numIndivsQC(0), mapAvailable(false),
    bedIndivsIdentity(false), Nautosomes(_Nautosomes) {
    if (Nstride < N || (Nstride&3) || _maskIndivs.size() != Nstride) {
      cerr << "ERROR: Invalid sample dimensions in Stage 1 model" << endl;
      exit(1);
    }
    maskIndivs = ALIGNED_MALLOC_DOUBLES(Nstride);
    memcpy(maskIndivs, &_maskIndivs[0], Nstride*sizeof(maskIndivs[0]));
    for (uint64 n = 0; n < Nstride; n++) numIndivsQC += maskIndivs[n] != 0;

    FID_IID_to_ind.reserve(N);
    indivs.reserve(N);
    for (uint64 n = 0; n < N; n++) {
      IndivInfo indiv;
      indiv.famID = _indivIds[n].first;
      indiv.indivID = _indivIds[n].second;
      indiv.paternalID = indiv.maternalID = "0";
      indiv.sex = 0;
      indiv.pheno = -9;
      string key = indiv.famID + " " + indiv.indivID;
      if (!FID_IID_to_ind.emplace(key, n).second) {
	cerr << "ERROR: Duplicate sample ID in Stage 1 model: " << key << endl;
	exit(1);
      }
      indivs.push_back(std::move(indiv));
    }
    vcNames.push_back("");

    if (!sampleFile.empty()) {
      vector < pair <string, string> > inputIds;
      if (psamFormat) {
	vector <PgenUtils::SampleInfo> psamIndivs = PgenUtils::readPsamFile(sampleFile);
	inputIds.reserve(psamIndivs.size());
	for (uint64 n = 0; n < psamIndivs.size(); n++)
	  inputIds.push_back(std::make_pair(psamIndivs[n].famID, psamIndivs[n].indivID));
      }
      else {
	FileUtils::AutoGzIfstream fin; fin.openOrExit(sampleFile);
	string line;
	while (getline(fin, line)) {
	  std::istringstream iss(line);
	  string FID, IID, paternalID, maternalID, pheno;
	  int sex;
	  if (!(iss >> FID >> IID >> paternalID >> maternalID >> sex >> pheno)) {
	    cerr << "ERROR: Incorrectly formatted fam file: " << sampleFile << endl;
	    exit(1);
	  }
	  inputIds.push_back(std::make_pair(FID, IID));
	}
	fin.close();
      }

      vector <bool> found(N, false);
      for (uint64 inputInd = 0; inputInd < inputIds.size(); inputInd++) {
	string key = inputIds[inputInd].first + " " + inputIds[inputInd].second;
	std::unordered_map <string, uint64>::const_iterator it = FID_IID_to_ind.find(key);
	if (it == FID_IID_to_ind.end())
	  bedIndivToRemoveIndex.push_back(-1);
	else {
	  if (found[it->second]) {
	    cerr << "ERROR: Duplicate Stage 1 sample in " << (psamFormat ? "psam" : "fam")
		 << " file: " << key << endl;
	    exit(1);
	  }
	  found[it->second] = true;
	  bedIndivToRemoveIndex.push_back(it->second);
	}
      }
      Nbed = bedIndivToRemoveIndex.size();
      for (uint64 n = 0; n < N; n++)
	if (!found[n]) {
	  cerr << "ERROR: Stage 1 sample is missing from Stage 2 "
	       << (psamFormat ? "psam" : "fam") << " file: "
	       << _indivIds[n].first << " " << _indivIds[n].second << endl;
	  exit(1);
	}
      bedIndivsIdentity = Nbed == N;
      for (uint64 nbed = 0; bedIndivsIdentity && nbed < Nbed; nbed++)
	bedIndivsIdentity = bedIndivToRemoveIndex[nbed] == static_cast<int>(nbed);
    }
  }

  SnpData::~SnpData() {
    freeGenotypes();
    if (maskSnps!=NULL) ALIGNED_FREE(maskSnps);
    if (maskIndivs!=NULL) ALIGNED_FREE(maskIndivs);
  }

  void SnpData::freeGenotypes() {
    if (genotypes == NULL) return;
    if (genotypesFileBacked) {
      if (munmap(genotypes, genotypeStorageBytes) != 0)
	cerr << "WARNING: Unable to unmap PGEN Stage 1 cache: "
	     << std::strerror(errno) << endl;
    }
    else
      ALIGNED_FREE(genotypes);
    genotypes = NULL;
    genotypeStorageBytes = 0;
    genotypesFileBacked = false;
  }

  vector <SnpInfo> SnpData::readBimFile(const string &bimFile, int Nauto,
					const std::unordered_set <string> *excludeSnpsPtr) {
    vector <SnpInfo> ret;
    string line;
    FileUtils::AutoGzIfstream fin; fin.openOrExit(bimFile);
    int numOutOfOrder = 0;
    while (getline(fin, line)) {
      std::istringstream iss(line);
      SnpInfo snp; string chrom_str;
      if (!(iss >> chrom_str >> snp.ID >> snp.genpos >> snp.physpos >> snp.allele1 >> snp.allele2))
	{
	cerr << "ERROR: Incorrectly formatted bim file: " << bimFile << endl;
	cerr << "Line " << ret.size()+1 << ":" << endl;
	cerr << line << endl;
	cerr << "Unable to input 6 values (2 string, 1 double, 1 int, 2 string)" << endl;
	exit(1);
      }
      snp.chrom = chrStrToInt(chrom_str, Nauto);
      if (snp.chrom == -1 &&
	  (excludeSnpsPtr==NULL || excludeSnpsPtr->find(snp.ID) == excludeSnpsPtr->end())) {
	cerr << "ERROR: Unknown chromosome code in bim file: " << bimFile << endl;
	cerr << "Line " << ret.size()+1 << ":" << endl;
	cerr << line << endl;
	exit(1);
      }
      if (!ret.empty() &&
	  (snp.chrom < ret.back().chrom ||
	   (snp.chrom == ret.back().chrom && (snp.physpos <= ret.back().physpos ||
					       snp.genpos < ret.back().genpos)))) {
	if (numOutOfOrder < 5) {
	  cerr << "WARNING: Out-of-order snp in bim file: " << bimFile << endl;
	  cerr << "Line " << ret.size()+1 << ":" << endl;
	  cerr << line << endl;
	}
	numOutOfOrder++;
	//exit(1);
      }
      ret.push_back(snp);
    }
    if (numOutOfOrder)
      cerr << "WARNING: Total number of out-of-order snps in bim file: " << numOutOfOrder << endl;
    fin.close();
    return ret;
  }

  uint64 SnpData::getM(void) const { return M; }
  // don't provide getN: don't want the rest of the program to even know N!
  uint64 SnpData::getNstride(void) const { return Nstride; }
  uint64 SnpData::getNbed(void) const { return Nbed; }
  const vector <int> &SnpData::getInputIndivToModelIndex(void) const {
    return bedIndivToRemoveIndex;
  }
  uint64 SnpData::getNumIndivsQC(void) const { return numIndivsQC; }
  const double* SnpData::getMaskIndivs(void) const { return maskIndivs; }
  void SnpData::writeMaskIndivs(double out[]) const {
    memcpy(out, maskIndivs, Nstride*sizeof(maskIndivs[0]));
  }
  void SnpData::writeMaskSnps(uchar out[]) const { memcpy(out, maskSnps, M*sizeof(maskSnps[0])); }
  const vector <SnpInfo> &SnpData::getSnpInfo(void) const { return snps; }
  const vector <int> &SnpData::getBedSnpToGrmIndex(void) const { return bedSnpToGrmIndex; }
  vector <double> SnpData::getFamPhenos(void) const {
    vector <double> phenos(N);
    for (uint64 n = 0; n < N; n++) phenos[n] = indivs[n].pheno;
    return phenos;
  }
  bool SnpData::getMapAvailable(void) const { return mapAvailable; }
  const uchar* SnpData::getGenotypes(void) const { return genotypes; }
  bool SnpData::getGenotypesFileBacked(void) const { return genotypesFileBacked; }
  int SnpData::getNumVCs(void) const { return vcNames.size()-1; }
  std::vector <string> SnpData::getVCnames(void) const { return vcNames; }
  vector < pair <string, string> > SnpData::getIndivIds(void) const {
    vector < pair <string, string> > ids;
    ids.reserve(N);
    for (uint64 n = 0; n < N; n++)
      ids.emplace_back(indivs[n].famID, indivs[n].indivID);
    return ids;
  }
  void SnpData::validateIndivIds(const vector < pair <string, string> > &ids,
				 const string &source) const {
    vector <bool> found(N, false);
    for (uint64 i = 0; i < ids.size(); i++) {
      string key = ids[i].first + " " + ids[i].second;
      std::unordered_map <string, uint64>::const_iterator it = FID_IID_to_ind.find(key);
      if (it != FID_IID_to_ind.end()) {
	if (found[it->second]) {
	  cerr << "ERROR: Duplicate Stage 1 sample in " << source << ": " << key << endl;
	  exit(1);
	}
	found[it->second] = true;
      }
    }
    for (uint64 n = 0; n < N; n++)
      if (!found[n]) {
	cerr << "ERROR: Stage 1 sample is missing from " << source << ": "
	     << indivs[n].famID << " " << indivs[n].indivID << endl;
	exit(1);
      }
  }

  /*
  vector <double> getMAFs(void) const {
    vector <double> MAFs(M);
    for (uint64 m = 0; m < M; m++) MAFs[m] = snps[m].MAF;;
    return MAFs;
  }
  */
  uint64 SnpData::getIndivInd(string &FID, string &IID) const {
    std::unordered_map <string, uint64>::const_iterator it = FID_IID_to_ind.find(FID+" "+IID);
    if (it != FID_IID_to_ind.end())
      return it->second;
    else
      return IND_MISSING;
  }
  
  // check same chrom and within EITHER gen or phys distance
  bool SnpData::isProximal(uint64 m1, uint64 m2, double genWindow, int physWindow) const {
    if (snps[m1].chrom != snps[m2].chrom) return false;
    return (mapAvailable && fabs(snps[m1].genpos - snps[m2].genpos) < genWindow) ||
      abs(snps[m1].physpos - snps[m2].physpos) < physWindow;
  }

  // note: pheno may have size Nstride, but we only output N values
  void SnpData::writeFam(const string &outFile, const vector <double> &pheno) const {
    FileUtils::AutoGzOfstream fout; fout.openOrExit(outFile);
    for (uint64 n = 0; n < N; n++)
      fout << indivs[n].famID << "\t" << indivs[n].indivID << "\t" << indivs[n].paternalID << "\t"
	   << indivs[n].maternalID << "\t" << indivs[n].sex << "\t"
	   << std::setprecision(10) << pheno[n] << endl;
    fout.close();
  }

  /*
  // creates map <string, double> and goes through snpInfo
  // plain text, 2-col: rs, LDscore; # to ignore
  void loadLDscore(const string &file) {
    // clear existing LDscore (set to NaN/-1?)
  }
  
  // use LDscore regression to calibrate stats[]
  void calibrateStats(const double stats[]) const {

  }
  */

  // writes Nstride values to out[]
  // assumes subMaskIndivs is a sub-mask of maskIndivs
  //   (presumably obtained by using writeMaskIndivs and taking a subset)
  // work: 256x4 aligned work array
  void SnpData::buildMaskedSnpVector(double out[], const double subMaskIndivs[], uint64 m,
			     const double lut0129[4], double (*work)[4],
			     bool allIndivsIncluded) const {

    /* lookupBedCode[4] = { value of 2 effect alleles (bed 00 = 2 effect alleles),
                            value of missing          (bed 01 = missing),
			    value of 1 effect allele  (bed 10 = 1 effect allele),
			    value of 0 effect alleles (bed 11 = 0 effect alleles) } */
    double lookupBedCode[4] = {lut0129[2], lut0129[3], lut0129[1], lut0129[0]};
    buildByteLookup(work, lookupBedCode);

    uchar *ptr = genotypes + m * (Nstride>>2);
    if (allIndivsIncluded) {
      const uint64 Nfull = N & ~(uint64) 3;
      for (uint64 n4 = 0; n4 < Nfull; n4 += 4) {
#if defined(USE_SSE)
	_mm_store_pd(&out[n4], _mm_load_pd(&work[*ptr][0]));
	_mm_store_pd(&out[n4+2], _mm_load_pd(&work[*ptr][2]));
#elif defined(USE_NEON)
	vst1q_f64(&out[n4], vld1q_f64(&work[*ptr][0]));
	vst1q_f64(&out[n4+2], vld1q_f64(&work[*ptr][2]));
#else
	memcpy(out + n4, work[*ptr], sizeof(work[0]));
#endif
	ptr++;
      }
      if (Nfull != N) {
	for (uint64 n = Nfull; n < N; n++)
	  out[n] = work[*ptr][n&3];
	for (uint64 n = N; n < Nstride; n++)
	  out[n] = 0;
      }
      return;
    }
    for (uint64 n4 = 0; n4 < Nstride; n4 += 4) {
#if defined(USE_SSE) // todo: add AVX instructions to do all at once?
      __m128d x01 = _mm_load_pd(&work[*ptr][0]);
      __m128d x23 = _mm_load_pd(&work[*ptr][2]);
      __m128d mask01 = _mm_load_pd(&subMaskIndivs[n4]);
      __m128d mask23 = _mm_load_pd(&subMaskIndivs[n4+2]);
      _mm_store_pd(&out[n4], _mm_mul_pd(x01, mask01));
      _mm_store_pd(&out[n4+2], _mm_mul_pd(x23, mask23));
#elif defined(USE_NEON)
      float64x2_t x01 = vld1q_f64(&work[*ptr][0]);
      float64x2_t x23 = vld1q_f64(&work[*ptr][2]);
      float64x2_t mask01 = vld1q_f64(&subMaskIndivs[n4]);
      float64x2_t mask23 = vld1q_f64(&subMaskIndivs[n4+2]);
      vst1q_f64(&out[n4], vmulq_f64(x01, mask01));
      vst1q_f64(&out[n4+2], vmulq_f64(x23, mask23));
#else
      // non-lookup approach
      /*
      uchar g = *ptr;
      out[n4] = lookupBedCode[g&3] * subMaskIndivs[n4];
      out[n4+1] = lookupBedCode[(g>>2)&3] * subMaskIndivs[n4+1];
      out[n4+2] = lookupBedCode[(g>>4)&3] * subMaskIndivs[n4+2];
      out[n4+3] = lookupBedCode[(g>>6)&3] * subMaskIndivs[n4+3];
      */
      memcpy(out + n4, work[*ptr], sizeof(work[0]));
      for (int k = 0; k < 4; k++)
	out[n4+k] *= subMaskIndivs[n4+k];
#endif
      ptr++;
    }
  }

  /**
   * performs fast rough computation of LD Scores among chip snps to use in regression weights
   * approximations:
   * - only uses a subsample of individuals
   * - replaces missing genotypes with means and uses them in dot product
   * - computes adjusted r^2 using 1/n correction instead of baseline
   */
  vector <double> SnpData::estChipLDscores(uint64 sampleSize) const {
    if (sampleSize > numIndivsQC) {
      cerr << "WARNING: Only " << numIndivsQC << " indivs available; using all" << endl;
      sampleSize = numIndivsQC;
    }
    if (sampleSize & 7) {
      sampleSize &= ~(uint64) 7;
      cout << "Reducing sample size to " << sampleSize << " for memory alignment" << endl;
    }
    
    // choose sampleSize indivs from maskIndivs
    uint64 step = numIndivsQC / sampleSize;
    vector <int> indivInds;
    for (uint64 n = 0, nGood = 0; n < Nstride; n++)
      if (maskIndivs[n]) {
	if (nGood % step == 0) {
	  indivInds.push_back(n);
	  if (indivInds.size() == sampleSize)
	    break;
	}
	nGood++;
      }

    vector <double> chipLDscores(M, NAN);
    for (int chrom = 1; chrom <= Nautosomes+1; chrom++)
      estChipLDscoresChrom(chipLDscores, chrom, indivInds);
    return chipLDscores;
  }




  // note n <= N because of potential missing data/masking
  double SnpData::compute_r2(const int x[], const int y[], int dim) const {
    int n = 0;
    double avg_x = 0, avg_y = 0;
    for (int i = 0; i < dim; i++)
      if (x[i] != 9 && y[i] != 9) {
	avg_x += x[i];
	avg_y += y[i];
	n++;
      }
    avg_x /= n; avg_y /= n;
    double sum_xx = 0, sum_xy = 0, sum_yy = 0;
    for (int i = 0; i < dim; i++)
      if (x[i] != 9 && y[i] != 9) {
	double xi = x[i]-avg_x;
	double yi = y[i]-avg_y;
	sum_xx += xi*xi;
	sum_xy += xi*yi;
	sum_yy += yi*yi;
      }
    double r2 = sum_xy*sum_xy / (sum_xx*sum_yy);
    return r2;
    //return r2 - (1-r2)/(n-2); adjusted r2
  }

  // fill Nstride-element array with 0, 1, 2, 9; missing or mask => 9
  void SnpData::fillSnpRow(int x[], uint64 m) const {
    /* lookupBedCode[4] = {value of 0, value of missing, value of 1, value of 2} */
    const int lookupBedCode[4] = {0, 9, 1, 2};
    for (uint64 n4 = 0; n4 < Nstride; n4 += 4) {
      uchar byte = genotypes[m * (Nstride>>2) + (n4>>2)];
      for (int k = 0; k < 4; k++)
	x[n4+k] = maskIndivs[n4+k] ? lookupBedCode[(byte>>(k+k))&3] : 9;
    }
  }

  bool SnpData::augmentLDscores(uint64 m, const vector <int> &mRow, uint64 mp, const vector <int> &mpRow,
		       const vector < std::pair <double, int> > &windows,
		       const vector <double> &alphaMAFdeps, vector <double> &LDscores,
		       vector <double> &windowCounts) const {

    int W = windows.size();
    int A = alphaMAFdeps.size();
    double r2 = compute_r2(&mRow[0], &mpRow[0], Nstride);
    bool foundProximal = false;
    for (int w = 0; w < W; w++)
      if (isProximal(m, mp, windows[w].first, windows[w].second)) {
	foundProximal = true;
	if (!std::isnan(r2)) {
	  for (int a = 0; a < A; a++) {
	    double weight = pow((snps[mp].MAF * (1-snps[mp].MAF)), alphaMAFdeps[a]);
	    LDscores[w*A+a] += weight * r2;
	    windowCounts[w*A+a] += weight;
	  }
	}
      }
    return foundProximal;
  }

  vector <double> SnpData::computeLDscore(uint64 m, const vector < std::pair <double, int> > &windows,
				 vector <double> &windowCounts, double &baselineR2,
				 const RestrictSnpSet &restrictPartnerSet,
				 const vector <double> &alphaMAFdeps) const {

    int W = windows.size();
    int A = alphaMAFdeps.size();
    vector <double> LDscores(W*A);
    windowCounts = vector <double> (W*A);

    vector <int> mRow(Nstride), mpRow(Nstride);
    fillSnpRow(&mRow[0], m);
    
    int winSnps1 = 0, winSnps2 = 0;
    for (int mp = (int) m; mp >= 0; mp--) {
      if (snps[m].chrom != snps[mp].chrom) break;
      if (!restrictPartnerSet.isAllowed(snps[mp])) continue;
      fillSnpRow(&mpRow[0], mp);
      if (!augmentLDscores(m, mRow, mp, mpRow, windows, alphaMAFdeps, LDscores, windowCounts))
	break;
      winSnps1++;
    }
    for (int mp = m+1; mp < (int) M; mp++) {
      if (snps[m].chrom != snps[mp].chrom) break;
      if (!restrictPartnerSet.isAllowed(snps[mp])) continue;
      fillSnpRow(&mpRow[0], mp);
      if (!augmentLDscores(m, mRow, mp, mpRow, windows, alphaMAFdeps, LDscores, windowCounts))
	break;
      winSnps2++;
    }

    // compute baseline average LD to random off-chrom SNPs
    //double maxWindowCount = *std::max_element(windowCounts.begin(), windowCounts.end());
    //uint64 mStep = M / (int) maxWindowCount; // max will be for alpha=0 (actual integer count)
    uint64 mStep = M / (winSnps1 + winSnps2);
    double totOffChrom_r2s = 0;
    int numOffChrom_r2s = 0;
    for (uint64 mp = m % mStep; mp < M; mp += mStep)
      if (snps[mp].chrom != snps[m].chrom) {
	fillSnpRow(&mpRow[0], mp);
	double r2 = compute_r2(&mRow[0], &mpRow[0], Nstride);
	if (!std::isnan(r2)) {
	  totOffChrom_r2s += r2;
	  numOffChrom_r2s++;
	}
      }
    baselineR2 = totOffChrom_r2s / numOffChrom_r2s;
    for (int w = 0; w < W; w++)
      for (int a = 0; a < A; a++)
	LDscores[w*A+a] = (1+baselineR2) * LDscores[w*A+a] - windowCounts[w*A+a] * baselineR2;
    return LDscores;
  }

};
