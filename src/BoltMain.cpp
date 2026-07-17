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

#include <vector>
#include <string>
#include <set>
#include <iostream>
#include <memory>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <utility>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "OpenMpCompat.hpp"

#include <boost/utility.hpp>
#include <boost/version.hpp>
#include <boost/numeric/ublas/matrix.hpp>

#include "BoltParams.hpp"
#include "BoltParEstCV.hpp"
#include "PhenoBuilder.hpp"
#include "MemoryUtils.hpp"
#include "Timer.hpp"
#include "FileUtils.hpp"
#include "NumericUtils.hpp"
#include "LDscoreCalibration.hpp"
#include "LapackConst.hpp"
#include "Stage1Model.hpp"
#ifdef BOLT_USE_CUDA
#include "CudaStep1.hpp"
#endif


using namespace LMM;
using namespace std;
namespace ublas = boost::numeric::ublas;
using FileUtils::getline;

namespace {

  struct TokenSpan {
    size_t begin;
    size_t end;
  };

  bool findSelectedColumns(const string &line, const int columns[], size_t count,
			   TokenSpan spans[]) {
    bool found[3] = {false, false, false};
    if (count > 3) return false;

    size_t pos = 0;
    int column = 0;
    size_t remaining = count;
    while (pos < line.size()) {
      while (pos < line.size() && (line[pos] == ' ' || line[pos] == '\t')) pos++;
      if (pos == line.size()) break;

      const size_t begin = pos;
      while (pos < line.size() && line[pos] != ' ' && line[pos] != '\t') pos++;
      for (size_t i = 0; i < count; i++) {
	if (!found[i] && columns[i] == column) {
	  spans[i].begin = begin;
	  spans[i].end = pos;
	  found[i] = true;
	  remaining--;
	}
      }
      if (remaining == 0) return true;
      column++;
    }
    return false;
  }

  uint64 hashToken(const string &line, const TokenSpan &span) {
    uint64 hash = 14695981039346656037ULL;
    for (size_t pos = span.begin; pos < span.end; pos++) {
      hash ^= static_cast<unsigned char>(line[pos]);
      hash *= 1099511628211ULL;
    }
    return hash;
  }

  bool tokenEquals(const string &line, const TokenSpan &span, const string &value) {
    const size_t length = span.end - span.begin;
    return value.size() == length &&
      std::memcmp(line.data() + span.begin, value.data(), length) == 0;
  }

  int parseIntToken(const string &line, const TokenSpan &span) {
    errno = 0;
    char *end = NULL;
    const long value = std::strtol(line.c_str() + span.begin, &end, 10);
    if (end == line.c_str() + span.begin || errno == ERANGE ||
	value < std::numeric_limits<int>::min() ||
	value > std::numeric_limits<int>::max()) {
      cerr << "ERROR: Unable to parse integer in LD Score file line: " << line << endl;
      exit(1);
    }
    return static_cast<int>(value);
  }

  double parseDoubleToken(const string &line, const TokenSpan &span) {
    errno = 0;
    char *end = NULL;
    const double value = std::strtod(line.c_str() + span.begin, &end);
    if (end == line.c_str() + span.begin || errno == ERANGE) {
      cerr << "ERROR: Unable to parse LD Score in line: " << line << endl;
      exit(1);
    }
    return value;
  }

  uint64 chrBpKey(int chrom, int bp) {
    return (static_cast<uint64>(static_cast<uint32_t>(chrom)) << 32) |
      static_cast<uint32_t>(bp);
  }

}

static void runStage2(const BoltParams &params, const Bolt &bolt,
		      const vector <Bolt::StatsDataRetroLOCO> &retroData, Timer &timer) {
  if (!params.statsFile.empty() && params.statsFile != "/dev/null") {
    cout << endl << "=== Streaming " << (params.pgenFile.empty() ? "PLINK" : "PGEN hardcall")
	 << " genotypes and writing association stats ===" << endl;
    if (params.pgenFile.empty())
      bolt.streamComputeRetroLOCO(params.statsFile, params.bimFiles, params.bedFiles,
				  params.geneticMapFile, params.excludeFiles,
				  params.maxMissingPerSnp, params.verboseStats, retroData);
    else
      bolt.streamComputeRetroLOCOPgen(params.statsFile, params.pgenFile, params.pvarFile,
				      params.geneticMapFile, params.excludeFiles,
				      params.maxMissingPerSnp, params.verboseStats, retroData);
    cout << endl << "Time for " << (params.pgenFile.empty() ? "PLINK" : "PGEN hardcall")
	 << " association readout = " << timer.update_time()
	 << " sec" << endl << endl;
  }

  if (!params.statsFileDosageSnps.empty()) {
    cout << endl << "=== Streaming dosage genotypes and writing association stats ===" << endl;
    bolt.streamDosages(params.statsFileDosageSnps, params.dosageFiles, params.dosageFidIidFile,
		       params.geneticMapFile, params.verboseStats, retroData);
    cout << endl << "Time for dosage association readout = " << timer.update_time()
	 << " sec" << endl << endl;
  }

  if (!params.statsFileImpute2Snps.empty()) {
    cout << endl << "=== Streaming IMPUTE2 genotypes and writing association stats ===" << endl;
    bolt.fastStreamImpute2(params.statsFileImpute2Snps, params.impute2Files,
			   params.impute2Chroms, params.impute2FidIidFile,
			   params.impute2MinMAF, params.geneticMapFile,
			   params.verboseStats, retroData, params.domRecHetTest);
    cout << endl << "Time for IMPUTE2 association readout = " << timer.update_time()
	 << " sec" << endl << endl;
  }

  if (!params.statsFileBgenSnps.empty()) {
    cout << endl << "=== Streaming BGEN genotypes and writing association stats ===" << endl;
    for (uint f = 0; f < params.bgenFiles.size(); f++) {
      cout << endl << "BGEN file: " << params.bgenFiles[f] << endl;
      if (params.bgenLayouts[f] == 1)
	bolt.streamBgen(params.statsFileBgenSnps, f, params.bgenFiles[f], params.sampleFiles[f],
			params.bgenMinMAF, params.bgenMinINFO, params.geneticMapFile,
			params.verboseStats, retroData, params.domRecHetTest);
      else
	bolt.streamBgen2(params.statsFileBgenSnps, f, params.bgenFiles[f], params.sampleFiles[f],
			 params.bgenMinMAF, params.bgenMinINFO, params.bgenMinMAC,
			 params.geneticMapFile, params.verboseStats, retroData,
			 params.domRecHetTest, params.bgenRefFirst, params.numThreads);
    }
    cout << endl << "Time for BGEN association readout = " << timer.update_time()
	 << " sec" << endl << endl;
  }

  if (!params.statsFileDosage2Snps.empty()) {
    cout << endl << "=== Streaming dosage2 genotypes and writing association stats ===" << endl;
    bolt.streamDosage2(params.statsFileDosage2Snps, params.dosage2MapFiles,
		       params.dosage2GenoFiles, params.verboseStats, retroData);
    cout << endl << "Time for dosage2 association readout = " << timer.update_time()
	 << " sec" << endl << endl;
  }
}

int main(int argc, char *argv[]) {

  Timer timer; double start_time = timer.get_time();

  cout << "                      +-----------------------------+" << endl;
  cout << "                      |                       ___   |" << endl;
  cout << "                      |   BOLT-LMM, v2.5     /_ /   |" << endl;
  cout << "                      |   June 21, 2025       /_/   |" << endl;
  cout << "                      |   Po-Ru Loh            //   |" << endl;
  cout << "                      |                        /    |" << endl;
  cout << "                      +-----------------------------+" << endl;
  cout << endl;
  cout << "Copyright (C) 2014-2025 Harvard University." << endl;
  cout << "Distributed under the GNU GPLv3 open source license." << endl << endl;

#ifdef VERBOSE
#ifdef BOLT_USE_OPENMP
  cout << "Compiled with OpenMP multithreading" << endl;
#else
  cout << "Compiled without OpenMP: serial execution only" << endl;
#endif
#ifdef USE_SSE
  cout << "Compiled with USE_SSE: fast aligned memory access" << endl;
#endif
#ifdef USE_NEON
  cout << "Compiled with USE_NEON: fast packed genotype decoding" << endl;
#endif
#ifdef BOLT_USE_CUDA
  cout << "Compiled with CUDA Step 1 acceleration" << endl;
#endif
#ifdef USE_MKL
  cout << "Compiled with USE_MKL: Intel Math Kernel Library linear algebra" << endl;
#endif
#ifdef USE_MKL_MALLOC
  cout << "Compiled with USE_MKL_MALLOC: Intel MKL memory allocation" << endl;
#endif
  cout << "Boost version: " << BOOST_LIB_VERSION << endl;
  cout << endl;
#endif

  printf("Command line options:\n\n");
  printf("%s ", argv[0]);
  for (int i = 1; i < argc; i++) {
    if (strlen(argv[i]) >= 2 && argv[i][0] == '-' && argv[i][1] == '-')
      printf("\\\n    ");
    bool hasSpace = false;
    for (int j = 0; j < (int) strlen(argv[i]); j++)
      if (isspace(argv[i][j]))
	hasSpace = true;
    if (hasSpace) {
      if (argv[i][0] == '-') {
	bool foundEquals = false;
	for (int j = 0; j < (int) strlen(argv[i]); j++) {
	  printf("%c", argv[i][j]);
	  if (argv[i][j] == '=' && !foundEquals) {
	    printf("\"");
	    foundEquals = true;
	  }
	}
	printf("\" ");
      }
      else
	printf("\"%s\" ", argv[i]);
    }
    else
      printf("%s ", argv[i]);
  }
  cout << endl << endl;

  BoltParams params;
  if (!params.processCommandLineArgs(argc, argv)) {
    cerr << "Aborting due to error processing command line arguments" << endl;
    cerr << "For list of arguments, run with -h (--help) option" << endl;
    exit(1);
  }

#ifdef BOLT_USE_CUDA
  if (params.useCuda) {
    CudaStep1::setPackedCacheLimitGiB(params.cudaCacheGiB);
    CudaStep1::setPackedHostCacheLimitGiB(params.cudaHostCacheGiB);
  }
#endif

  cout << "Setting number of threads to " << params.numThreads << endl;
  omp_set_num_threads(params.numThreads);
#ifdef USE_MKL
  mkl_set_num_threads(params.numThreads);
#endif

  if (params.stage == 0) {
    cout << "Running Step 0 persistent PGEN cache build" << endl;
    cout << "PGEN cache file: " << params.pgenCacheFile << endl;
  }
  else {
    cout << "Running Stage " << params.stage << endl;
    cout << "Stage 1 model: " << params.stage1Model << endl;
  }

  if (params.stage == 2) {
    cout << endl << "=== Loading Stage 1 model ===" << endl << endl;
    Stage1Model model = Stage1Model::load(params.stage1Model);
    if (params.Nautosomes != model.Nautosomes) {
      cerr << "ERROR: --Nautosomes (" << params.Nautosomes
	   << ") does not match the Stage 1 model (" << model.Nautosomes << ")" << endl;
      exit(1);
    }
    cout << "Loaded " << model.indivIds.size() << " samples, " << model.Cindep
	 << " covariate basis vectors, and " << model.retroData.size()
	 << " association statistic(s)" << endl;
    if (!params.verboseStats && model.retroData.size() > 1)
      for (vector <Bolt::StatsDataRetroLOCO>::iterator it = model.retroData.begin();
	   it != model.retroData.end(); ) {
	if (it->statName == "LINREG") it = model.retroData.erase(it);
	else ++it;
      }
    const string &sampleFile = params.pgenFile.empty() ? params.famFile : params.psamFile;
    SnpData snpData(model.indivIds, model.maskIndivs, model.Nstride, sampleFile,
		    model.Nautosomes, !params.pgenFile.empty());
    Bolt bolt(snpData, model.maskIndivs, model.covBasis, model.Cindep, model.Nautosomes,
	      params.bgenVariantsToTest);
    vector<double>().swap(model.maskIndivs);
    vector<double>().swap(model.covBasis);
    vector< pair<string, string> >().swap(model.indivIds);
    cout << "Time for loading Stage 1 model = " << timer.update_time() << " sec" << endl;
    runStage2(params, bolt, model.retroData, timer);
    cout << "Total elapsed time for Stage 2 = " << (timer.get_time() - start_time) << " sec"
	 << endl;
    return 0;
  }
  
  if (!params.pgenFile.empty()) {
    cout << "pgen: " << params.pgenFile << endl;
    cout << "pvar: " << params.pvarFile << endl;
    cout << "psam: " << params.psamFile << endl;
  }
  else {
    cout << "fam: " << params.famFile << endl;
    cout << "bim(s): ";
    for (uint i = 0; i < params.bimFiles.size(); i++) cout << params.bimFiles[i] << endl;
    cout << "bed(s): ";
    for (uint i = 0; i < params.bedFiles.size(); i++) cout << params.bedFiles[i] << endl;
  }

  /***** SET UP SNPDATA *****/

  cout << endl << "=== Reading genotype data ===" << endl << endl;

  std::unique_ptr<SnpData> snpDataPtr;
  if (!params.pgenFile.empty())
    snpDataPtr.reset(new SnpData(params.pgenFile, params.pvarFile, params.psamFile,
				 params.geneticMapFile, params.excludeFiles,
				 params.modelSnpsFiles, params.removeFiles,
				 params.maxMissingPerSnp, params.maxMissingPerIndiv,
			 params.noMapCheck, params.remlGuessVCnames,
			 params.stage == 1 && !params.reml && params.lmmInf, params.Nautosomes,
			 params.pgenCacheDir, params.pgenCacheFile, params.stage == 0));
  else
    snpDataPtr.reset(new SnpData(params.famFile, params.bimFiles, params.bedFiles,
				 params.geneticMapFile, params.excludeFiles,
				 params.modelSnpsFiles, params.removeFiles,
				 params.maxMissingPerSnp, params.maxMissingPerIndiv,
				 params.noMapCheck, params.remlGuessVCnames,
				 !params.reml && params.lmmInf, params.Nautosomes));
  SnpData &snpData = *snpDataPtr;
  const vector <SnpInfo> &snps = snpData.getSnpInfo();
  // snpData status at this point:
  // - N = # of indivs in .fam file not in --remove file
  // - maskIndivs is zero-filled to Nstride; only indivs failing missingness QC are masked
  // - M = # of snps in .bim file not in --exclude file and not failing QC
  // - maskSnps is all-1s M-vector (leave in code; useful if masking eventually needs to be done)

  if (params.stage == 0) {
    cout << "Total elapsed time for Step 0 = " << (timer.get_time() - start_time) << " sec"
	 << endl;
    return 0;
  }

  if ((int) snps.size() > params.maxModelSnps) {
    cerr << "ERROR: Number of SNPs exceeds maxModelSnps = " << params.maxModelSnps << endl;
    cerr << "       Use the --exclude option to reduce the number of SNPs" << endl;
    cerr << "       e.g., perform LD pruning with PLINK and --exclude prune.out" << endl;
    cerr << "       Alternatively, increase --maxModelSnps" << endl;
    cerr << "       (at the expense of computational cost and possibly poor convergence)" << endl;
    exit(1);
  }

  cout << "Time for SnpData setup = " << timer.update_time() << " sec" << endl << endl;

  cout << "=== Reading phenotype and covariate data ===" << endl << endl;

  vector <string> covarColNames;
  for (uint i = 0; i < params.covarCols.size(); i++)
    covarColNames.push_back(params.covarCols[i].first);
  DataMatrix covarDataT(params.covarFile, snpData, covarColNames);

  vector <double> maskIndivs(snpData.getNstride());
  snpData.writeMaskIndivs(&maskIndivs[0]);

  // note: pheno vector(s) not masked, but maskIndivs is updated to reflect missing phenos
  vector < vector <double> > phenoVecs; double phenoMissingKey = -9;
  if (!params.phenoFile.empty()) { // phenotypes provided in file
    phenoVecs.resize(params.phenoCols.size());
    DataMatrix phenoDataT(params.phenoFile, snpData, params.phenoCols);
    for (uint p = 0; p < phenoVecs.size(); p++) {
      const string &phenoCol = params.phenoCols[p];
      phenoVecs[p] = phenoDataT.getRowDbl(phenoCol); // assign pheno from column of file
      if (phenoVecs[p].empty()) {
	cerr << "ERROR: Phenotype data matrix does not contain column " << phenoCol << endl;
	exit(1);
      }
    }
    phenoMissingKey = phenoDataT.missing_key_dbl;
  }
  if (params.phenoUseFam) { // single phenotype vector provided in last (6th) column of fam file
    phenoVecs.resize(1);
    phenoVecs[0] = snpData.getFamPhenos();
    phenoMissingKey = -9;
  }
  if (!phenoVecs.empty()) { // phenotypes provided; mask indivs with missing phenotype
    int numGoodIndivs = 0;
    for (uint64 n = 0; n < phenoVecs[0].size(); n++) {
      for (uint p = 0; p < phenoVecs.size(); p++)
	if (phenoVecs[p][n] == phenoMissingKey)
	  maskIndivs[n] = 0;
      numGoodIndivs += (int) maskIndivs[n];
    }
    cout << "Number of indivs with no missing phenotype(s) to use: " << numGoodIndivs << endl;
  }
  // maskIndivs at this point: indivs failing QC or missing phenotype

  BoltParEstCV boltCV(snpData, covarDataT, &maskIndivs[0], params.covarCols,
		      params.covarMaxLevels, params.covarUseMissingIndic, params.mBlockMultX,
		      params.Nautosomes, params.bgenVariantsToTest, params.useCuda);
  const Bolt &bolt(boltCV.getBoltRef());
  cout << "Time for covariate data setup + Bolt initialization = " << timer.update_time()
       << " sec" << endl << endl;
  // bolt.getProjMaskSnps() has final snp mask: killed by projecting out covars
  // bolt.getMaskIndivs() has final indiv mask: failing QC or missing phenotype or missing covars
  // (samples with missing covars are masked unless covarUseMissingIndic is set)

  if (params.numLeaveOutChunks <= 0) // set to number of chroms with >= 1 good snp
    params.numLeaveOutChunks = bolt.getNumChromsProjMask();

  vector <double> simBetas;
  vector <PhenoBuilder::SnpRegionType> simRegions;
  bool simPheno = phenoVecs.empty();

  /***** GENERATE PHENO *****/
  if (simPheno) {
    cout << "=== Generating phenotype vector ===" << endl << endl;
    phenoVecs.resize(1);

    set <int> highH2Chroms;
    for (int c = 1; c <= params.highH2ChromMax; c++) highH2Chroms.insert(c);

    PhenoBuilder phenoBuilder(bolt, (int) (params.seed /*+ 1e9*params.h2causal[0]*/
					   + params.Mcausal
					   + 1e8*params.lambdaRegion + 1e4*params.pRegion),
			      params.effectDist == 0 ? PhenoBuilder::GAUSSIAN
			      : PhenoBuilder::LAPLACE);
    int VCs = snpData.getNumVCs();
    if ((int) params.h2causal.size() != VCs) {
      cerr << "ERROR: # of --h2causal values (" << params.h2causal.size()
	   << ") doesn't match # of VCs (" << VCs << ")" << endl;
      exit(1);
    }
    vector < vector <double> > phenoCausals;
    if (params.lambdaRegion == 0) {
      for (int k = 1; k <= VCs; k++) {
	cout << "Generating genetic effects from VC " << k
	     << " explaining h2=" << params.h2causal[k-1] << " of variance" << endl;
	phenoCausals.push_back(phenoBuilder.genPhenoCausal(simBetas, simRegions, k, params.Mcausal,
							   params.MAFhistFile, params.stdPow,
							   highH2Chroms,
							   params.midChromHalfBufferPhyspos));
      }
    }
    else {
      cout << "Generating genetic effects: h2=" << params.h2causal[0] << ", lambdaRegion="
	   << params.lambdaRegion << ", pRegion=" << params.pRegion << endl;
      phenoCausals.push_back(phenoBuilder.genPhenoCausalRegions(simBetas, simRegions,
								params.lambdaRegion,
								params.pRegion));
    }
      
    vector <double> phenoCandidate = phenoBuilder.genPhenoCandidate(simRegions, params.Mcandidate);
    vector <double> phenoStrat;
    if (!params.phenoStratFile.empty())
      phenoStrat = phenoBuilder.genPhenoStrat(params.phenoStratFile);
    vector <double> phenoEnv = phenoBuilder.genPhenoEnv();
    phenoVecs[0] = phenoBuilder.combinePhenoComps(params.h2causal, params.h2candidate,
						  params.h2strat, phenoCausals, phenoCandidate,
						  phenoStrat, phenoEnv);
    if (!params.phenoOutFile.empty()) snpData.writeFam(params.phenoOutFile, phenoVecs[0]);

    cout << "Time for phenotype generation = " << timer.update_time() << " sec" << endl << endl;
  }

  for (uint p = 0; p < phenoVecs.size(); p++) {
    vector <double> phenoVec;
    const double *subMaskIndivs = bolt.getMaskIndivs();
    for (uint64 n = 0; n < phenoVecs[p].size(); n++)
      if (subMaskIndivs[n])
	phenoVec.push_back(phenoVecs[p][n]);
    pair <double, double> muSigma = NumericUtils::meanStdDev(&phenoVec[0], phenoVec.size());
    cout << "Phenotype " << p+1 << ":   N = " << phenoVec.size() << "   mean = " << muSigma.first
	 << "   std = " << muSigma.second << endl;
  }
  cout << endl;

  double *HinvPhiCovCompVec = ALIGNED_MALLOC_DOUBLES(bolt.getNCstride()); // save for later use

  if (params.reml) {

  /***** ESTIMATE HERITABILITY *****/

  cout << "=== Estimating variance parameters ===" << endl << endl;

  vector <uchar> snpVCnums(snps.size());
  for (uint64 m = 0; m < snps.size(); m++) snpVCnums[m] = (uchar) snps[m].vcNum;
  int VCs = snpData.getNumVCs();
  int D = params.runUnivarRemls ? 1 : phenoVecs.size();
  //cout << "Running " << D << "-trait stochastic REML" << endl << endl;
  int reps = phenoVecs.size() / D;
  for (int r = 0; r < reps; r++) {
    vector < vector <double> > phenoVecsRep(phenoVecs.begin()+r*D, phenoVecs.begin()+(r+1)*D);
    vector < ublas::matrix <double> > Vegs(1+VCs, ublas::identity_matrix <double> (D));
    if (!params.remlGuessVegs.empty())
      Vegs = params.remlGuessVegs;
    else {
      for (int d = 0; d < D; d++) {
	cout << "=== Making initial guesses for phenotype " << (d+1) << " ===" << endl << endl;
	const vector <double> &pheno = phenoVecsRep[d];
	vector <double> h2Guesses(VCs);

	const int remlMCtrialsGuess = 3; // if VCs>1, just use 3 MCtrials to get initial guess
	const double logDeltaTol = 0.01; // TODO: decide
	double sigma2Kest = params.h2gGuess, logDeltaEst;
	logDeltaEst = bolt.estLogDelta(&sigma2Kest, HinvPhiCovCompVec, pheno, remlMCtrialsGuess,
				       logDeltaTol, params.maxIters, params.CGtol, params.seed,
				       params.allowh2g01);
	double h2all = 1 / (1 + exp(logDeltaEst)); // not quite right b/c of proj, but estimate ok
	cout << "h2 with all VCs:     " << h2all << endl;

	if (VCs == 1) h2Guesses[0] = h2all;
	else { // re-estimate (LOVCO) using one step of secant iteration
	  uchar *batchMaskSnps = ALIGNED_MALLOC_UCHARS(snps.size()*VCs);
	  vector <uint64> Mused(VCs);
	  const uchar *projMaskSnps = bolt.getProjMaskSnps();
	  for (uint64 m = 0; m < snps.size(); m++)
	    for (int v = 0; v < VCs; v++) {
	      batchMaskSnps[m*VCs+v] = snpVCnums[m] != v+1 && projMaskSnps[m];
	      Mused[v] += snpVCnums[m] != v+1 && projMaskSnps[m];
	    }
	  vector <double> logDeltasLOVCO =
	    bolt.reEstLogDeltas(pheno, logDeltaEst, batchMaskSnps, &Mused[0], VCs,
				remlMCtrialsGuess, params.maxIters, params.CGtol, params.seed);
	  ALIGNED_FREE(batchMaskSnps);

	  vector <double> h2sLOVCO(VCs);
	  for (int v = 0; v < VCs; v++) {
	    h2sLOVCO[v] = std::max(1e-9, 1 / (1 + exp(logDeltasLOVCO[v])));
	    cout << "h2 leaving out VC " << (v+1) << ": " << h2sLOVCO[v] << endl;
	  }
	  double h2sLOVCOsum = std::accumulate(h2sLOVCO.begin(), h2sLOVCO.end(), 0.0);
	  double minH2vGuess = 1e-9;
	  for (int v = 0; v < VCs; v++) {
	    double h2vGuess = (1 - (VCs-1)*h2sLOVCO[v]/h2sLOVCOsum) * h2all;
	    cout << "guess h2 for VC " << (v+1) << ":   " << h2vGuess;
	    if (h2vGuess <= minH2vGuess) {
	      h2vGuess = minH2vGuess;
	      cout << " (setting to " << minH2vGuess << ")";
	    }
	    cout << endl;
	    h2Guesses[v] = h2vGuess; //log(h2vGuess/(1-h2all));
	  }
	}
	for (int v = 0; v < VCs; v++)
	  Vegs[v+1](d, d) = h2Guesses[v];
	Vegs[0](d, d) = 1-accumulate(h2Guesses.begin(), h2Guesses.end(), 0.0);
      }
    }
    bool usePhenoCorrs = params.remlGuessVegs.empty();
    bolt.remlAI(Vegs, usePhenoCorrs, phenoVecsRep, &snpVCnums[0], params.remlMCtrials,
		params.remlNoRefine?0:100, params.maxIters, params.CGtol, params.seed);
  }

  }
  // end if params.reml
  else {

  /***** COMPUTE ASSOC STATS *****/

  vector <Bolt::StatsDataRetroLOCO> retroData;
  const vector <double> &pheno = phenoVecs[0];

  // LINREG

  int linregInd = -1;
  if (!params.noLinreg) {
    cout << "=== Computing linear regression (LINREG) stats ===" << endl << endl;
    linregInd = retroData.size();
    // Store LINREG by default so Stage 2 can independently choose --verboseStats.
    retroData.push_back(bolt.computeLINREG(pheno));
    cout << "Time for computing LINREG stats = " << timer.update_time() << " sec" << endl
	 << endl;
  }
  else
    cout << "Skipping LINREG statistics (--noLinreg)" << endl << endl;

  // LMM
  
  vector <double> LDscores, LDscoresChip;
  double LINREGinflationEst = 0;
  
  int lmmInfInd = -1; // index of inf. model stats in retroData
  if (params.lmmInf) {

    if (params.numLeaveOutChunks <= 1) {
      cerr << "ERROR: LOCO mixed model analysis requires >= 2 chromosomes or chunks" << endl;
      cerr << "       To run this analysis, set --numLeaveOutChunks" << endl;
      exit(1);
    }
    
    // reml (fast approximation)
    
    cout << "=== Estimating variance parameters ===" << endl << endl;

    double CGtolEst = 10*params.CGtol;
    cout << "Using CGtol of " << CGtolEst << " for this step" << endl;
    double logDeltaTol = 0.01; // TODO: decide
    double sigma2Kest = params.h2gGuess, logDeltaEst; // variance parameters to estimate
    logDeltaEst = bolt.estLogDelta(&sigma2Kest, HinvPhiCovCompVec, pheno, params.h2EstMCtrials,
				   logDeltaTol, params.maxIters, CGtolEst, params.seed,
				   params.allowh2g01);

    cout << "Time for fitting variance components = " << timer.update_time() << " sec" << endl
	 << endl;

    vector <double> sigma2Ks, logDeltas;
    // set variance parameters all to all-chrom estimates if no re-estimation per LOCO rep
    logDeltas = vector <double> (params.numLeaveOutChunks, logDeltaEst);
    sigma2Ks = vector <double> (params.numLeaveOutChunks, sigma2Kest);

    if (params.reEstMCtrials) { // re-estimate variance parameters per LOCO rep
      cout << "=== Re-estimating variance parameters for each left-out group of SNPs ===" << endl
	   << endl;
      bolt.reEstVCs(pheno, logDeltas, sigma2Ks, params.reEstMCtrials, params.genWindow,
		    params.physWindow, params.maxIters, params.CGtol, params.seed);
      cout << endl << "Time for re-fitting variance components = " << timer.update_time() << " sec"
	   << endl << endl;
    }


    // lmmInf

    cout << "=== Computing mixed model assoc stats (inf. model) ===" << endl << endl;

    lmmInfInd = retroData.size();
    retroData.push_back
      (bolt.computeLmmInf(pheno, logDeltas, sigma2Ks, HinvPhiCovCompVec, params.numCalibSnps,
			  params.genWindow, params.physWindow, params.maxIters, params.CGtol,
			  params.seed));
    cout << endl << "Time for computing infinitesimal model assoc stats = " << timer.update_time()
	 << " sec" << endl << endl;
  
    // compute chip LD Scores either for non-inf stat calibration or to estimate LINREG inflation
    // (in the latter case, if ref LD provided)

    if (params.lmmBayes || params.lmmBayesMCMC || !params.LDscoresFile.empty()) {
      int sampleSize = 400;
      cout << "=== Estimating chip LD Scores using " << sampleSize << " indivs ===" << endl
	   << endl;
      LDscoresChip = snpData.estChipLDscores(sampleSize);
      cout << endl << "Time for estimating chip LD Scores = " << timer.update_time() << " sec"
	   << endl << endl;
      /*
	FileUtils::AutoGzOfstream fout; fout.openOrExit("chipLD.txt");
	for (uint64 m = 0; m < LD.size(); m++) fout << LD[m] << endl;
	fout.close();
      */
    }

    // read LDscores file
    if (!params.LDscoresFile.empty()) {
      cout << "=== Reading LD Scores for calibration of Bayesian assoc stats ===" << endl << endl;

      LDscores = vector <double> (snps.size(), NAN); // NAN if missing
      int numFound = 0;
      if (!params.LDscoresMatchBp) {
	unordered_map <uint64, vector <uint> > snpHashToIndices;
	snpHashToIndices.reserve(2 * snps.size());
	for (uint64 m = 0; m < snps.size(); m++)
	  snpHashToIndices[hashToken(snps[m].ID, TokenSpan{0, snps[m].ID.size()})].push_back(m);
	cout << "Looking up LD Scores..." << endl;
	int colSNP = FileUtils::lookupColumnInd(params.LDscoresFile, " \t", "SNP");
	cout << "  Looking for column header 'SNP': column number = " << (colSNP+1) << endl;
	int colLD = FileUtils::lookupColumnInd(params.LDscoresFile, " \t", params.LDscoresCol);
	cout << "  Looking for column header '" << params.LDscoresCol
	     << "': column number = " << (colLD+1) << endl;
	if (colSNP < 0 || colLD < 0) {
	  if (colSNP < 0) cerr << "ERROR: Unable to find column 'SNP'" << endl;
	  if (colLD < 0) cerr << "ERROR: Unable to find column '" << params.LDscoresCol
			      << "'" << endl;
	  exit(1);
	}
	FileUtils::AutoGzIfstream fin; fin.openOrExit(params.LDscoresFile);
	string line;
	getline(fin, line); // get rid of header
	const int columns[2] = {colSNP, colLD};
	while (getline(fin, line)) {
	  TokenSpan spans[2];
	  if (!findSelectedColumns(line, columns, 2, spans)) {
	    cerr << "ERROR: Too few columns in LD Score file line: " << line << endl;
	    exit(1);
	  }
	  unordered_map <uint64, vector <uint> >::const_iterator bucket =
	    snpHashToIndices.find(hashToken(line, spans[0]));
	  if (bucket != snpHashToIndices.end()) {
	    // Reverse order preserves the previous last-duplicate-wins behavior.
	    for (vector <uint>::const_reverse_iterator it = bucket->second.rbegin();
		 it != bucket->second.rend(); ++it) {
	      if (tokenEquals(line, spans[0], snps[*it].ID)) {
		LDscores[*it] = parseDoubleToken(line, spans[1]);
		numFound++;
		break;
	      }
	    }
	  }
	}
	fin.close();
	if (numFound < 1000) {
	  cerr << "ERROR: Found LD Scores for only " << numFound << "/" << LDscores.size()
	       << " SNPs" << endl;
	  cerr << "       If your bim file does not contain rsIDs, try using --LDscoresMatchBp"
	       << endl;
	  exit(1);
	}
      }
      else {
	unordered_map <uint64, uint> snpChrBpToIndex;
	snpChrBpToIndex.reserve(2 * snps.size());
	for (uint64 m = 0; m < snps.size(); m++)
	  snpChrBpToIndex[chrBpKey(snps[m].chrom, snps[m].physpos)] = m;
	cout << "Looking up LD Scores..." << endl;
	int colCHR = FileUtils::lookupColumnInd(params.LDscoresFile, " \t", "CHR");
	cout << "  Looking for column header 'CHR': column number = " << (colCHR+1) << endl;
	int colBP = FileUtils::lookupColumnInd(params.LDscoresFile, " \t", "BP");
	cout << "  Looking for column header 'BP': column number = " << (colBP+1) << endl;
	int colLD = FileUtils::lookupColumnInd(params.LDscoresFile, " \t", params.LDscoresCol);
	cout << "  Looking for column header '" << params.LDscoresCol
	     << "': column number = " << (colLD+1) << endl;
	if (colCHR < 0 || colBP < 0 || colLD < 0) {
	  if (colCHR < 0) cerr << "ERROR: Unable to find column 'CHR'" << endl;
	  if (colBP < 0) cerr << "ERROR: Unable to find column 'BP'" << endl;
	  if (colLD < 0) cerr << "ERROR: Unable to find column '" << params.LDscoresCol
			      << "'" << endl;
	  exit(1);
	}
	FileUtils::AutoGzIfstream fin; fin.openOrExit(params.LDscoresFile);
	string line;
	getline(fin, line); // get rid of header
	const int columns[3] = {colCHR, colBP, colLD};
	while (getline(fin, line)) {
	  TokenSpan spans[3];
	  if (!findSelectedColumns(line, columns, 3, spans)) {
	    cerr << "ERROR: Too few columns in LD Score file line: " << line << endl;
	    exit(1);
	  }
	  const uint64 key = chrBpKey(parseIntToken(line, spans[0]),
				      parseIntToken(line, spans[1]));
	  unordered_map <uint64, uint>::const_iterator it = snpChrBpToIndex.find(key);
	  if (it != snpChrBpToIndex.end()) {
	    LDscores[it->second] = parseDoubleToken(line, spans[2]);
	    numFound++;
	  }
	}
	fin.close();
      }
      cout << "Found LD Scores for " << numFound << "/" << LDscores.size() << " SNPs" << endl;

      if (linregInd != -1) {
	cout << endl << "Estimating inflation of LINREG chisq stats using MLMe as reference..."
	     << endl;
	// TODO: update with final values of constants for LD Score regression
	double minMAF = 0.01;
	double outlierVarFracThresh = 0.001;
	int varianceDegree = 2;
	std::pair <double, double> calibrationFactorMeanStd =
	  LDscoreCalibration::calibrateStatPair(snpData.getSnpInfo(),
						retroData[lmmInfInd].stats,
						retroData[linregInd].stats, LDscores,
						LDscoresChip, minMAF, bolt.getNused(),
						outlierVarFracThresh,
						snpData.getMapAvailable(), varianceDegree);
	LINREGinflationEst = 1/calibrationFactorMeanStd.first;
	cout << "LINREG intercept inflation = " << LINREGinflationEst << endl;
	if (params.verboseStats)
	  cout << "NOTE: LINREG stats in output are NOT corrected for estimated inflation"
	       << endl;
      }
    }
    else if (params.lmmBayes || params.lmmBayesMCMC) { // will need to calibrate with chip LD
      cerr << "WARNING: No LDscoresFile provided; using estimated LD among chip SNPs" << endl;
      LDscores = LDscoresChip;
    }
    cout << endl;

    // lmmBayes[MCMC] mixture param estimation
  
    double pEst = params.pEst, f2Est = params.varFrac2Est;
    int VBiters = 0; // save the number of iters used by VB to set number of MCMC iters
    vector <double> warmRawEffects;
  
    if (params.lmmBayes || params.lmmBayesMCMC) { // estimate (f2, p): CV using VB algorithm
      if (pEst == BoltParams::MIX_PARAM_ESTIMATE_FLAG ||
	  f2Est == BoltParams::MIX_PARAM_ESTIMATE_FLAG) {
	cout << "=== Estimating mixture parameters by cross-validation ===" << endl << endl;
	int maxItersCV = params.maxIters / 2;
	cout << "Setting maximum number of iterations to " << maxItersCV << " for this step"
	     << endl;
	double predBoost = 0, predBoostMin = params.lmmForceNonInf ? -1e9 : 0.01;
	bool useMCMCinCV = false; //params.lmmBayesMCMC;
	VBiters = boltCV.estMixtureParams
	  (&f2Est, &pEst, &predBoost, pheno, logDeltaEst, sigma2Kest, params.CVfoldsSplit,
	   params.CVfoldsCompute, params.CVnoEarlyExit, predBoostMin, useMCMCinCV,
	   maxItersCV, params.approxLLtol, params.mBlockMultX, params.Nautosomes,
	   params.lmmBayes ? &warmRawEffects : NULL); // TODO: change MCMC param back to VB?
	if (predBoost < predBoostMin && !params.lmmForceNonInf) {
	  cout << "Bayesian non-infinitesimal model does not fit substantially better" << endl;
	  cout << "=> Not computing non-inf assoc stats (to override, use --lmmForceNonInf)"
	       << endl;
	  params.lmmBayes = params.lmmBayesMCMC = false;
	}
	cout << endl << "Time for estimating mixture parameters = " << timer.update_time()
	     << " sec" << endl << endl;
      }
    }

    // lmmBayes

    if (params.lmmBayes) {
      cout << "=== Computing Bayesian mixed model assoc stats with mixture prior ==="
	   << endl << endl;
      if (lmmInfInd == -1) { // shouldn't happen
	cerr << "ERROR: Inf. model stats must be computed to calibrate Bayesian stats" << endl;
	exit(1);
      }
      retroData.push_back
	(bolt.computeLmmBayes(pheno, logDeltas, sigma2Ks, f2Est, pEst, false, params.genWindow,
			      params.physWindow, params.maxIters, params.approxLLtol,
			      retroData[lmmInfInd].stats, LDscores, LDscoresChip,
			      warmRawEffects.empty() ? NULL : &warmRawEffects));
      cout << endl << "Time for computing Bayesian mixed model assoc stats = "
	   << timer.update_time() << " sec" << endl << endl;
    }

    // lmmBayesMCMC

    if (params.lmmBayesMCMC) {
      cout << "=== Computing Bayesian mixed model assoc stats using MCMC ==="
	   << endl << endl;    
      if (lmmInfInd == -1) { // shouldn't happen
	cerr << "ERROR: Inf. model stats must be computed to calibrate Bayesian stats" << endl;
	exit(1);
      }
      int MCMCiters;
      if (VBiters) { // VB algorithm previously run during mixture param estimation
	if (params.MCMCiters != 0) {
	  MCMCiters = params.MCMCiters;
	  cout << "Setting number of MCMC iterations to --params.MCMCiters = " << MCMCiters
	       << endl;
	}
	else if (5*VBiters <= params.maxIters) {
	  MCMCiters = 5*VBiters; // set iters to 5 x (VB iters)
	  cout << "Setting number of MCMC iterations to " << MCMCiters << endl;
	}
	else {
	  cerr << "WARNING: maxIters < 5 x # of iters required by variational Bayes" << endl;
	  cerr << "Iterations set to maxIters = " << params.maxIters
	       << " but MCMC may not converge" << endl;
	  MCMCiters = params.maxIters;
	}
      }
      else { // just use maxIters
	cout << "Setting number of MCMC iterations to maxIters = " << params.maxIters << endl;
	MCMCiters = params.maxIters;
      }

      retroData.push_back
	(bolt.computeLmmBayes(pheno, logDeltas, sigma2Ks, f2Est, pEst, true, params.genWindow,
			      params.physWindow, MCMCiters, params.approxLLtol,
			      retroData[lmmInfInd].stats, LDscores, LDscoresChip));
      cout << endl << "Time for computing Bayesian mixed model assoc stats using MCMC = "
	   << timer.update_time() << " sec" << endl << endl;
    }

    if (!params.predBetasFile.empty()) {
      cout << endl << "=== Computing and writing betas for polygenic prediction ===" << endl
	   << endl;

      if (pEst == BoltParams::MIX_PARAM_ESTIMATE_FLAG ||
	  f2Est == BoltParams::MIX_PARAM_ESTIMATE_FLAG)
	pEst = f2Est = 0.5; // infinitesimal model
      bolt.computeWritePredBetas(params.predBetasFile, pheno, logDeltaEst, sigma2Kest, f2Est,
				 pEst, params.lmmBayesMCMC, params.maxIters, params.approxLLtol);

      cout << endl << "Time for computing and writing betas = "
	   << timer.update_time() << " sec" << endl << endl;
    }
  }

  ALIGNED_FREE(HinvPhiCovCompVec);

  cout << "Calibration stats: mean and lambdaGC (over SNPs used in GRM)" << endl;
  cout << "  (note that both should be >1 because of polygenicity)" << endl;
  for (uint64 s = 0; s < retroData.size(); s++) {
    int numSnps = 0;
    double totStats = 0;
    vector <double> sortStats;
    for (uint64 m = 0; m < retroData[s].stats.size(); m++)
      if (retroData[s].stats[m] > 0) { // not a masked-out snp
	totStats += retroData[s].stats[m];
	sortStats.push_back(retroData[s].stats[m]);
	numSnps++;
      }
    std::sort(sortStats.begin(), sortStats.end());
    cout << "Mean " << retroData[s].statName << ": " << totStats / numSnps
	 << " (" << numSnps << " good SNPs)   ";
    cout << "lambdaGC: " << sortStats[numSnps/2]/0.4549364 << endl;
  }
  if (LINREGinflationEst > 1 && params.verboseStats)
    cout << "Note that LINREG may be confounded by a factor of " << LINREGinflationEst << endl;

  /***** FREE GENOTYPES STORED IN RAM (NO LONGER NEEDED) *****/
  snpData.freeGenotypes();

  cout << endl << "=== Writing Stage 1 model ===" << endl << endl;
  Stage1Model::save(params.stage1Model, snpData, bolt, retroData, params.Nautosomes);
  cout << "Wrote Stage 1 model: " << params.stage1Model << endl;
  cout << "Time for writing Stage 1 model = " << timer.update_time() << " sec" << endl << endl;

  // write simulated snp effects (NOT including candidate snps) and LD Scores at GRM snps

  if (!params.snpInfoFile.empty()) {
    FileUtils::AutoGzOfstream fout; fout.openOrExit(params.snpInfoFile);
    fout << std::fixed;
    fout << "SNP" << "\t" << "CHR" << "\t" << "BP" << "\t" << "MAF" << "\t" << "GENPOS";
    if (!LDscoresChip.empty()) fout << "\t" << "CHIPLD";
    if (simPheno) fout << "\t" << "simBeta" << "\t" << "simRegion";
    for (uint64 s = 0; s < retroData.size(); s++)
      fout << "\t" << "CHISQ_" << retroData[s].statName;
    fout << endl;

    for (uint64 m = 0; m < retroData[0].stats.size(); m++) {
      fout << snps[m].ID << "\t" << snps[m].chrom << "\t" << snps[m].physpos << "\t"
	   << snps[m].MAF << "\t" << snps[m].genpos;
      if (!LDscoresChip.empty()) {
	char buf[20]; sprintf(buf, "\t%.2f", LDscoresChip[m]);
	fout << string(buf);
      }
      if (simPheno) fout << "\t" << simBetas[m] << "\t" << simRegions[m];
      for (uint64 s = 0; s < retroData.size(); s++)
	fout << "\t" << retroData[s].stats[m];
      fout << endl;
    }
    fout.close();
  }

  }
  cout << "Total elapsed time for analysis = " << (timer.get_time() - start_time) << " sec"
       << endl;
}
