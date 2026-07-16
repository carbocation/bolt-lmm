/*
   This file is part of the BOLT-LMM linear mixed model software package.
   Copyright (C) 2014-2025 Harvard University.
*/

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>

#include "Stage1Model.hpp"

namespace LMM {

  using std::cerr;
  using std::endl;
  using std::string;
  using std::vector;

  namespace {
    const char MAGIC[8] = {'B','O','L','T','S','T','G','1'};
    const uint32_t VERSION = 1;
    const uint32_t ENDIAN_MARKER = 0x01020304;
    const uint64 MAX_COUNT = 1ULL<<40;

    void fail(const string &file, const string &message) {
      cerr << "ERROR: Unable to use Stage 1 model '" << file << "': " << message << endl;
      exit(1);
    }

    struct CheckedWriter {
      std::ofstream out;
      uint64 hash;
      string file;
      explicit CheckedWriter(const string &_file) : hash(14695981039346656037ULL), file(_file) {
	out.open(file.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
	if (!out) fail(file, "could not open temporary output file");
      }
      void bytes(const void *data, size_t size) {
	out.write((const char *) data, size);
	if (!out) fail(file, "write failed");
	const unsigned char *p = (const unsigned char *) data;
	for (size_t i = 0; i < size; i++) { hash ^= p[i]; hash *= 1099511628211ULL; }
      }
      template <class T> void value(const T &x) { bytes(&x, sizeof(x)); }
      void text(const string &x) {
	if (x.size() > std::numeric_limits<uint32_t>::max()) fail(file, "string is too long");
	uint32_t size = x.size(); value(size);
	if (size) bytes(x.data(), size);
      }
      void finish() {
	out.write((const char *) &hash, sizeof(hash));
	out.close();
	if (!out) fail(file, "could not finish output file");
      }
    };

    struct CheckedReader {
      std::ifstream in;
      uint64 hash;
      string file;
      explicit CheckedReader(const string &_file) : hash(14695981039346656037ULL), file(_file) {
	in.open(file.c_str(), std::ios::in | std::ios::binary);
	if (!in) fail(file, "could not open file");
      }
      void bytes(void *data, size_t size) {
	in.read((char *) data, size);
	if (!in) fail(file, "file is truncated");
	const unsigned char *p = (const unsigned char *) data;
	for (size_t i = 0; i < size; i++) { hash ^= p[i]; hash *= 1099511628211ULL; }
      }
      template <class T> T value() { T x; bytes(&x, sizeof(x)); return x; }
      string text() {
	uint32_t size = value<uint32_t>();
	if (size > (1U<<24)) fail(file, "invalid string length");
	string x(size, '\0');
	if (size) bytes(&x[0], size);
	return x;
      }
      void finish() {
	uint64 expected;
	in.read((char *) &expected, sizeof(expected));
	if (!in) fail(file, "file is truncated before checksum");
	if (expected != hash) fail(file, "checksum mismatch (file is corrupt or incomplete)");
	if (in.get() != EOF) fail(file, "unexpected trailing data");
      }
    };

    uint64 checkedProduct(const string &file, uint64 a, uint64 b) {
      if (a > MAX_COUNT || b > MAX_COUNT || (b && a > MAX_COUNT/b))
	fail(file, "invalid array dimensions");
      return a*b;
    }
  }

  void Stage1Model::save(const string &file, const SnpData &snpData, const Bolt &bolt,
			 const vector <Bolt::StatsDataRetroLOCO> &retroData, int Nautosomes) {
    if (file.empty()) fail(file, "output path is empty");
    if (retroData.empty()) fail(file, "no association statistics were produced");
    const string tmpFile = file + ".tmp";
    CheckedWriter out(tmpFile);
    out.bytes(MAGIC, sizeof(MAGIC));
    out.value(VERSION);
    out.value(ENDIAN_MARKER);

    vector < std::pair <string, string> > ids = snpData.getIndivIds();
    uint64 N = ids.size();
    uint64 Nstride = snpData.getNstride();
    uint64 Cindep = bolt.getCovBasis().getCindep();
    int32_t Nauto = Nautosomes;
    uint32_t numStats = retroData.size();
    out.value(N); out.value(Nstride); out.value(Cindep); out.value(Nauto); out.value(numStats);
    for (uint64 n = 0; n < N; n++) { out.text(ids[n].first); out.text(ids[n].second); }
    out.bytes(bolt.getMaskIndivs(), Nstride*sizeof(double));
    if (Cindep) out.bytes(bolt.getCovBasis().getBasis(false), Cindep*Nstride*sizeof(double));

    for (uint32_t s = 0; s < numStats; s++) {
      const Bolt::StatsDataRetroLOCO &data = retroData[s];
      if (data.calibratedResids.empty() || data.snpChunkEnds.empty())
	fail(file, "association statistic has no LOCO residuals or chunk boundaries");
      uint32_t chunks = data.calibratedResids.size();
      uint32_t boundaries = data.snpChunkEnds.size();
      uint32_t scales = data.VinvScaleFactors.size();
      out.text(data.statName); out.value(chunks); out.value(boundaries); out.value(scales);
      for (uint32_t c = 0; c < chunks; c++) {
	if (data.calibratedResids[c].size() != Nstride)
	  fail(file, "LOCO residual length does not match sample stride");
	out.bytes(&data.calibratedResids[c][0], Nstride*sizeof(double));
      }
      for (uint32_t b = 0; b < boundaries; b++) {
	int32_t chrom = data.snpChunkEnds[b].chrom;
	int32_t physpos = data.snpChunkEnds[b].physpos;
	int32_t chunk = data.snpChunkEnds[b].chunk;
	out.value(chrom); out.value(physpos); out.value(chunk);
      }
      if (scales) out.bytes(&data.VinvScaleFactors[0], scales*sizeof(double));
    }
    out.finish();
    if (std::rename(tmpFile.c_str(), file.c_str()) != 0)
      fail(file, "could not atomically move temporary model into place");
  }

  Stage1Model Stage1Model::load(const string &file) {
    CheckedReader in(file);
    char magic[sizeof(MAGIC)]; in.bytes(magic, sizeof(magic));
    if (memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) fail(file, "not a BOLT Stage 1 model");
    uint32_t version = in.value<uint32_t>();
    if (version != VERSION) fail(file, "unsupported model version");
    if (in.value<uint32_t>() != ENDIAN_MARKER) fail(file, "incompatible byte order");

    Stage1Model model;
    uint64 N = in.value<uint64>();
    model.Nstride = in.value<uint64>();
    model.Cindep = in.value<uint64>();
    model.Nautosomes = in.value<int32_t>();
    uint32_t numStats = in.value<uint32_t>();
    if (N == 0 || N > model.Nstride || (model.Nstride&3) || model.Nstride > MAX_COUNT ||
	model.Cindep == 0 || model.Cindep > N || model.Nautosomes <= 0 ||
	model.Nautosomes > 1000 || numStats == 0 || numStats > 100)
      fail(file, "invalid dimensions");
    checkedProduct(file, model.Cindep, model.Nstride);

    model.indivIds.reserve(N);
    std::set < std::pair <string, string> > seenIds;
    for (uint64 n = 0; n < N; n++) {
      string FID = in.text();
      string IID = in.text();
      if (FID.empty() || IID.empty() || !seenIds.insert(std::make_pair(FID, IID)).second)
	fail(file, "invalid or duplicate sample ID");
      model.indivIds.push_back(std::make_pair(FID, IID));
    }
    model.maskIndivs.resize(model.Nstride);
    if (model.Nstride) in.bytes(&model.maskIndivs[0], model.Nstride*sizeof(double));
    for (uint64 n = 0; n < model.Nstride; n++)
      if ((model.maskIndivs[n] != 0 && model.maskIndivs[n] != 1) ||
	  (n >= N && model.maskIndivs[n] != 0))
	fail(file, "invalid sample mask");
    model.covBasis.resize(model.Cindep*model.Nstride);
    if (!model.covBasis.empty())
      in.bytes(&model.covBasis[0], model.covBasis.size()*sizeof(double));

    for (uint32_t s = 0; s < numStats; s++) {
      string statName = in.text();
      uint32_t chunks = in.value<uint32_t>();
      uint32_t boundaries = in.value<uint32_t>();
      uint32_t scales = in.value<uint32_t>();
      if (chunks == 0 || chunks > 100000 || boundaries == 0 || boundaries > 200000 ||
	  scales > 100000)
	fail(file, "invalid statistic dimensions");
      checkedProduct(file, chunks, model.Nstride);
      vector < vector <double> > resids(chunks, vector <double> (model.Nstride));
      for (uint32_t c = 0; c < chunks; c++)
	if (model.Nstride) in.bytes(&resids[c][0], model.Nstride*sizeof(double));
      vector <Bolt::SnpChunkBoundary> chunkEnds;
      chunkEnds.reserve(boundaries);
      for (uint32_t b = 0; b < boundaries; b++) {
	int32_t chrom = in.value<int32_t>();
	int32_t physpos = in.value<int32_t>();
	int32_t chunk = in.value<int32_t>();
	if (chrom <= 0 || chrom > model.Nautosomes+1 || physpos < 0 ||
	    chunk < 0 || (uint32_t) chunk >= chunks)
	  fail(file, "invalid LOCO chunk boundary");
	chunkEnds.push_back(Bolt::SnpChunkBoundary(chrom, physpos, chunk));
      }
      vector <double> scaleFactors(scales);
      if (scales) in.bytes(&scaleFactors[0], scales*sizeof(double));
      if (scales != 0 && scales != 1 && scales != chunks)
	fail(file, "calibration scale count does not match LOCO chunks");
      for (uint32_t i = 0; i < scales; i++)
	if (!std::isfinite(scaleFactors[i])) fail(file, "invalid calibration scale");
      model.retroData.push_back(Bolt::StatsDataRetroLOCO(statName, vector <double> (), resids,
							chunkEnds, scaleFactors));
    }
    in.finish();
    return model;
  }
}
