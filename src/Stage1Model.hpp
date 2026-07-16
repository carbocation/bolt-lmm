/*
   This file is part of the BOLT-LMM linear mixed model software package.
   Copyright (C) 2014-2025 Harvard University.
*/

#ifndef STAGE1MODEL_HPP
#define STAGE1MODEL_HPP

#include <string>
#include <utility>
#include <vector>

#include "Bolt.hpp"

namespace LMM {

  struct Stage1Model {
    int Nautosomes;
    uint64 Nstride;
    uint64 Cindep;
    std::vector < std::pair <std::string, std::string> > indivIds;
    std::vector <double> maskIndivs;
    std::vector <double> covBasis;
    std::vector <Bolt::StatsDataRetroLOCO> retroData;

    static void save(const std::string &file, const SnpData &snpData, const Bolt &bolt,
		     const std::vector <Bolt::StatsDataRetroLOCO> &retroData,
		     int Nautosomes);
    static Stage1Model load(const std::string &file);
  };
}

#endif
