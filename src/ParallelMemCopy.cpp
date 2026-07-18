/*
   This file is part of the BOLT-LMM software package developed by Po-Ru Loh.
   Copyright (C) 2014-2025 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/

#include <cstring>

#include "OpenMpCompat.hpp"
#include "ParallelMemCopy.hpp"

namespace LMM {

  void parallelMemcpy(void *destination, const void *source, std::size_t bytes) {
#ifdef BOLT_USE_OPENMP
    const std::size_t parallelThreshold = 8ULL << 20;
    if (bytes >= parallelThreshold && omp_get_max_threads() > 1) {
      unsigned char *output = static_cast<unsigned char *>(destination);
      const unsigned char *input = static_cast<const unsigned char *>(source);
#pragma omp parallel
      {
        const std::size_t thread = static_cast<std::size_t>(omp_get_thread_num());
        const std::size_t threads = static_cast<std::size_t>(omp_get_num_threads());
        const std::size_t begin = bytes * thread / threads;
        const std::size_t end = bytes * (thread + 1) / threads;
        std::memcpy(output + begin, input + begin, end - begin);
      }
      return;
    }
#endif
    std::memcpy(destination, source, bytes);
  }

}
