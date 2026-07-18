/*
   This file is part of the BOLT-LMM software package developed by Po-Ru Loh.
   Copyright (C) 2014-2025 Harvard University.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
*/

#ifndef PARALLELMEMCOPY_HPP
#define PARALLELMEMCOPY_HPP

#include <cstddef>

namespace LMM {

  void parallelMemcpy(void *destination, const void *source, std::size_t bytes);

}

#endif
