#ifndef HOST_CODE_H
#define HOST_CODE_H

#include "types.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

std::vector<std::vector<Pair>>
get_pairs(const int *const h_image, const int img_height, const int img_width);

#endif