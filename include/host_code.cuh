#ifndef HOST_CODE_H
#define HOST_CODE_H

#include "types.h"
#include <vector>

/**
 * @brief For all masks in image, get the pixels with minimum distance between
 * two masks.
 *
 * @param h_image Image array in cpu memory.
 * @param img_height Image height in pixels
 * @param img_width Image width in pixels.
 * @return vector<vector<Pair>> All pixels pairs with minimum distance between
 * masks.
 */
std::vector<std::vector<Pair>>
get_pairs(const int *const h_image, const int img_height, const int img_width);

#endif