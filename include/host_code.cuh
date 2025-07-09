#ifndef HOST_CODE_H
#define HOST_CODE_H

#include "../include/types.h"

std::tuple<int, int> launch_index_shapes(int *h_image, int img_height,
                                         int img_width, int *d_as, int *d_bs);

Pair launch_min_pair_thread_per_a(int num_as, int num_bs, int img_width,
                                  int *d_as, int *d_bs);

Pair launch_min_pair_thread_per_pair(int num_as, int num_bs, int img_width,
                                     int *d_as, int *d_bs);

Pair get_min_pair(int *h_image, int img_height, int img_width);

#endif