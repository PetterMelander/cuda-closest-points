#include "../include/host_code.cuh"
// #include "../include/kernels.cuh"
#include <cstring>
#include <iostream>
#include <random>

int main(int argc, char *argv[]) {
  int img_size = 1024;
  int total_pixels = img_size * img_size;
  int *image = new int[total_pixels];

  // Use random device and mt19937 for reproducibility
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_int_distribution<> dis2(1, 32);

  // Assign 10% to 1, 10% to 2, rest to 0
  for (int i = 0; i < total_pixels; ++i) {
    double r = dis(gen);
    if (r < 0.2)
      image[i] = dis2(gen);
    else
      image[i] = 0;
  }
  // image[0] = 1;
  // image[1023] = 2;
  // image[255 * 1024 + 255] = 1;
  // image[255 * 1024 + 767] = 2;
  // image[767 * 1024 + 255] = 3;

  // for (int i = 0; i < 10; ++i) {
  // get_pairs(image, img_size, img_size);
  // }

  std::vector<std::vector<Pair>> pairs = get_pairs(image, img_size, img_size);

  for (auto vec : pairs) {
    for (auto pair : vec) {
      std::cout << pair.to_string();
    }
  }
  // int img_size = 10;
  // int total_pixels = img_size * img_size;
  // int *image = new int[total_pixels];

  // for (int i = 0; i < img_size; ++i) {
  //   for (int j = 0; j < img_size; ++j) {
  //     if (i < 3 && j < 3)
  //       image[i + j * img_size] = 1;
  //     else
  //       image[i + j * img_size] = 0;
  //   }
  // }

  // int *d_image, *d_output;
  // cudaMalloc(&d_image, sizeof(int) * total_pixels);
  // cudaMalloc(&d_output, sizeof(int) * total_pixels);
  // cudaMemcpy(d_image, image, sizeof(int) * total_pixels,
  //            cudaMemcpyHostToDevice);
  // dim3 block_size(32, 16);
  // uint num_blocks_x = (img_size + 30 - 1) / 30;
  // uint num_blocks_y = (img_size + 14 - 1) / 14;
  // dim3 grid_size(num_blocks_x, num_blocks_y);
  // find_edges<<<grid_size, block_size>>>(d_image, img_size, img_size,
  // d_output);

  // int *output = new int[total_pixels];
  // for (int i = 0; i < total_pixels; ++i) {
  //   output[i] = -1;
  // }
  // cudaMemcpy(output, d_output, sizeof(int) * total_pixels,
  //            cudaMemcpyDeviceToHost);

  // for (int i = 0; i < img_size; ++i) {
  //   for (int j = 0; j < img_size; ++j) {
  //     if (i < 3 && j < 3) {
  //       std::cout << "value at (" << i << ", " << j
  //                 << "): " << output[i + j * img_size] << "\n";
  //     } else {
  //       if (output[i + j * img_size] != 0) {
  //         std::cout << "error at " << i << ", " << j << "\n";
  //       }
  //     }
  //   }
  // }
}