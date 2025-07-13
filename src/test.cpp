#include "../include/host_code.cuh"
#include <iostream>
#include <random>

int main() {
  int img_size = 1024;
  int total_pixels = img_size * img_size;
  int *image = new int[total_pixels];

  // Use random device and mt19937 for reproducibility
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  std::uniform_int_distribution<> dis2(1, 8);

  // Assign 10% to 1, 10% to 2, rest to 0
  for (int i = 0; i < total_pixels; ++i) {
    double r = dis(gen);
    if (r < 0.25)
      image[i] = dis2(gen);
    else
      image[i] = 0;
  }

  std::vector<std::vector<Pair>> pairs = get_pairs(image, img_size, img_size);

  for (auto vec : pairs) {
    for (auto pair : vec) {
      std::cout << pair.a;
    }
  }
}