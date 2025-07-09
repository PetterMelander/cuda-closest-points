#include "../include/host_code.cuh"
#include <random>

int main() {
  int img_size = 1024;
  int total_pixels = img_size * img_size;
  int *image = new int[total_pixels];

  // Use random device and mt19937 for reproducibility
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  // Assign 10% to 1, 10% to 2, rest to 0
  for (int i = 0; i < total_pixels; ++i) {
    double r = dis(gen);
    if (r < 0.001)
      image[i] = 1;
    else if (r < 0.002)
      image[i] = 2;
    else
      image[i] = 0;
  }

  Pair min_pair = get_min_pair(image, img_size, img_size);

  printf("Distance: %d\nIndex 1: (%d, %d)\nIndex 2: (%d, %d)\n",
         min_pair.distance, min_pair.ax, min_pair.ay, min_pair.bx, min_pair.by);
  return 0;
}