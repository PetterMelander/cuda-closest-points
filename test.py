import closest_points_cuda
import numpy as np
import random

# --- Create some test data ---
img_size = 1024
image = np.full((img_size, img_size), 0, dtype=np.int32)
for i in range(img_size):
    for j in range(img_size):
        rand_val = random.random()
        if rand_val < 0.1:
            image[i, j] = 1
        elif rand_val < 0.2:
            image[i, j] = 2
# image[0,0] = 1
# image[256, 512] = 2

result_pair = closest_points_cuda.closest_points(image)

print("\n--- Results ---")
print(f"Calculated distance: {result_pair.distance}")
print(f"Point A: ({result_pair.ax}, {result_pair.ay})")
print(f"Point B: ({result_pair.bx}, {result_pair.by})")
