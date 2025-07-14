import closest_points_cuda
import numpy as np
import random
import timeit

# --- Create some test data ---
num_masks = 32
proportion_masked = 0.01
img_size = 1024

image = np.full((img_size, img_size), 0, dtype=np.int32)
for i in range(img_size):
    for j in range(img_size):
        rand_float = random.random()
        rand_val = random.randint(1, num_masks)
        if rand_float < proportion_masked:
            image[i, j] = rand_val
# image[0,0] = 1
# image[0, 512] = 2
# image[512, 0] = 3


result_pairs = closest_points_cuda.closest_points(image)

print("\n--- Results ---")
print(result_pairs[0][1])

# for i in range(100):
#     print(i)
#     closest_points_cuda.closest_points(image)

num_runs = 100
execution_time = timeit.timeit(lambda: closest_points_cuda.closest_points(image), number=num_runs)
print(execution_time / num_runs)
