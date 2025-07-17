# Cuda closest points finder

This is a Cuda application for finding the closest points between masks in images. It has python bindings so that it can be used on numpy arrays. 

## How it works

To maximize efficiency, the algorithm is divided into two steps. The first step selects the pixels of each mask that need to be checked for their distance to other masks. The second step does a brute force calculation of the distance between all relevant pixels.

### Step 1: Indexing of masks

Because only the pixels on the edge of a mask can be the closest points to any other mask, the all edges are first extracted from the image. To prepare for further processing, the value and one-dimensional index of each mask edge pixel are stored to arrays.  This is done with a Cuda kernel that assigns one thread to each pixel. If a mask pixel's 4 closest neighbours are not all part of the same mask, the pixel is an edge pixel and its value and index are saved. The purpose of edge detection is that a pixel on the "inside" of a mask cannot be the closest point to any other mask, and can be safely ignored for the remaining calculations. This can vastly reduce the computational complexity for masks that are "blobby" in shape (have low ratio of circumference to area).

Then, the arrays of pixel indices and corresponding mask values are sorted by mask value using CUB's DeviceRadixSort::SortPairs, using the mask values as key to sort by. The purpose of this is to group the indices by mask so they can be efficiently read from memory, and to be able to find the mask values, sizes, and number of masks from the values array.

Finally, CUB's DeviceRunLengthEncode::Encode is used on the values array to find the mask values, sizes, and number of masks. Since the indices are sorted by value, the subarray containing all indices of a given mask can be found using the mask sizes.

### Step 2: Distance reductions

Handwritten Cuda kernels are used to find the closest pixels for all combinations of masks. The kernels use a brute force method of computing the (squared) distance between all combinations of pixels in two masks. One kernel is launched, asynchronously, for each pair of masks. The reduction to find the pixel pair with minimum distance is done first through a grid stride loop, then a warp shuffle reduction to reduce to one value per block, and finally a separate kernel with one block is launched to find the global minimum.

There are two kernels: one that uses one thread per pixel in the largest mask. Each thread then loops over all pixels in the smaller mask to find its minimum. The other kernel uses one thread per pixel combination. The kernel that uses one thread per pixel pair is generally slower, but can be faster than the other if both masks are very small, due to achieving higher occupancy.

## Python bindings

The main entry point function and its return type is bound to Python using Pybind11 and scikit-build-core. The resulting Python function takes a numpy int32 array and returns a two dimensional grid of pixel pairs of size n by n, where n is the number of masks in the image.