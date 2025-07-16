#include "../include/host_code.cuh"
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

/**
 * @brief Takes in a numpy array, runs some basic checks, and calls main
 * function.
 *
 * @param image Numpy array.
 * @return std::vector<std::vector<Pair>> 2d array of pixel pairs with minimum
 * distance between masks.
 */
std::vector<std::vector<Pair>>
closest_points(const py::array_t<const int> &image) {
  auto buf = image.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Image must be 2-dimensional");
  }

  const int height = buf.shape[0];
  const int width = buf.shape[1];
  const int *const ptr = static_cast<const int *const>(buf.ptr);
  std::vector<std::vector<Pair>> min_pairs = get_pairs(ptr, height, width);

  return min_pairs;
}

PYBIND11_MODULE(closest_points_cuda, m) {

  py::class_<Pair>(m, "Pair")
      .def(py::init<>())
      .def("__repr__", &Pair::to_string)
      .def_readwrite("distance", &Pair::distance)
      .def_readwrite("a", &Pair::a)
      .def_readwrite("ax", &Pair::ax)
      .def_readwrite("ay", &Pair::ay)
      .def_readwrite("b", &Pair::b)
      .def_readwrite("bx", &Pair::bx)
      .def_readwrite("by", &Pair::by);

  m.def("closest_points", &closest_points,
        "Find the closest points between masks in a 2d image",
        py::arg("image"));
}
