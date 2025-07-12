#include "../include/host_code.cuh"
#include "pybind11/pybind11.h"
#include <pybind11/numpy.h>
#include <utility>

namespace py = pybind11;

Pair closest_points(const py::array_t<int> &image) {
  auto buf = image.request();
  if (buf.ndim != 2) {
    throw std::runtime_error("Image must be 2-dimensional");
  }

  const int height = buf.shape[0];
  const int width = buf.shape[1];
  const int *const ptr = static_cast<const int *const>(buf.ptr);
  Pair min_pair = get_min_pair(ptr, height, width);

  // Swap coordinates from cuda (x,y) ordering to numpy (row, col)
  std::swap(min_pair.ax, min_pair.ay);
  std::swap(min_pair.bx, min_pair.by);
  return min_pair;
}

PYBIND11_MODULE(closest_points_cuda, m) {

  py::class_<Pair>(m, "Pair")
      .def(py::init<>())
      .def_readwrite("distance", &Pair::distance)
      .def_readwrite("ax", &Pair::ax)
      .def_readwrite("ay", &Pair::ay)
      .def_readwrite("bx", &Pair::bx)
      .def_readwrite("by", &Pair::by);

  m.def("closest_points", &closest_points,
        "Find the closest points between segments 1 and 2 in a 2d image",
        py::arg("image"));
}
