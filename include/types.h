#ifndef TYPES_H
#define TYPES_H

#include <sstream>
#include <string>

struct MinResult {
  int distance;
  int a_idx;
  int b_idx;
};

struct Pair {
  int a;
  int ax;
  int ay;
  int b;
  int bx;
  int by;
  float distance;

  Pair transpose() {
    return Pair{b, bx, by, a, ax, ay, distance};
  }

  std::string to_string() const {
    std::ostringstream oss;
    oss << a << ": (" << ax << ", " << ay << ")\n"
        << b << ": (" << bx << ", " << by << ")\n"
        << "distance: " << distance << "\n";
    return oss.str();
  }
};

#endif