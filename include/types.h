#ifndef TYPES_H
#define TYPES_H

struct MinResult {
  int distance;
  int a_idx;
  int b_idx;
};

struct MinResultSingleIndex {
  int distance;
  int idx;
};

struct Pair {
  int distance;
  int ax;
  int ay;
  int bx;
  int by;
};

#endif