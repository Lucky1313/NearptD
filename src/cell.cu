#pragma once

#include <iostream>

namespace nearpt3 {

  typedef short int Cell_Index_T;
  class Cell3 {
  public:
    Cell_Index_T c[3];

    __host__ __device__
    Cell3(const Cell_Index_T x, const Cell_Index_T y, const Cell_Index_T z) {
      c[0] = x; c[1] = y; c[2] = z; }

    __host__ __device__
    Cell3(const Cell3 &a) { c[0] = a[0]; c[1] = a[1]; c[2] = a[2]; }

    __host__ __device__
    Cell3() { c[0] = -1; c[1] = -1; c[2] = -1; }

    __host__ __device__
    Cell_Index_T & operator[] (const int i)  {  return c[i];  }

    __host__ __device__
    const Cell_Index_T & operator[] (const int i) const {  return c[i];  }

    __host__ __device__
    const Cell3 operator+(const Cell3 &d) const {
      Cell3 r;
      r[0] = c[0]+d[0];
      r[1] = c[1]+d[1];
      r[2] = c[2]+d[2];
      return r;
    }

    __host__ __device__
    const Cell3 operator*(const int *d) const {
      Cell3 r;
      r[0] = c[0]*d[0];
      r[1] = c[1]*d[1];
      r[2] = c[2]*d[2];
      return r;
    }

    __host__ __device__
    bool operator==(const Cell3 &d)  const {
      return c[0]==d[0] && c[1]==d[1] && c[2]==d[2];
    }

    //    const Cell3 operator*(const int *) const;
  };

  void write(std::ostream &o, const Cell3& c) {
    o << '(' << c[0] << ',' << c[1] << ',' << c[2] << ") ";
  }
};
