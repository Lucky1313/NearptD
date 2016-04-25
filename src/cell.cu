#pragma once

#include <iostream>

#include "tuple_utility.cu"

namespace nearpt3 {

  template<size_t Dim>
  class Cell {
  public:
    typedef short int Cell_Index_T;
    typedef typename ntuple<Cell_Index_T, Dim>::tuple Cell_Tuple;
    ntuple<Cell_Index_T, Dim> Cell_Ntuple;
    
    Cell_Index_T c[Dim];

    __host__ __device__
    Cell(const Cell_Index_T* a) {
      for (size_t i=0; i<Dim; ++i) {
        c[i] = a[i];
      }
    }

    __host__ __device__
    Cell(const Cell_Tuple& a) {
      Cell_Ntuple.make_array(a, c);
    }

    // __host__ __device__
    // Cell(const Cell_Index_T x, const Cell_Index_T y, const Cell_Index_T z) {
    //   c[0] = x;
    //   c[1] = y;
    //   c[2] = z;
    // }

    __host__ __device__
    Cell(const Cell<Dim> &a) {
      for (size_t i=0; i<Dim; ++i) {
        c[i] = a[i];
      }
    }

    __host__ __device__
    Cell() {
      for (size_t i=0; i<Dim; ++i) {
        c[i] = -1;
      }
    }

    __host__ __device__
    Cell_Index_T & operator[] (const int i)  {  return c[i];  }

    __host__ __device__
    const Cell_Index_T & operator[] (const int i) const {  return c[i];  }

    __host__ __device__
    const Cell<Dim> operator+(const Cell<Dim> &d) const {
      Cell<Dim> r;
      for (size_t i=0; i<Dim; ++i) {
        r[i] = c[i] + d[i];
      }
      return r;
    }

    __host__ __device__
    const Cell<Dim> operator*(const int *d) const {
      Cell<Dim> r;
      for (size_t i=0; i<Dim; ++i) {
        r[i] = c[i] * d[i];
      }
      return r;
    }

    __host__ __device__
    bool operator==(const Cell<Dim> &d)  const {
      for (size_t i=0; i<Dim; ++i) {
        if (c[i] != d[i]) return false;
      }
      return true;
    }
  };

  template<size_t Dim>
  void write(std::ostream &o, const Cell<Dim>& c) {
    o << '(';
    for (size_t i=0; i<Dim-1; ++i) {
      o << c[i] << ',';
    }
    o << c[Dim-1] << ") ";
  }
};
