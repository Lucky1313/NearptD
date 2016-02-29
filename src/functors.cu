#include <thrust/functional.h>

namespace nearpt3 {

  template<typename T>
  struct point_to_id_functor : public thrust::unary_function<T, int>
  {
    const int ng;
    const double r_cell;
    const double d1;
    const double d2;
    const double d3;

    point_to_id_functor(int ng, double r_cell, double d1, double d2, double d3)
      : ng(ng), r_cell(r_cell), d1(d1), d2(d2), d3(d3) { }
  
    __host__ __device__
    int operator()(const T& a) const {
      int ix = static_cast<short int>(static_cast<double>(thrust::get<0>(a))*r_cell+d1);
      int iy = static_cast<short int>(static_cast<double>(thrust::get<1>(a))*r_cell+d2);
      int iz = static_cast<short int>(static_cast<double>(thrust::get<2>(a))*r_cell+d3);

      if (ix < 0 || ix >= ng || iy < 0 || iy >= ng || iz < 0 || iz >= ng) return -1;
      return (ix*ng + iy)*ng + iz;
    }
  };

  template<typename T>
  struct distance2_functor : public thrust::unary_function<T, double>
  {
    const int x;
    const int y;
    const int z;

    distance2_functor(int x, int y, int z) : x(x), y(y), z(z) { }

    __host__ __device__
    double square(const double t) const {
      return t * t;
    }
  
    __host__ __device__
    double operator()(const T& a) const {
      return (square((thrust::get<0>(a)-x)) +
              square((thrust::get<1>(a)-y)) +
              square((thrust::get<2>(a)-z)));
    }
  };

};