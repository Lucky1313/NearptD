#pragma once

#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "cell.cu"
#include "tuple_utility.cu"

namespace nearpt3 {
  
  // clamp_USI     Convert to an unsigned short int while clamping
  template<typename T> __host__ __device__
  unsigned short int clamp_USI(T a) {
    const T mm(static_cast<T>(USHRT_MAX));
    return static_cast<unsigned short int>(a > mm ? mm : (a > 0 ? static_cast<unsigned short int>(a) : 0));
  }

  template<typename T>
  struct greater_functor : public thrust::unary_function<T, bool>
  {
    const int b;

    greater_functor(int b) : b(b) {}

    __host__ __device__
    bool operator()(const T& a) const {
      return thrust::get<0>(a) > b;
    }
  };

  template<typename T>
  struct less_functor: public thrust::unary_function<T, bool>
  {
    T b;

    less_functor(T b) : b(b) {}

    __host__ __device__
    bool operator()(const T& a) const {
      return a < b;
    }
  };
  
  template<typename Coord_T>
  struct square_difference : public thrust::binary_function<Coord_T, Coord_T, double>
  {
    __host__ __device__
    double square(const double t) const {return t*t;}

    __host__ __device__
    double operator()(const Coord_T& a, const Coord_T& b) {
      return square(a - b);
    }
  };
  
  template<typename Coord_T, size_t Dim>
  struct distance2_functor : public thrust::unary_function<typename ntuple<Coord_T, Dim>::tuple, double>
  {
    typedef typename ntuple<double, Dim>::tuple Double_Tuple;
    typedef typename ntuple<Coord_T, Dim>::tuple Coord_Tuple;
    
    const Coord_Tuple& q;
    square_difference<Coord_T> square_diff;
    tuple_binary_apply<Coord_Tuple, Coord_Tuple, Double_Tuple, square_difference<Coord_T>, Dim> make_dist;
    tuple_reduce<Double_Tuple, double, thrust::plus<double>, Dim> total;
    
    __host__ __device__
    distance2_functor(const Coord_Tuple& q) : q(q) { }
  
    __host__ __device__
    double operator()(const Coord_Tuple& a) {
      Double_Tuple dists = make_dist(a, q, square_diff);
      return total(dists, thrust::plus<double>());
    }
  };

  template<typename Coord_T>
  struct coord_to_short : public thrust::binary_function<Coord_T, double, short int>
  {
    double r_cell;
    coord_to_short() : r_cell(-1) {}
    coord_to_short(double r_cell) : r_cell(r_cell) {}

    __host__ __device__
    short int operator()(const Coord_T& a, const double& b) {
      return static_cast<short int>(static_cast<double>(a)*r_cell + b);
    }
  };

  template<typename Coord_Ptr, typename Coord_T>
  struct get_point : public thrust::binary_function<Coord_Ptr, int, Coord_T>
  {
    __host__ __device__
    Coord_T operator()(const Coord_Ptr& a, const int& i) {
      return a[i];
    }
  };

  template<typename Coord_T>
  struct near_cell : public thrust::unary_function<Coord_T, Coord_T>
  {
    double distf;
    bool lohi;
    __host__ __device__
    near_cell(double distf, bool lohi) : distf(distf), lohi(lohi) {}

    __host__ __device__
    Coord_T operator()(const Coord_T& a)  {
      if (lohi) {
        return clamp_USI(static_cast<double>(a) - distf);
      }
      else {
        return clamp_USI(static_cast<double>(a) + distf + 1.0);
      }
    }
  };

  template<typename Coord_T>
  struct scale : public thrust::binary_function<Coord_T, Coord_T, double> {
    const int ng;
    scale(int ng) : ng(ng) {}

    __host__ __device__
    double operator()(const Coord_T& lo, const Coord_T& hi) {
      return 0.99 * ng / static_cast<double>(hi - lo);
    }
  };

  template<typename Coord_T>
  struct cell_dim : public thrust::binary_function<Coord_T, Coord_T, double> {
    const int ng;
    const double r_cell;
    cell_dim(int ng, double r_cell) : ng(ng), r_cell(r_cell) {}

    __host__ __device__
    double operator()(const Coord_T& lo, const Coord_T& hi) {
      return 0.5 * ((ng - 1) - r_cell * (lo + hi));
    }
  };
};