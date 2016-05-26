#pragma once

#include <sstream>
#include <string>

#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/adjacent_difference.h>

#include "cell.cu"
#include "tuple_utility.cu"
#include "utils.cpp"

namespace nearptd {
  
  // Convert to type T1 while clamping
  // Hard coded to unsigned short int
  template<typename T1, typename T2> __host__ __device__
  T1 clamp(T2 a) {
    const T2 mm(static_cast<T2>(USHRT_MAX));
    return static_cast<T1>(a > mm ? mm : (a > 0 ? static_cast<T1>(a) : 0));
  }

  // Return true if greater, takes in a tuple of ints
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

  // Return true if less
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

  // Return squared difference of inputs as double
  template<typename T>
  struct square_difference : public thrust::binary_function<T, T, double>
  {
    __host__ __device__
    double square(const double t) const {return t*t;}

    __host__ __device__
    double operator()(const T& a, const T& b) {
      return square(a - b);
    }
  };

  // Calculate distance from given tuple of points to initialized poitn
  template<typename Coord_T, size_t Dim>
  struct distance2_functor : public thrust::unary_function<typename ntuple<Coord_T, Dim>::tuple, double>
  {
    typedef typename ntuple<double, Dim>::tuple Double_Tuple;
    typedef typename ntuple<Coord_T, Dim>::tuple Coord_Tuple;
    
    square_difference<Coord_T> square_diff;
    tuple_binary_apply<Coord_Tuple, Coord_Tuple, Double_Tuple, square_difference<Coord_T>, Dim> make_dist;
    tuple_reduce<Double_Tuple, double, thrust::plus<double>, Dim> total;
    
    const Coord_Tuple q;
    
    __host__ __device__
    distance2_functor(const Coord_Tuple q) : q(q) { }
  
    __host__ __device__
    double operator()(const Coord_Tuple& a) {
      Double_Tuple dists = make_dist(a, q, square_diff);
      return total(dists, thrust::plus<double>());
    }
  };

  // Convert a coordinate point to a cell index point based on grid values
  template<typename Coord_T, typename Cell_Index_T>
  struct coord_to_cell_index : public thrust::binary_function<Coord_T, double, Cell_Index_T>
  {
    double r_cell;
    coord_to_cell_index() : r_cell(-1) {}
    coord_to_cell_index(double r_cell) : r_cell(r_cell) {}

    __host__ __device__
    Cell_Index_T operator()(const Coord_T& a, const double& d) const {
      return static_cast<Cell_Index_T>(static_cast<double>(a)*r_cell + d);
    }
  };

  // Functor for getting point in array
  template<typename Coord_Ptr, typename Coord_T>
  struct get_point : public thrust::binary_function<Coord_Ptr, int, Coord_T>
  {
    __host__ __device__
    Coord_T operator()(const Coord_Ptr& a, const int& i) {
      return a[i];
    }
  };

  // Shift a coordinate by distf, either low or high
  template<typename Coord_T>
  struct shift_coord : public thrust::unary_function<Coord_T, Coord_T>
  {
    double distf;
    bool lohi;
    
    __host__ __device__
    shift_coord(double distf, bool lohi) : distf(distf), lohi(lohi) {}

    __host__ __device__
    Coord_T operator()(const Coord_T& a)  {
      if (lohi) {
        return clamp<Coord_T>(static_cast<double>(a) - distf);
      }
      else {
        return clamp<Coord_T>(static_cast<double>(a) + distf + 1.0);
      }
    }
  };

  // Calculate scale for number of cells
  template<typename Coord_T>
  struct scale : public thrust::binary_function<Coord_T, Coord_T, double> {
    const int ng;
    scale(int ng) : ng(ng) {}

    __host__ __device__
    double operator()(const Coord_T& lo, const Coord_T& hi) {
      return 0.99 * ng / static_cast<double>(hi - lo);
    }
  };

  // Calculate dimension of a cell
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