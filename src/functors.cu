#pragma once

#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/adjacent_difference.h>

#include "cell.cu"
#include "tuple_utility.cu"
#include "utils.cpp"

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
  
  template<typename T>
  struct add : public thrust::unary_function<T, T>
  {
    const T b;

    add(const T b) : b(b) {}

    __host__ __device__
    T operator()(const T& a) const {
      return a + b;
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
    
    const Coord_Tuple q;
    square_difference<Coord_T> square_diff;
    tuple_binary_apply<Coord_Tuple, Coord_Tuple, Double_Tuple, square_difference<Coord_T>, Dim> make_dist;
    tuple_reduce<Double_Tuple, double, thrust::plus<double>, Dim> total;
    
    __host__ __device__
    distance2_functor(const Coord_Tuple q) : q(q) { }
  
    __host__ __device__
    double operator()(const Coord_Tuple& a) {
      Double_Tuple dists = make_dist(a, q, square_diff);
      return total(dists, thrust::plus<double>());
    }
  };

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

  // Construct array of permutations of signs
  template<size_t N>
  struct sign {
    int s[1<<N][N];
    sign() {
      for (int i=0; i<(1<<N); ++i) {
        for (int j=0; j<N; ++j) {
          s[i][j] = ((i >> j) & (0x01)) ? -1 : 1;
        }
      } 
    }
    __host__ __device__
    int* operator[](const int& i) {
      return s[i];
    }
  };

  // Calculate factorial at compile time
  template<size_t N>
  struct factorial {
    enum {value = N * factorial<N-1>::value};
  };

  template<>
  struct factorial<0> {
    enum {value = 1};
  };

  // Construct array of permutation of indices
  template<size_t N>
  struct perm {
    int p[factorial<N>::value][N];
    perm() {
      for (int i=0; i<factorial<N>::value; ++i) {
        for (int j=0; j<N; ++j) {
          if (i == 0) {
            p[i][j] = j;
          }
          else {
            p[i][j] = p[i-1][j];
          }
        }
        if (i > 0) {
          int t = p[i][i % N];
          p[i][i % N] = p[i][(i-1) % N];
          p[i][(i-1) % N] = t;
        }
      }
    }
    __host__ __device__
    int* operator[](const int& i) {
      return p[i];
    }
  };

  int cell_count(int N, int dmax) {
    if (N == 1) return dmax;
    return ((dmax + N-1) * cell_count(N-1, dmax)) / N;
  }


  template<size_t Dim>
  struct cellsearchcreate {
    typedef typename Cell<Dim>::Cell_Index_T Cell_Index_T;
    typedef typename ntuple<Cell_Index_T, Dim>::tuple Cell_Index_Tuple;
    typedef typename Point_Vector<Cell_Index_T, Dim>::Coord_Tuple_Iterator Cell_Tuple_Iterator;
    typedef typename Point_Vector<Cell_Index_T, Dim>::Coord_Ptr Coord_Ptr;
    ntuple<Cell_Index_T, Dim> Cell_Index_Ntuple;  
    Point_Vector<Cell_Index_T, Dim> *cells;
    thrust::device_vector<int> stop;
    size_t size;

    cellsearchcreate() {
      int dmax = 1;
      while (cell_count(Dim, dmax) < 1 << 20) {dmax++;}
      size = cell_count(Dim, dmax);
      Cell_Index_T coords[Dim];
      size_t index = Dim-1;
      for (size_t i=0; i<Dim; ++i) {coords[i] = 0;}

      thrust::host_vector<Cell_Index_T> arr(size*Dim);
      for (int i=0; i<size; ++i) {
        for (size_t j=0; j<Dim; ++j) {
          arr[i*Dim+j] = coords[j];
        }
        coords[Dim-1]++;

        while (coords[index] >= dmax) {
          if (index != 0) {
            index--;
            coords[index]++;
            int t=index;
            while (t < Dim-1) {
              coords[t+1] = coords[t];
              t++;
            }
          }
          else {
            break;
          }
        }
        index = Dim-1;
      }
      cells = new Point_Vector<Cell_Index_T, Dim>(size, arr);

      // Cells[0] is all zeros, all distances are calculated relative to zero
      distance2_functor<Cell_Index_T, Dim> dist2((*cells)[0]);
      thrust::device_vector<double> dists(size, 0);
      thrust::transform(cells->begin(), cells->end(), dists.begin(), dist2);
      thrust::sort_by_key(dists.begin(), dists.end(), cells->begin());
      stop = thrust::device_vector<int>(size);

      //add<Cell_Index_T> add2(2);
      //tuple_unary_apply<Cell_Index_Tuple, Cell_Index_Tuple, add<Cell_Index_T>, Dim> add_2;

      for (int i=0; i<Dim; ++i) {
        coords[i] = -2;
      }
      
      distance2_functor<Cell_Index_T, Dim> distplus2(Cell_Index_Ntuple.make_tuple(coords));
      typedef thrust::transform_iterator<distance2_functor<Cell_Index_T, Dim>, Cell_Tuple_Iterator> dist2_itr;
      dist2_itr begin(cells->begin(), distplus2);
      dist2_itr end(cells->end(), distplus2);
      
      thrust::upper_bound(dists.begin(), dists.end(),
                          begin, end, stop.begin());
      // thrust::adjacent_difference(stop.begin(), stop.end(),
      //                             stop.begin(), thrust::maximum<int>());
      
      // int istop = 0;
      // for (int i=0; i<size; ++i) {
      //   const double d2 = dist2(add_2((*cells)[i], add2));
      //   for (; istop<size; ++istop) {
      //     if (dists[istop] > d2) break;
      //   }
      //   stop[i] = istop;
      // }
    }
  };
};