#pragma once

#include <thrust/tuple.h>

namespace nearptd {
  // Calculate factorial at compile time
  template<size_t N>
  struct factorial {
    enum {value = N * factorial<N-1>::value};
  };

  template<>
  struct factorial<0> {
    enum {value = 1};
  };
  
  // Construct array of permutations of signs
  template<size_t N>
  struct sign {
    int s[1<<N][N];
    sign() {
      for (int i=0; i<(1<<N); ++i) {
        for (int j=0; j<N; ++j) {
          // Shift outer count by inner count and bitmask last bit
          // Then convert 0 and 1 to -1 and 1
          s[i][j] = ((i >> j) & (0x01)) ? -1 : 1;
        }
      } 
    }
    __host__ __device__
    int* operator[](const int& i) {
      return s[i];
    }
  };

  // Construct array of permutation of indices
  template<size_t N>
  struct perm {
    int p[factorial<N>::value][N];
    perm() {
      int id[N];
      int c = 1;
      int i = 1;
      // First sequence is in order
      for (int j=0; j<N; ++j) {
        p[0][j] = j;
        id[j] = 0;
      }
      // Iterative Heap's algorithm
      while (i < N) {
        if (id[i] < i) {
          // Swap elemnts from previous sequence
          int s = i % 2 * id[i];
          p[c][i] = p[c-1][s];
          p[c][s] = p[c-1][i];
          // Copy the rest
          for (int j=0; j<N; ++j) {
            if (j!=i && j!=s) p[c][j] = p[c-1][j];
          }
          id[i]++;
          i=1;
          c++;
        }
        else {
          id[i] = 0;
          ++i;
        }
      }
    }
    __host__ __device__
    int* operator[](const int& i) {
      return p[i];
    }
  };

  // Calculate the number of cells that would be created
  // for N dimensions and dmax number in each dimension
  int cell_count(int N, int dmax) {
    if (N == 1) return dmax;
    return ((dmax + N-1) * cell_count(N-1, dmax)) / N;
  }

  // Construct cell search order
  template<size_t Dim>
  struct cellsearchorder {
    // Convenience typedefs
    typedef typename Cell<Dim>::Cell_Index_T Cell_Index_T;
    typedef typename ntuple<Cell_Index_T, Dim>::tuple Cell_Index_Tuple;
    typedef typename Point_Vector<Cell_Index_T, Dim>::Coord_Tuple_Iterator Cell_Tuple_Iterator;
    typedef typename Point_Vector<Cell_Index_T, Dim>::Coord_Ptr Coord_Ptr;
    ntuple<Cell_Index_T, Dim> Cell_Index_Ntuple;
    
    Point_Vector<Cell_Index_T, Dim> *cells;
    thrust::device_vector<int> stop;
    size_t size;

    cellsearchorder() {
      int dmax = 1;
      // Find dmax to have around 1 million cells
      while (cell_count(Dim, dmax) < 1 << 20) {dmax++;}
      size = cell_count(Dim, dmax);
      Cell_Index_T coords[Dim];
      size_t index = Dim-1;
      for (size_t i=0; i<Dim; ++i) {coords[i] = 0;}

      thrust::host_vector<Cell_Index_T> arr(size*Dim);
      // Iterative nested for loop
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
      // Calculate and sort distances
      thrust::transform(cells->begin(), cells->end(), dists.begin(), dist2);
      thrust::sort_by_key(dists.begin(), dists.end(), cells->begin());
      stop = thrust::device_vector<int>(size);

      for (int i=0; i<Dim; ++i) {
        coords[i] = -2;
      }
      // Calculate distances, adding 2 to each dimension
      distance2_functor<Cell_Index_T, Dim> distplus2(Cell_Index_Ntuple.make_tuple(coords));
      typedef thrust::transform_iterator<distance2_functor<Cell_Index_T, Dim>, Cell_Tuple_Iterator> dist2_itr;
      dist2_itr begin(cells->begin(), distplus2);
      dist2_itr end(cells->end(), distplus2);

      // Find where in the dist+2 calculation the actual distances
      // would be placed, effectively calculates how far further
      // a search must continue
      thrust::upper_bound(dists.begin(), dists.end(),
                          begin, end, stop.begin());

      // Ensure monotonically increasing
      // Faster to copy to host and do sequentially than perform on GPU
      // Adjacent difference does not work for multiple repeated values
      thrust::host_vector<int> stoph(size);
      thrust::copy(stop.begin(), stop.end(), stoph.begin());
      int s = 0;
      for (int i=0; i<size; ++i) {
        if (stoph[i] > s) {
          s = stoph[i];
        }
        else {
          stoph[i] = s;
        }
      }
      // Copy back
      thrust::copy(stoph.begin(), stoph.end(), stop.begin());
    }
  };
};