#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/extrema.h>

#include "strided_range.cu"
#include "tuple_utility.cu"

namespace nearpt3 {

  template<typename Coord_T, size_t Dim>
  class Point_Vector {
  public:
    // Typedefs for internal use
    typedef thrust::device_vector<Coord_T> Coord_Vector;
    typedef typename Coord_Vector::iterator Coord_Iterator;
    typedef thrust::pair<Coord_Iterator, Coord_Iterator> Coord_Iterator_Pair;
    typedef typename ntuple<Coord_Iterator, Dim>::tuple Coord_Itr_Tuple;
    typedef thrust::device_ptr<Coord_T> Coord_Ptr;
    
    // Typedefs for external use
    typedef typename ntuple<Coord_T, Dim>::tuple Coord_Tuple;
    typedef thrust::zip_iterator<Coord_Itr_Tuple> Coord_Iterator_Tuple;
    typedef typename ntuple<Coord_Ptr, Dim>::tuple Coord_Ptr_Tuple;

    Point_Vector(const int npts, thrust::host_vector<Coord_T> pts)
      : npts(npts) {
      typedef typename thrust::host_vector<Coord_T>::iterator Host_Itr;
      for (size_t i=0; i<Dim; ++i) {
        vectors[i] = Coord_Vector(npts);
        begins[i] = vectors[i].begin();
        ends[i] = vectors[i].end();
        strided_range<Host_Itr> p(pts.begin()+i, pts.end(), Dim);
        thrust::copy(p.begin(), p.end(), vectors[i].begin());
      }
    }

    int get_size() {
      return npts;
    }
    
    // Taken from zip iterator example
    Coord_Iterator_Tuple begin() {
      return thrust::make_zip_iterator(Coord_Itr_Ntuple.make(begins));
    }

    Coord_Iterator_Tuple end() {
      return thrust::make_zip_iterator(Coord_Itr_Ntuple.make(ends));
    }

    // Return pair of minimum and maximum in each dimension
    // Not technically a coord tuple, as they are not a point in the original data, but makes data easier to send
    thrust::pair<Coord_Tuple, Coord_Tuple> minmax() {
      Coord_T los[Dim];
      Coord_T his[Dim];
      for (size_t i=0; i<Dim; ++i) {
        Coord_Iterator_Pair pair = thrust::minmax_element(begins[i], ends[i]);
        los[i] = *thrust::get<0>(pair);
        his[i] = *thrust::get<1>(pair);
      }
      Coord_Tuple lo(Coord_Ntuple.make(los));
      Coord_Tuple hi(Coord_Ntuple.make(his));
      return thrust::pair<Coord_Tuple, Coord_Tuple>(lo, hi);
    }

    // Get device pointers of vectors (for referencing on GPU)
    Coord_Ptr_Tuple get_ptrs() {
      for (size_t i=0; i<Dim; ++i) {
        ptrs[i] = vectors[i].data();
      }
      return Coord_Ptr_Ntuple.make(ptrs);
    }
    
    Coord_Tuple operator[] (int i) {
      for (size_t d=0; d<Dim; ++d) {
        c[d] = vectors[d][i];
      }
      return Coord_Ntuple.make(c);
    }

  private:
    int npts;
    Coord_Vector vectors[Dim];
    // Used making tuples
    Coord_T c[Dim];
    Coord_Ptr ptrs[Dim];
    Coord_Iterator begins[Dim];
    Coord_Iterator ends[Dim];

    ntuple<Coord_T, Dim> Coord_Ntuple;
    ntuple<Coord_Ptr, Dim> Coord_Ptr_Ntuple;
    ntuple<Coord_Iterator, Dim> Coord_Itr_Ntuple;
  };
};