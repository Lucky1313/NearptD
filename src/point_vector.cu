#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/extrema.h>

#include "strided_range.cu"

namespace nearpt3 {

  template<typename Coord_T>
  class Point_Vector {
  public:
    // Convenience Typedefs
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord_Tuple;
    typedef thrust::device_vector<Coord_T> Coord_Vector;
    typedef typename Coord_Vector::iterator Coord_Iterator;
    typedef thrust::pair<Coord_Iterator, Coord_Iterator> Coord_Iterator_Pair;
    typedef thrust::zip_iterator<thrust::tuple<Coord_Iterator, Coord_Iterator, Coord_Iterator> > Coord_Iterator_Tuple;

    typedef thrust::device_ptr<Coord_T> Coord_Ptr;
    typedef thrust::tuple<Coord_Ptr, Coord_Ptr, Coord_Ptr> Coord_Ptr_Tuple;

    Point_Vector(const int npts, thrust::host_vector<Coord_T> pts)
      : npts(npts) {
      // Create device vectors
      px = Coord_Vector(npts);
      py = Coord_Vector(npts);
      pz = Coord_Vector(npts);

      // Stride host vector for x, y, z
      typedef typename thrust::host_vector<Coord_T>::iterator Host_Itr;
      strided_range<Host_Itr> x(pts.begin(), pts.end(), 3);
      strided_range<Host_Itr> y(pts.begin()+1, pts.end(), 3);
      strided_range<Host_Itr> z(pts.begin()+2, pts.end(), 3);

      // Copy to device
      thrust::copy(x.begin(), x.end(), px.begin());
      thrust::copy(y.begin(), y.end(), py.begin());
      thrust::copy(z.begin(), z.end(), pz.begin());
    }

    // Taken from zip iterator example
    Coord_Iterator_Tuple begin() {
      return thrust::make_zip_iterator(make_tuple(px.begin(), py.begin(), pz.begin()));
    }

    Coord_Iterator_Tuple end() {
      return thrust::make_zip_iterator(make_tuple(px.end(), py.end(), pz.end()));
    }
    
    int get_size() {
      return npts;
    }

    // Return pair of minimum and maximum in each dimension
    // Not technically a coord tuple, as they are not a point in the original data, but makes data easier to send
    thrust::pair<Coord_Tuple, Coord_Tuple> minmax() {
      Coord_Iterator_Pair xpair = thrust::minmax_element(px.begin(), px.end());
      Coord_Iterator_Pair ypair = thrust::minmax_element(py.begin(), py.end());
      Coord_Iterator_Pair zpair = thrust::minmax_element(pz.begin(), pz.end());
      Coord_Tuple lo(thrust::make_tuple(*thrust::get<0>(xpair),
                                        *thrust::get<0>(ypair),
                                        *thrust::get<0>(zpair)));
      Coord_Tuple hi(thrust::make_tuple(*thrust::get<1>(xpair),
                                        *thrust::get<1>(ypair),
                                        *thrust::get<1>(zpair)));
      return thrust::pair<Coord_Tuple, Coord_Tuple>(lo, hi);
    }

    // Get device pointers of vectors (for referencing on GPU)
    Coord_Ptr_Tuple get_ptrs() {
      return thrust::make_tuple(px.data(), py.data(), pz.data());
    }
    
    Coord_Tuple operator[] (int i) {
      return thrust::make_tuple(px[i], py[i], pz[i]);
    }

  private:
    int npts;
    Coord_Vector px;
    Coord_Vector py;
    Coord_Vector pz;
  };
};