#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <boost/multi_array.hpp>

#include "strided_range.cu"

using boost::array;

namespace nearpt3 {

  template<typename Coord_T>
  class Points_Vector {
  public:
    // Convenience Typedefs
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord_Tuple;
    typedef thrust::device_vector<Coord_T> Coord_Vector;
    typedef typename Coord_Vector::iterator Coord_Iterator;
    typedef thrust::pair<Coord_Iterator, Coord_Iterator> Coord_Iterator_Pair;
    typedef thrust::zip_iterator<thrust::tuple<Coord_Iterator, Coord_Iterator, Coord_Iterator> > Coord_Iterator_Tuple;

    Points_Vector(const int npts, thrust::host_vector<Coord_T> pts)
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

    thrust::pair<array<Coord_T,3>, array<Coord_T,3> > minmax() {
      Coord_Iterator_Pair xpair = thrust::minmax_element(px.begin(), px.end());
      Coord_Iterator_Pair ypair = thrust::minmax_element(py.begin(), py.end());
      Coord_Iterator_Pair zpair = thrust::minmax_element(pz.begin(), pz.end());
      array<Coord_T,3> lo = {*thrust::get<0>(xpair), *thrust::get<0>(ypair), *thrust::get<0>(zpair)};
      array<Coord_T,3> hi = {*thrust::get<1>(xpair), *thrust::get<1>(ypair), *thrust::get<1>(zpair)};
      return thrust::pair<array<Coord_T,3>, array<Coord_T,3> >(lo, hi);
    }    

  private:
    int npts;
    Coord_Vector px;
    Coord_Vector py;
    Coord_Vector pz;
  };

};