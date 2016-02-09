#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/copy.h>

#include <algorithm>
#include <boost/multi_array.hpp> 
#include <iomanip>
#include <iostream>
#include <fstream>
#include <math.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/times.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

using namespace std;
using boost::array;

#define DEBUG

// Print an expression's name then its value, possibly followed by a comma or endl.  
// Ex: cout << PRINTC(x) << PRINTN(y);

#define PRINT(arg)  #arg "=" << (arg)
#define PRINTC(arg)  #arg "=" << (arg) << ", "
#define PRINTN(arg)  #arg "=" << (arg) << endl

// From thrust example code
template <typename Iterator>
class strided_range
{
    public:

    typedef typename thrust::iterator_difference<Iterator>::type difference_type;

    struct stride_functor : public thrust::unary_function<difference_type,difference_type>
    {
        difference_type stride;

        stride_functor(difference_type stride)
            : stride(stride) {}

        __host__ __device__
        difference_type operator()(const difference_type& i) const
        { 
            return stride * i;
        }
    };

    typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
    typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
    typedef typename thrust::permutation_iterator<Iterator,TransformIterator>     PermutationIterator;

    // type of the strided_range iterator
    typedef PermutationIterator iterator;

    // construct strided_range for the range [first,last)
    strided_range(Iterator first, Iterator last, difference_type stride)
        : first(first), last(last), stride(stride) {}
   
    iterator begin(void) const
    {
        return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
    }

    iterator end(void) const
    {
        return begin() + ((last - first) + (stride - 1)) / stride;
    }
    
    protected:
    Iterator first;
    Iterator last;
    difference_type stride;
};

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
  int operator()(const T& a) const{
    int ix = static_cast<short int>(static_cast<double>(thrust::get<0>(a))*r_cell+d1);
    int iy = static_cast<short int>(static_cast<double>(thrust::get<1>(a))*r_cell+d2);
    int iz = static_cast<short int>(static_cast<double>(thrust::get<2>(a))*r_cell+d3);

    if (ix < 0 || ix >= ng || iy < 0 || iy >= ng || iz < 0 || iz >= ng) return -1;
    return (ix*ng + iy)*ng + iz;
  }
};

template <typename T>
void write(ostream &o, const thrust::tuple<T, T, T>& c) {
  o << "(" << thrust::get<0>(c) << "," << thrust::get<1>(c) << "," << thrust::get<2>(c) << ")";
}

template <typename T>
void write(ostream &o, const array<T,3>& c) {
    o << "(" << c[0] << "," << c[1] << "," << c[2] << ")";
}


namespace nearpt3 {
	double ng_factor = 1.6;

  // cellsearchorder:
  // First 3 elements of each row:  the order in which to search (one 48th-ant of the) cells adjacent to the current cell.
  // 4th element:   where, in cellsearchorder, to stop searching after the first point is found.
  const static int  cellsearchorder[][4] = {
#include "cellsearchorder"
  };
  // Number of cells in cellsearchorder (before expanding symmetries).
  const static int ncellsearchorder = 
    sizeof(nearpt3::cellsearchorder) / sizeof(nearpt3::cellsearchorder[0][0])/4;

  typedef short int Cell3_Index_T;
  class Cell3 {
  public:
    array<Cell3_Index_T,3> c;

    Cell3(const Cell3_Index_T x, const Cell3_Index_T y, const Cell3_Index_T z) {
      c[0] = x; c[1] = y; c[2] = z; }

    Cell3(const Cell3 &a) { c[0] = a[0]; c[1] = a[1]; c[2] = a[2]; }

    Cell3() { c[0] = -1; c[1] = -1; c[2] = -1; }

    Cell3_Index_T & operator[] (const int i)  {  return c[i];  }

    const Cell3_Index_T & operator[] (const int i) const {  return c[i];  }

    const Cell3 operator+(const Cell3 &d) const {
      Cell3 r;
      r[0] = c[0]+d[0];
      r[1] = c[1]+d[1];
      r[2] = c[2]+d[2];
      return r;
    }

    const Cell3 operator*(const int *d) const {
      Cell3 r;
      r[0] = c[0]*d[0];
      r[1] = c[1]*d[1];
      r[2] = c[2]*d[2];
      return r;
    }

    bool operator==(const Cell3 &d)  const {
      return c[0]==d[0] && c[1]==d[1] && c[2]==d[2];
    }

    //    const Cell3 operator*(const int *) const;
  };

  void write(ostream &o, const Cell3& c) {
    o << '(' << c[0] << ',' << c[1] << ',' << c[2] << ") ";
  }

  template<typename Coord_T>
  class Points_T {
    // Convenience Typedefs
    typedef thrust::device_vector<Coord_T> Coord_Vector;
    typedef typename Coord_Vector::iterator Coord_Iterator;
    typedef thrust::tuple<Coord_Iterator, Coord_Iterator, Coord_Iterator> Coord_Iterator_Tuple;
    typedef thrust::zip_iterator<Coord_Iterator_Tuple> Coord_3_Iterator;
    typedef thrust::pair<Coord_Iterator, Coord_Iterator> Coord_Iterator_Pair;
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord3;
  public:

    Points_T(const int npts, thrust::host_vector<Coord_T> pts)
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

    // Take from zip iterator example
    Coord_3_Iterator begin() {
      return thrust::make_zip_iterator(make_tuple(px.begin(), py.begin(), pz.begin()));
    }

    Coord_3_Iterator end() {
      return thrust::make_zip_iterator(make_tuple(px.end(), py.end(), pz.end()));
    }

    Coord_Iterator_Pair x_minmax() {
      return thrust::minmax_element(px.begin(), px.end());
    }
    
    Coord_Iterator_Pair y_minmax() {
      return thrust::minmax_element(py.begin(), py.end());
    }
    
    Coord_Iterator_Pair z_minmax() {
      return thrust::minmax_element(pz.begin(), pz.end());
    }

    thrust::pair<array<Coord_T,3>, array<Coord_T,3> > minmax() {
      Coord_Iterator_Pair xpair = x_minmax();
      Coord_Iterator_Pair ypair = y_minmax();
      Coord_Iterator_Pair zpair = z_minmax();
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

  

  template<typename Coord_T>
  class Grid_T {
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord3;

    typedef thrust::device_vector<Coord_T> Coord_Vector;
    typedef typename Coord_Vector::iterator Coord_Iterator;
    typedef thrust::tuple<Coord_Iterator, Coord_Iterator, Coord_Iterator> Coord_Iterator_Tuple;
    typedef thrust::zip_iterator<Coord_Iterator_Tuple> Coord_3_Iterator;
    typedef thrust::pair<Coord_Iterator, Coord_Iterator> Coord_Iterator_Pair;
  public:
    int ng;
    int ng3;
    double r_cell;
    array<double,3> d_cell;
    int nfixpts;
    Points_T<Coord_T>* pts;
    thrust::device_vector<int> cells;
    thrust::device_vector<int> base;

    int point_to_id(const int& n) {
      Coord_3_Iterator p = pts->begin();
      int ix = static_cast<short int>(static_cast<double>(thrust::get<0>(p[n]))*r_cell+d_cell[0]);
      int iy = static_cast<short int>(static_cast<double>(thrust::get<1>(p[n]))*r_cell+d_cell[1]);
      int iz = static_cast<short int>(static_cast<double>(thrust::get<2>(p[n]))*r_cell+d_cell[2]);

      if (ix < 0 || ix >= ng || iy < 0 || iy >= ng || iz < 0 || iz >= ng) return -1;
      return (ix*ng + iy)*ng + iz;
    }
    
  };
  

  template<typename Coord_T> Grid_T<Coord_T>*
  Preprocess(const int nfixpts, Points_T<Coord_T>* pts) {
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord3;
    
    Grid_T<Coord_T> *g;
    g = new Grid_T<Coord_T>;
    g->nfixpts = nfixpts;
    int &ng = g->ng;
    ng = static_cast<int> (ng_factor * cbrt(static_cast<double>(nfixpts)));

    ng = min(2000, max(1, ng));
    g->ng3 = ng * ng * ng;
    g->pts = pts;

    for (int i=1; i<ncellsearchorder; ++i)
      if (nearpt3::cellsearchorder[i-1][3] > nearpt3::cellsearchorder[i][3]) 
	throw "cellsearchorder is not monotonic";

    thrust::pair<array<Coord_T,3>, array<Coord_T,3> > minmax = pts->minmax();
    array<Coord_T,3> lo = thrust::get<0>(minmax);
    array<Coord_T,3> hi = thrust::get<1>(minmax);

    #ifdef DEBUG
    cout << "Min/Max" << endl;
    cout << lo[0] << ", " << lo[1] << ", " << lo[2] << endl;
    cout << hi[0] << ", " << hi[1] << ", " << hi[2] << endl;
    #endif

    array<double,3> s;
    array<double,3> d;
    
    for (int i=0; i<3; ++i) {
      s[i] = 0.99 * ng / static_cast<double>(hi[i] - lo[i]);
    }
    g->r_cell = min(min(s[0], s[1]), s[2]);

    for(int i=0; i<3; i++) {
      g->d_cell[i] = ((ng-1)-(lo[i]+hi[i])*g->r_cell) * 0.5;
    }

    #ifdef DEBUG
    cout << "Grid info:";
    cout << "\nng: " << g->ng;
    cout << "\nng3: " << g->ng3;
    cout << "\ns: (" << s[0] << ", " << s[1] << ", " << s[2] << ")";
    cout << "\nr_cell: " << g->r_cell;
    cout << "\nd_cell: " << g->d_cell[0] << ", " << g->d_cell[1] << ", " << g->d_cell[2] << ")";
    cout << endl;
    #endif

    g->base = thrust::device_vector<int>(g->ng3+1, 1);
    g->cells = thrust::device_vector<int>(g->nfixpts);

    double aaa[3];
    aaa[0] = g->d_cell[0];
    aaa[1] = g->d_cell[1];
    aaa[2] = g->d_cell[2];
    // Calculate cell id from point
    thrust::transform(pts->begin(), pts->end(), g->cells.begin(),
                      point_to_id_functor<Coord3>(g->ng, g->r_cell, aaa[0], aaa[1], aaa[2]));

    #ifdef DEBUG
    cout << "Cell IDs (cells): [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    // Ensure no cells are -1 (outside range)
    if (thrust::find(g->cells.begin(), g->cells.end(), -1) != g->cells.end()) {
      throw "Bad cell";
    }

    thrust::sort(g->cells.begin(), g->cells.end());

    #ifdef DEBUG
    cout << "Sorted Cell IDs (cells): [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    // Taken from thrust histogram example
    thrust::counting_iterator<int> search(0);
    thrust::lower_bound(g->cells.begin(), g->cells.end(),
                        search, search + g->ng3 + 1,
                        g->base.begin());

    #ifdef DEBUG
    cout << "Count: " << *search << endl;
    cout << "Lower bound (base): [";
    thrust::copy(g->base.begin(), g->base.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    /*
    if (*g->base.end() != nfixpts) {
      cout << "Error, internal inconsistency; wrong " << PRINTN(g->base[g->ng3]);
      throw "Internal Inconsistency";
    }
    */

    /*
    thrust::adjacent_difference(g->cells.begin(), g->cells.end(), g->cells.begin());

    #ifdef DEBUG
    cout << "Adjacent difference (cells): [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    */
    
    thrust::fill(g->cells.begin(), g->cells.end(), 0);
    
    for (int n=0; n<g->nfixpts; ++n) {
      const int ic(g->point_to_id(n));
      const int pitc = g->cells[g->base[ic+1]-1]++;
      g->cells[g->base[ic]+pitc] = n;
    }

    /*
    #ifdef DEBUG
    cout << "Count (cells): [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    typedef thrust::device_vector<int>::iterator intiterator;
    //thrust::transform_iterator<ptif, intiterator> itr(pts->begin(), point_to_id_functor<Coord3>(g->ng, g->r_cell, aaa[0], aaa[1], aaa[2]));

    #ifdef DEBUG
    cout << "Transform: [";
    thrust::copy(thrust::make_transform_iterator(pts->begin(),
                                                      point_to_id_functor<Coord3>(g->ng, g->r_cell, g->d_cell[0], g->d_cell[1], g->d_cell[2])),
                 thrust::make_transform_iterator(pts->end(),
                                                 point_to_id_functor<Coord3>(g->ng, g->r_cell, g->d_cell[0], g->d_cell[1], g->d_cell[2])),
                 ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    thrust::transform(g->cells.begin(), g->cells.end(),
                      thrust::make_transform_iterator(pts->begin(),
                                                      point_to_id_functor<Coord3>(g->ng, g->r_cell, g->d_cell[0], g->d_cell[1], g->d_cell[2])),
                      g->cells.begin(),
                      cell_to_point_functor(thrust::raw_pointer_cast(g->base.data())));

    #ifdef DEBUG
    cout << "(cells): [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    */

    
    #ifdef DEBUG
    cout << "Iterative (cells): [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    return g;
  }

  template<typename Coord_T> int
  Query(Grid_T<Coord_T>* g, const array<Coord_T, 3> q) {
    
    
  }
  
};
