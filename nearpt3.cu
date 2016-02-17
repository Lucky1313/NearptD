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

// clamp_USI     Convert to an unsigned short int while clamping

template <typename T>
unsigned short int clamp_USI(T a) {
  const unsigned short int m(numeric_limits<unsigned short int>::max());
  const T mm(static_cast<T>(m));
  return  
    static_cast<unsigned short int>(a > mm ? mm : (a > 0 ? static_cast<unsigned short int>(a) : 0));
}

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

  public:
    int ng;
    int ng3;
    double r_cell;
    array<double,3> d_cell;
    int nfixpts;
    Points_T<Coord_T>* pts;
    thrust::device_vector<int> cells;
    thrust::device_vector<int> base;

    // Check if this is a legal cell.
    bool check(const Cell3 a) const {
      if (a[0] < 0 || a[0] >= ng ) return false;
      if (a[1] < 0 || a[1] >= ng ) return false;
      if (a[2] < 0 || a[2] >= ng ) return false;
      return true;
    }

    // clip is needed because roundoff errors may cause a number to be slightly outside the legal range. 
    void clip(Cell3 &a) {

      //    a[0] = min(ng-1, max(0, a[0]));

      if (a[0] < 0) a[0] = 0;
      if (a[0] >= ng) a[0] = ng-1;
      if (a[1] < 0) a[1] = 0;
      if (a[1] >= ng) a[1] = ng-1;
      if (a[2] < 0) a[2] = 0;
      if (a[2] >= ng) a[2] = ng-1;
    }
    
    // Compute_Cell_Containing_Point: Return the cell number containing point p.  Note the p is a
    // array<Coord_T, 3>, not an id of a point.  This is necessary since
    // Compute_Cell_Containing_Point is used for both fixed and query points.

    const Cell3 Compute_Cell_Containing_Point(const array<Coord_T,3> p) {
      const short int ix = static_cast<short int>(static_cast<double>(p[0])*r_cell+d_cell[0]);   // This must truncate not round.
      const short int iy = static_cast<short int>(static_cast<double>(p[1])*r_cell+d_cell[1]);
      const short int iz = static_cast<short int>(static_cast<double>(p[2])*r_cell+d_cell[2]);
      Cell3 c(ix, iy, iz);
      return c;
    }
    
    int point_to_id(const int& n) {
      Coord_3_Iterator p = pts->begin();
      int ix = static_cast<short int>(static_cast<double>(thrust::get<0>(p[n]))*r_cell+d_cell[0]);
      int iy = static_cast<short int>(static_cast<double>(thrust::get<1>(p[n]))*r_cell+d_cell[1]);
      int iz = static_cast<short int>(static_cast<double>(thrust::get<2>(p[n]))*r_cell+d_cell[2]);

      if (ix < 0 || ix >= ng || iy < 0 || iy >= ng || iz < 0 || iz >= ng) return -1;
      return (ix*ng + iy)*ng + iz;
    }

    int qpoint_to_id(const array<Coord_T, 3> q) {
      int ix = static_cast<short int>(static_cast<double>(q[0])*r_cell+d_cell[0]);
      int iy = static_cast<short int>(static_cast<double>(q[1])*r_cell+d_cell[1]);
      int iz = static_cast<short int>(static_cast<double>(q[2])*r_cell+d_cell[2]);

      if (ix < 0 || ix >= ng || iy < 0 || iy >= ng || iz < 0 || iz >= ng) return -1;
      return (ix*ng + iy)*ng + iz;
    }
    
    int cellid_to_int(const Cell3 a) const { 
      if (a[0]<0 || a[0] >=ng || a[1]<0 || a[1] >=ng || a[2]<0 || a[2] >=ng) return -1;
      return  (static_cast<int> (a[0])*ng + static_cast<int>(a[1]))*ng + a[2]; 
    }

    int num_points_id(const int id) {
      if (id<0) return 0;
      return base[id+1] - base[id];
    }

    void querythiscell(const Cell3 thiscell, const array<Coord_T, 3> q,
                       int &closestpt, double &dist2) {
      const int queryint(qpoint_to_id(q));
      const int npitc(num_points_id(queryint));
      if (npitc<=0) {
        closestpt = -1;
        dist2 = numeric_limits<double>::max();
        return;
      }
      typedef thrust::device_vector<int>::iterator IntItr;
      typedef thrust::permutation_iterator<Coord_3_Iterator, IntItr> PermItr;
      typedef thrust::transform_iterator<distance2_functor<Coord3>, PermItr> dist2_itr;
      PermItr ptsbegin(pts->begin(), cells.begin());
      dist2_itr begin(ptsbegin + base[queryint],
                      distance2_functor<Coord3>(q[0], q[1], q[2]));
      dist2_itr end(ptsbegin + base[queryint+1] - 1,
                    distance2_functor<Coord3>(q[0], q[1], q[2]));
      dist2_itr result = thrust::min_element(begin, end);
      closestpt = cells[result - begin + base[queryint]];
      dist2 = *result;
      return;
    }

    int Query_Fast_Case(const array<Coord_T, 3> q) {
      const int queryint(qpoint_to_id(q));
      const int npitc(num_points_id(queryint));

      #ifdef DEBUG
      cout << PRINTC(q[0]) << PRINTC(q[1]) << PRINTN(q[2]);
      cout << PRINTC(queryint) << PRINTN(npitc);
      cout << PRINTC(base[queryint]) << PRINTN(base[queryint+1]);
      #endif
      
      if (npitc<=0) return -1; // No points in this cell

      int closestpt = -1;

      // Thrust iterator black magic
      typedef thrust::device_vector<int>::iterator IntItr;
      typedef thrust::permutation_iterator<Coord_3_Iterator, IntItr> PermItr;
      typedef thrust::transform_iterator<distance2_functor<Coord3>, PermItr> dist2_itr;
      PermItr ptsbegin(pts->begin(), cells.begin());
      dist2_itr begin(ptsbegin + base[queryint],
                      distance2_functor<Coord3>(q[0], q[1], q[2]));
      dist2_itr end(ptsbegin + base[queryint+1] - 1,
                    distance2_functor<Coord3>(q[0], q[1], q[2]));
      
      Coord_3_Iterator i(pts->begin());
      i += cells[base[queryint]];
      Coord_T x = thrust::get<0>(*i);
      Coord_T y = thrust::get<1>(*i);
      Coord_T z = thrust::get<2>(*i);
      #ifdef DEBUG
      cout << PRINTC(cells[base[queryint]]) << PRINTN(cells[base[queryint+1]-1]);
      cout << PRINTC(*begin) << PRINTN(*end);
      cout << PRINTC(x) << PRINTC(y) << PRINTN(z);
      #endif
      
      dist2_itr result = thrust::min_element(begin, end);
      double dist2 = *result;

      closestpt = cells[result - begin + base[queryint]];
      //closestpt = cells[result - begin];
      
      const double distf = sqrt(dist2) * 1.00001;

      #ifdef DEBUG
      cout << PRINTC(closestpt) << PRINTC(end - begin) <<  PRINTC(result - begin) <<
        PRINTC(result - begin + base[queryint]) << PRINTC(*result) << PRINTN(distf);
      #endif

      array<Coord_T, 3> lopt, hipt;
      for (int i=0; i<3; ++i) {
        lopt[i] = static_cast<unsigned short int> (clamp_USI(static_cast<double>(q[i]) - distf));
        hipt[i] = static_cast<unsigned short int> (clamp_USI(static_cast<double>(q[i]) + distf + 1.0));
      }

      Cell3 locell(Compute_Cell_Containing_Point(lopt));
      Cell3 hicell(Compute_Cell_Containing_Point(hipt));

      clip(locell);
      clip(hicell);

      cout << PRINTC(lopt) << PRINTN(hipt);
      Cell3 qcell(Compute_Cell_Containing_Point(q));
      if (locell == qcell && hicell == qcell) return closestpt;

      
      for (Coord_T x=locell[0]; x<=hicell[0]; x++) {
        for (Coord_T y=locell[1]; y<=hicell[1]; y++) {
          // Do a whole z-row of cells at once.
          const int i01 = (static_cast<int>(x)*ng + static_cast<int>(y))*ng;
          const int i0 = i01 + locell[2];
          const int i1 = i01 + hicell[2];
          #ifdef DEBUG
          cout << PRINTC(x) << PRINTC(y) << PRINTC(i01) <<
            PRINTC(i0) << PRINTC(i1) << PRINTC(base[i0]) << PRINTN(base[i1+1]);
          #endif

          dist2_itr b(ptsbegin + base[i0],
                      distance2_functor<Coord3>(q[0], q[1], q[2]));
          cout << "A" << endl;
          dist2_itr e(ptsbegin + base[i1+1] - 1,
                      distance2_functor<Coord3>(q[0], q[1], q[2]));
          cout << "B" << endl;
          cout << PRINTC(*b) << PRINTN(*e);
          dist2_itr r = thrust::min_element(b, e);
          double d2 = *r;
          cout << PRINTC(base[i1+1]-1) << PRINTC(d2) << PRINTC(r - b) << PRINTC(e - b) << PRINTN(r - b + base[i0]);
          if (d2 < dist2 || (d2==dist2 && cells[r - b + base[i0]]<closestpt)) {
            cout << "C" << endl;
            dist2 = d2;
              closestpt = cells[r - b + base[i0]];
              cout << PRINTC(d2) << PRINTC(r - b + base[i0]) << PRINTN(cells[r - b + base[i0]]);
              cout << "D" << endl;
          }
        }
      }

      cout << PRINTC(closestpt) << PRINTN(dist2);
      cout << endl;
    
      return closestpt;
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

    // Calculate cell id from point
    thrust::transform(pts->begin(), pts->end(), g->cells.begin(),
                      point_to_id_functor<Coord3>(g->ng, g->r_cell, g->d_cell[0],
                                                  g->d_cell[1], g->d_cell[2]));

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
    thrust::counting_iterator<int> count(0);
    thrust::lower_bound(g->cells.begin(), g->cells.end(),
                        count, count + g->ng3 + 1,
                        g->base.begin());

    #ifdef DEBUG
    cout << "Count: " << *count << endl;
    cout << "Lower bound (base): [";
    thrust::copy(g->base.begin(), g->base.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    if (g->base[g->ng3] != nfixpts) {
      cout << "ERROR: Internal inconsistency; wrong " << PRINTN(g->base[g->ng3]);
      throw "Internal inconsistency";
    }
    
    thrust::fill(g->cells.begin(), g->cells.end(), 0);

    // SERIAL
    for (int n=0; n<g->nfixpts; ++n) {
      const int ic(g->point_to_id(n));
      const int pitc = g->cells[g->base[ic+1]-1]++;
      g->cells[g->base[ic]+pitc] = n;
    }
    /*
    thrust::transform(pts->begin(), pts->end(), g->cells.begin(),
                      point_to_id_functor<Coord3>(g->ng, g->r_cell, g->d_cell[0],
                                                  g->d_cell[1], g->d_cell[2]));
    thrust::stable_sort_by_key();
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

    int closestpt(g->Query_Fast_Case(q));
    if (closestpt>=0) {
      return closestpt;
    }

    Cell3 querycell(g->Compute_Cell_Containing_Point(q));

    double dist(numeric_limits<double>::max());
    int closecell(-1);
    int goodsortnum;
    bool foundit(false);
    int nstop(ncellsearchorder);
    
    for (int isort=0; isort<nstop; ++isort) {
      int thisclosest;
      double thisdist;
      Cell3 s (cellsearchorder[isort][0], cellsearchorder[isort][1], 
	       cellsearchorder[isort][2]);

      for (int isign=0; isign<8; isign++) {      // Iterate over all combinations of signs;
        static const int sign3[8][3] = {{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},
                                        {-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}};
        if (s[0]==0 && sign3[isign][0]== -1) continue;
        if (s[1]==0 && sign3[isign][1]== -1) continue;
        if (s[2]==0 && sign3[isign][2]== -1) continue;

        const Cell3 s2(s*sign3[isign]);

        for (int iperm=0; iperm<6; iperm++) {   // Iterate over all permutations of coordinates.
          switch (iperm) {
          case 1:
            if (s[1]==s[2]) continue;
            break;
          case 2: 
            if (s[0]==s[1]) continue;
            break;
          case 3:
          case 4:
            if (s[0]==s[1] && s[0]==s[2]) continue;
            break;
          case 5:
            if (s[0]==s[2]) continue;
            break;
          }
          static const int perm3[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
          const Cell3 s3(s2[perm3[iperm][0]], s2[perm3[iperm][1]], s2[perm3[iperm][2]]);
          const Cell3 c2(querycell+s3);
          if (!g->check(c2)) continue;  // outside the universe?
          goodsortnum = isort;
          g->querythiscell(c2, q, thisclosest, thisdist);
          if (thisclosest < 0) continue;

          // If two fixed points are the same distance from the query, then return the one with the
          // smallest index.  This removes ambiguities, but complicates the code in several places.
          
          if (thisdist<dist || (thisdist==dist && thisclosest<closestpt)) {
            dist = thisdist;
            closestpt = thisclosest;
            closecell =  g->cellid_to_int(c2);
            if (!foundit) {
              foundit = true;
              nstop = cellsearchorder[isort][3];
              if (nstop >= ncellsearchorder) {
                // It took so long to find any cell with a point that cellsearchorder doesn't have
                // enough cells to be sure of finding the closest point.  Fall back to naive
                // exhaustive searching.
                goto L_end_isort;
              }
            }
          }
        }
      }
    }

  L_end_isort: if (closestpt>=0) {
      return closestpt;
    }
    
    // No nearby points, so exhaustively search over all the fixed points.
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord3;
    typedef thrust::device_vector<Coord_T> Coord_Vector;
    typedef typename Coord_Vector::iterator Coord_Iterator;
    typedef thrust::tuple<Coord_Iterator, Coord_Iterator, Coord_Iterator> Coord_Iterator_Tuple;
    typedef thrust::zip_iterator<Coord_Iterator_Tuple> Coord_3_Iterator;
    typedef thrust::device_vector<int>::iterator IntItr;
    typedef thrust::transform_iterator<distance2_functor<Coord3>, Coord_3_Iterator> dist2_itr;
    dist2_itr begin(g->pts->begin(),
                    distance2_functor<Coord3>(q[0], q[1], q[2]));
    dist2_itr end(g->pts->end(),
                    distance2_functor<Coord3>(q[0], q[1], q[2]));
    dist2_itr result = thrust::min_element(begin, end);
    closestpt = g->cells[result - begin];
    
    return closestpt;
  }
  
};

template<typename Coord_T> 
ostream &operator<<(ostream &o, const array<Coord_T,3> &c) {
  o << '(' << c[0] << ',' << c[1] << ',' << c[2] << ')';
  return o;
}