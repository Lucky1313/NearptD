#include <thrust/device_vector.h>

#include <boost/multi_array.hpp>
#include <iostream>

#include "points.cu"
#include "functors.cu"
#include "cell.cpp"

using namespace std;
using boost::array;

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

template <typename T>
void write(ostream &o, const thrust::tuple<T, T, T>& c) {
  o << "(" << thrust::get<0>(c) << "," << thrust::get<1>(c) << "," << thrust::get<2>(c) << ")";
}

template <typename T>
void write(ostream &o, const array<T,3>& c) {
    o << "(" << c[0] << "," << c[1] << "," << c[2] << ")";
}

template<typename Coord_T> 
ostream &operator<<(ostream &o, const array<Coord_T,3> &c) {
  o << '(' << c[0] << ',' << c[1] << ',' << c[2] << ')';
  return o;
}

namespace nearpt3 {

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

};