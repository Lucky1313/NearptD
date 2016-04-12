#include <thrust/device_vector.h>

#include <boost/multi_array.hpp>
#include <iostream>
#include <vector>

#include "points.cu"
#include "functors.cu"

using namespace std;
using boost::array;

// Print an expression's name then its value, possibly followed by a comma or endl.  
// Ex: cout << PRINTC(x) << PRINTN(y);

#define PRINT(arg)  #arg "=" << (arg)
#define PRINTC(arg)  #arg "=" << (arg) << ", "
#define PRINTN(arg)  #arg "=" << (arg) << endl

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

  public:
    int ng;
    int ng3;
    double r_cell;
    array<double,3> d_cell;
    double d[3];
    int nfixpts;
    Points_Vector<Coord_T>* pts;
    thrust::device_vector<int> cells;
    thrust::device_vector<int> base;
    thrust::device_vector<int> cell_indices;
    thrust::device_vector<int> cellsearch;

    #ifdef STATS
    thrust::device_vector<int> Num_Points_Per_Cell;
    int Min_Points_Per_Cell;
    int Max_Points_Per_Cell;
    float Avg_Points_Per_Cell;
    int Num_Fast_Queries;
    int Num_Slow_Queries;
    int Num_Exhaustive_Queries;
    static const int Max_Cells_Searched = 1000;
    vector<int> Num_Cells_Searched;
    int Total_Cells_Searched;
    static const int Max_Points_Checked = 10000;
    vector<int> Num_Points_Checked;
    int Total_Points_Checked;
    int Points_Checked;
    #endif

    // Typedefs from Point_Vector class
    typedef typename Points_Vector<Coord_T>::Coord_Tuple Coord_Tuple;
    typedef typename Points_Vector<Coord_T>::Coord_Iterator_Tuple Coord_Iterator_Tuple;

    // Functors
    check_cell_functor check_cell;
    clip_cell_functor clip_cell;
    cell_containing_point_functor<Coord_Tuple> cell_containing_point;
    cell_to_id_functor cell_to_id;
    point_to_id_functor<Coord_Tuple> point_to_id;
    num_points_in_cell_id_functor num_points_in_cell_id;
    query_cell_functor<Coord_T> query_cell;
    fast_query_functor<Coord_T> fast_query;
    slow_query_functor<Coord_T> slow_query;

    int exhaustive_query(const Coord_Tuple& q) {
      typedef thrust::transform_iterator<distance2_functor<Coord_Tuple>, Coord_Iterator_Tuple> dist2_itr;
      distance2_functor<Coord_Tuple> distance2(q);
      dist2_itr begin(pts->begin(), distance2);
      dist2_itr end(pts->end(), distance2);
      dist2_itr result = thrust::min_element(begin, end);
      int closestpt = result - begin;
      return closestpt;
    }
  };
};