#pragma once

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/extrema.h>

#include "point_vector.cu"
#include "functors.cu"

using namespace std;

namespace nearpt3 {

  template<typename Coord_T>
  class Grid_T {
  public:
    // Typedefs from Point_Vector class
    typedef typename Point_Vector<Coord_T>::Coord_Tuple Coord_Tuple;
    typedef typename Point_Vector<Coord_T>::Coord_Iterator_Tuple Coord_Iterator_Tuple;

    int ng;
    int ng3;
    double r_cell;
    Coord_Tuple d_cell;
    int nfixpts;
    Point_Vector<Coord_T>* pts;
    thrust::device_vector<int> cells;
    thrust::device_vector<int> base;
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