#pragma once

#include <iostream>

#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#ifdef STATS
#include <thrust/adjacent_difference.h>
#endif

#include "point_vector.cu"
#include "grid.cu"
#include "functors.cu"
#include "utils.cpp"

using namespace std;

namespace nearpt3 {
  
	double ng_factor = 1.6;

  // cellsearchorder:
  // First 3 elements of each row:  the order in which to search (one 48th-ant of the) cells adjacent to the current cell.
  // 4th element:   where, in cellsearchorder, to stop searching after the first point is found.
  const int cellsearchorder[] = {
#include "cellsearchorder"
  };
  // Number of cells in cellsearchorder (before expanding symmetries).
  const int ncellsearchorder = 
    sizeof(nearpt3::cellsearchorder) / sizeof(nearpt3::cellsearchorder[0])/4;

  // Process all fixed points into a uniform grid
  template<typename Coord_T> Grid_T<Coord_T>*
  Preprocess(const int nfixpts, Point_Vector<Coord_T>* pts) {
    // Typedefs derived from Grid class
    typedef typename Grid_T<Coord_T>::Coord_Tuple Coord_Tuple;
    typedef typename Grid_T<Coord_T>::Coord_Iterator_Tuple Coord_Iterator_Tuple;
    
    Grid_T<Coord_T> *g;
    g = new Grid_T<Coord_T>;
    
    g->nfixpts = nfixpts;
    int &ng = g->ng;
    ng = static_cast<int> (ng_factor * cbrt(static_cast<double>(nfixpts)));

    ng = min(2000, max(1, ng));
    g->ng3 = ng * ng * ng;
    g->pts = pts;

    // Ensure monotonic cell search order
    for (int i=1; i<ncellsearchorder; ++i)
      if (nearpt3::cellsearchorder[(i-1)*4+3] > nearpt3::cellsearchorder[i*4+3]) 
        throw "cellsearchorder is not monotonic";

    // Get min and max of data range
    thrust::pair<Coord_Tuple, Coord_Tuple> minmax = pts->minmax();
    Coord_Tuple lo = thrust::get<0>(minmax);
    Coord_Tuple hi = thrust::get<1>(minmax);

    #ifdef DEBUG
    cout << "Min/Max" << endl;
    cout << thrust::get<0>(lo) << ", " << thrust::get<1>(lo) << ", " << thrust::get<2>(lo) << endl;
    cout << thrust::get<0>(hi) << ", " << thrust::get<1>(hi) << ", " << thrust::get<2>(hi) << endl;
    #endif

    Double_Tuple s(thrust::make_tuple(0.99 * ng / static_cast<double>(thrust::get<0>(hi) - thrust::get<0>(lo)),
                                      0.99 * ng / static_cast<double>(thrust::get<1>(hi) - thrust::get<1>(lo)),
                                      0.99 * ng / static_cast<double>(thrust::get<2>(hi) - thrust::get<2>(lo))));
    
    g->r_cell = min(min(thrust::get<0>(s), thrust::get<1>(s)), thrust::get<2>(s));
    g->d_cell = thrust::make_tuple(((ng-1)-(thrust::get<0>(lo)+thrust::get<0>(hi))*g->r_cell) * 0.5,
                                   ((ng-1)-(thrust::get<1>(lo)+thrust::get<1>(hi))*g->r_cell) * 0.5,
                                   ((ng-1)-(thrust::get<2>(lo)+thrust::get<2>(hi))*g->r_cell) * 0.5);

    // Create device vectors (Must be before functors)
    g->base = thrust::device_vector<int>(g->ng3+1, 1);
    g->cells = thrust::device_vector<int>(g->nfixpts);

    // Copy cell search order to device memory
    g->cellsearch = thrust::device_vector<int>(cellsearchorder, cellsearchorder+ncellsearchorder*4);

    // Define grid functors
    g->check_cell = check_cell_functor(g->ng);
    g->clip_cell = clip_cell_functor(g->ng);
    g->cell_containing_point = cell_containing_point_functor<Coord_Tuple>(g->r_cell, g->d_cell);
    g->cell_to_id = cell_to_id_functor(g->ng);
    g->point_to_id = point_to_id_functor<Coord_Tuple>(g->cell_containing_point, g->cell_to_id);
    g->num_points_in_cell_id = num_points_in_cell_id_functor(g->base.data());
    g->query_cell = query_cell_functor<Coord_T>(g->num_points_in_cell_id,
                                                g->point_to_id,
                                                g->base.data(),
                                                g->cells.data(),
                                                pts->get_ptrs());
    g->fast_query = fast_query_functor<Coord_T>(g->clip_cell,
                                                g->cell_containing_point,
                                                g->query_cell);
    g->slow_query = slow_query_functor<Coord_T>(ncellsearchorder,
                                                g->cellsearch.data(),
                                                g->check_cell,
                                                g->cell_containing_point,
                                                g->query_cell);
    
    #ifdef DEBUG
    cout << "Grid info:";
    cout << "\nng: " << g->ng;
    cout << "\nng3: " << g->ng3;
    cout << "\ns: (" << thrust::get<0>(s) << ", " << thrust::get<1>(s) << ", " << thrust::get<2>(s) << ")";
    cout << "\nr_cell: " << g->r_cell;
    cout << "\nd_cell: (" << thrust::get<0>(g->d_cell) << ", " << thrust::get<1>(g->d_cell) << ", " << thrust::get<2>(g->d_cell) << ")";
    cout << endl;
    #endif

    // Index of cells to safely reorder cells
    thrust::device_vector<int> cell_indices(g->nfixpts);
    thrust::sequence(cell_indices.begin(), cell_indices.end());
    
    // Calculate cell id from point
    thrust::transform(pts->begin(), pts->end(), g->cells.begin(), g->point_to_id);
    
    // Ensure no cells are -1 (outside range)
    if (thrust::find(g->cells.begin(), g->cells.end(), -1) != g->cells.end()) {
      throw "Bad cell";
    }

    // Keep track of the indices of cells during sorting
    thrust::stable_sort_by_key(g->cells.begin(), g->cells.end(), cell_indices.begin());
    //thrust::sort_by_key(g->cells.begin(), g->cells.end(), cell_indices.begin());
    
    // Taken from thrust histogram example
    thrust::counting_iterator<int> count(0);
    thrust::lower_bound(g->cells.begin(), g->cells.end(),
                        count, count + g->ng3 + 1,
                        g->base.begin());

    #ifdef DEBUG
    cout << "Base: [";
    thrust::copy(g->base.begin(), g->base.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    if (g->base[g->ng3] != nfixpts) {
      cout << "ERROR: Internal inconsistency; wrong point count: " << g->base[g->ng3];
      throw "Internal inconsistency";
    }

    // Transform iterator to compute point ids 
    typedef thrust::transform_iterator<point_to_id_functor<Coord_Tuple>, Coord_Iterator_Tuple> IdItr;
    IdItr id_begin(pts->begin(), g->point_to_id);
    IdItr id_end(pts->end(), g->point_to_id);

    // Permutation from calculated point id to value in base vector
    typedef thrust::device_vector<int>::iterator IntItr;
    typedef thrust::permutation_iterator<IntItr, IdItr> BaseItr;
    BaseItr base_begin(g->base.begin(), id_begin);

    // Exclusive scan by key to get count for number of points in cell
    thrust::constant_iterator<int> one(1);
    thrust::exclusive_scan_by_key(g->cells.begin(), g->cells.end(), one, g->cells.begin());

    // 'Undo' previous sort to have an increasing point count per cell
    //thrust::stable_sort_by_key(cell_indices.begin(), cell_indices.end(), g->cells.begin());
    thrust::sort_by_key(cell_indices.begin(), cell_indices.end(), g->cells.begin());

    // Offset calculated base indices from permutation iterator by point per cell count
    thrust::transform(g->cells.begin(), g->cells.end(), base_begin, cell_indices.begin(), thrust::plus<int>());

    // Fill cells with increasing count
    thrust::sequence(g->cells.begin(), g->cells.end());

    // Reorder indices by offset base indices
    thrust::stable_sort_by_key(cell_indices.begin(), cell_indices.end(), g->cells.begin());
    //thrust::sort_by_key(cell_indices.begin(), cell_indices.end(), g->cells.begin());
    
    #ifdef DEBUG
    cout << "Cells: [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    #ifdef STATS
    g->Num_Points_Per_Cell.resize(g->ng3, 0);
    g->Num_Cells_Searched.resize(g->Max_Cells_Searched+1, 0);
    g->Num_Points_Checked.resize(g->Max_Points_Checked+1, 0);

    g->Min_Points_Per_Cell = -1;
    g->Max_Points_Per_Cell = -1;
    g->Avg_Points_Per_Cell = -1;
    g->Num_Fast_Queries = 0;
    g->Num_Slow_Queries = 0;
    g->Num_Exhaustive_Queries = 0;
    g->Total_Cells_Searched = 0;
    g->Total_Points_Checked = 0;
    g->Points_Checked = 0;

    thrust::adjacent_difference(g->base.begin(), g->base.end(), g->Num_Points_Per_Cell.begin());
    g->Min_Points_Per_Cell = *thrust::min_element(g->Num_Points_Per_Cell.begin()+1,
                                                  g->Num_Points_Per_Cell.end());
    g->Max_Points_Per_Cell = *thrust::max_element(g->Num_Points_Per_Cell.begin()+1,
                                                  g->Num_Points_Per_Cell.end());
    g->Avg_Points_Per_Cell = static_cast<float>(nfixpts) / static_cast<float>(g->ng3);
    #endif

    return g;
  }

  // Perform a single query
  template<typename Coord_T>
  void Query(Grid_T<Coord_T>* g, thrust::tuple<Coord_T, Coord_T, Coord_T>& q, int& closest) {
    // Get id of cell containing query
    const int queryint(g->point_to_id(q));
    // Get number of points in cell
    const int num_points_in_cell(g->num_points_in_cell_id(queryint));

    // If cell contains any points, perform a fast query
    if (num_points_in_cell > 0) {
      closest = g->fast_query(q);
    }
    else {
      // Perform a slow query
      closest = g->slow_query(q);
      // If query failed do exhaustive search
      if (closest < 0) {
        closest = g->exhaustive_query(q);
      }
    }
  }
  

  // Parallel query on preprocessed grid
  template<typename Coord_T>
  void Query(Grid_T<Coord_T>* g, Point_Vector<Coord_T>* q, thrust::host_vector<int>* closest) {
    // Typedefs derived from Grid class
    typedef typename Grid_T<Coord_T>::Coord_Tuple Coord_Tuple;
    typedef typename Grid_T<Coord_T>::Coord_Iterator_Tuple Coord_Iterator_Tuple;
    
    // Initialize vector of indices
    const int nqpts(q->get_size());
    thrust::device_vector<int> qindices(nqpts);
    thrust::sequence(qindices.begin(), qindices.end());
    thrust::device_vector<int> qcells(nqpts, -1);
    
    // Calculate cell id for query points
    thrust::transform(q->begin(), q->end(), qcells.begin(), g->point_to_id);

    #ifdef DEBUG
    cout << "Query IDs: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Calculate number of points in each query point's cell
    thrust::transform(qcells.begin(), qcells.end(), qcells.begin(), g->num_points_in_cell_id);
    
    #ifdef DEBUG
    cout << "Number of points in cells: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    // Zip iterator to reorder the cells and indices at same time
    typedef thrust::device_vector<int>::iterator IntItr;
    typedef thrust::zip_iterator<thrust::tuple<IntItr, IntItr> > ZipItr;
    greater_functor<thrust::tuple<int, int> > greater_zero(0);
    ZipItr index_begin(thrust::make_zip_iterator(thrust::make_tuple(qcells.begin(), qindices.begin())));
    ZipItr index_end(thrust::make_zip_iterator(thrust::make_tuple(qcells.end(), qindices.end())));
    // Partition by number of points in a cell, so all non-empty ones are together and can be iterated over
    ZipItr index_split = thrust::partition(index_begin, index_end, greater_zero);
    // Index of where the partition was split
    int split = index_split - index_begin;

    #ifdef DEBUG
    cout << "Partition: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    cout << "Indices: [";
    thrust::copy(qindices.begin(), qindices.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Permutation from indices to actual points
    typedef thrust::permutation_iterator<Coord_Iterator_Tuple, IntItr> QueryItr;
    QueryItr qbegin(q->begin(), qindices.begin());
    // Do fast case query on all query points that have points in the cell
    thrust::transform(qbegin, qbegin + split, qcells.begin(), g->fast_query);
    #ifdef STATS
    g->Num_Fast_Queries = split;
    #endif

    #ifdef DEBUG
    cout << "Fast on (" << 0 << ", " << split << ")" << endl;
    cout << "Fast Query Results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Slow case query on empty cell queries
    thrust::transform(qbegin + split, qbegin + nqpts, qcells.begin() + split, g->slow_query);
    #ifdef STATS
    g->Num_Slow_Queries = nqpts - split;
    #endif

    #ifdef DEBUG
    cout << "Slow on (" << split << ", " << nqpts << ")" << endl;
    cout << "Slow Query Results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Any slow case queries that returned -1 need to be done exhaustively.
    // It will be faster to do parallel search over all points, rather than parallel exhaustive searches
    greater_functor<thrust::tuple<int, int> > positive(-1);
    index_split = thrust::partition(index_begin, index_end, positive);
    split = index_split - index_begin;

    #ifdef DEBUG
    cout << "Repartition: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Perform exhaustive queries
    for (int i = split; i < nqpts; ++i) {
      qcells[i] = g->exhaustive_query((*q)[i]);
    }
    
    #ifdef STATS
    g->Num_Exhaustive_Queries = nqpts - split;
    #endif

    #ifdef DEBUG
    cout << "Exhaustive on (" << split << ", " << nqpts << ")" << endl;
    cout << "Exhaustive Query Results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    thrust::sort_by_key(qindices.begin(), qindices.end(), qcells.begin());

    #ifdef DEBUG
    cout << "Resorted Query results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Copy back to host
    thrust::copy(qcells.begin(), qcells.end(), closest->begin());
  }
};