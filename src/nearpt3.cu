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

#include "grid.cu"

using namespace std;
using boost::array;

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

  template<typename Coord_T> Grid_T<Coord_T>*
  Preprocess(const int nfixpts, Points_Vector<Coord_T>* pts) {
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

    for (int i=1; i<ncellsearchorder; ++i)
      if (nearpt3::cellsearchorder[(i-1)*4+3] > nearpt3::cellsearchorder[i*4+3]) 
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
      g->d[i] = g->d_cell[i];
    }

    // Create device vectors (Must be before functors)
    g->base = thrust::device_vector<int>(g->ng3+1, 1);
    g->cells = thrust::device_vector<int>(g->nfixpts);
    g->cell_indices = thrust::device_vector<int>(g->nfixpts);
    thrust::sequence(g->cell_indices.begin(), g->cell_indices.end());

    g->cellsearch = thrust::device_vector<int>(cellsearchorder, cellsearchorder+ncellsearchorder*4);

    // Define functors
    g->check_cell = check_cell_functor(g->ng);
    g->clip_cell = clip_cell_functor(g->ng);
    g->cell_containing_point = cell_containing_point_functor<Coord_Tuple>(g->r_cell, g->d[0], g->d[1], g->d[2]);
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
    cout << "\ns: (" << s[0] << ", " << s[1] << ", " << s[2] << ")";
    cout << "\nr_cell: " << g->r_cell;
    cout << "\nd_cell: (" << g->d_cell[0] << ", " << g->d_cell[1] << ", " << g->d_cell[2] << ")";
    cout << endl;
    #endif
    
    // Calculate cell id from point
    thrust::transform(pts->begin(), pts->end(), g->cells.begin(), g->point_to_id);
    
    // Ensure no cells are -1 (outside range)
    if (thrust::find(g->cells.begin(), g->cells.end(), -1) != g->cells.end()) {
      throw "Bad cell";
    }

    // Keep track of the indices of cells during sorting
    //thrust::stable_sort_by_key(g->cells.begin(), g->cells.end(), g->cell_indices.begin());
    thrust::sort_by_key(g->cells.begin(), g->cells.end(), g->cell_indices.begin());
    
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
      cout << "ERROR: Internal inconsistency; wrong " << PRINTN(g->base[g->ng3]);
      throw "Internal inconsistency";
    }

    // Transform iterator to compute point ids then permutation iterator to get value from base
    typedef thrust::transform_iterator<point_to_id_functor<Coord_Tuple>, Coord_Iterator_Tuple> ptid_itr;
    ptid_itr ptid_begin(pts->begin(), g->point_to_id);
    ptid_itr ptid_end(pts->end(), g->point_to_id);

    typedef thrust::device_vector<int>::iterator IntItr;
    typedef thrust::permutation_iterator<IntItr, ptid_itr> PermItr;
    PermItr cbbegin(g->base.begin(), ptid_begin);

    // Exclusive scan by key to get count for number of points in cell
    thrust::constant_iterator<int> one(1);
    thrust::exclusive_scan_by_key(g->cells.begin(), g->cells.end(), one, g->cells.begin());

    // 'Undo' previous sort to have an increasing point count per cell
    //thrust::stable_sort_by_key(g->cell_indices.begin(), g->cell_indices.end(), g->cells.begin());
    thrust::sort_by_key(g->cell_indices.begin(), g->cell_indices.end(), g->cells.begin());

    // Offset calculated base indices from permutation iterator by point per cell count
    thrust::plus<int> plus_op;
    thrust::transform(g->cells.begin(), g->cells.end(), cbbegin, g->cell_indices.begin(), plus_op);

    // Fill cells with increasing count
    thrust::sequence(g->cells.begin(), g->cells.end());

    // Reorder indices by offset base indices
    //thrust::stable_sort_by_key(g->cell_indices.begin(), g->cell_indices.end(), g->cells.begin());
    thrust::sort_by_key(g->cell_indices.begin(), g->cell_indices.end(), g->cells.begin());
    
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

  template<typename Coord_T>
  void Query(Grid_T<Coord_T>* g, Points_Vector<Coord_T>* q, thrust::device_vector<int> *closest) {
    typedef typename Grid_T<Coord_T>::Coord_Tuple Coord_Tuple;
    typedef typename Grid_T<Coord_T>::Coord_Iterator_Tuple Coord_Iterator_Tuple;
    
    const int nqpts(q->size());
    thrust::device_vector<int> qindices(nqpts);
    thrust::sequence(qindices.begin(), qindices.end());

    // Calculate id for query points
    thrust::device_vector<int> qcells(nqpts);
    thrust::transform(q->begin(), q->end(), qcells.begin(), g->point_to_id);

    #ifdef DEBUG
    cout << "Query IDs: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    thrust::transform(qcells.begin(), qcells.end(), qcells.begin(), g->num_points_in_cell_id);
    
    #ifdef DEBUG
    cout << "Number of points in cells: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    // Partition by number of points in a cell, so all non-empty ones are together and can be iterated over
    typedef thrust::device_vector<int>::iterator IntItr;
    typedef thrust::zip_iterator<thrust::tuple<IntItr, IntItr> > ZipItr;
    greater_functor<thrust::tuple<int, int> > greater_zero(0);
    //less_functor<int> less_zero(0);
    // Is greater or less faster?
    // Zip iterator to reorder the cells and indices
    ZipItr zbegin(thrust::make_zip_iterator(thrust::make_tuple(qcells.begin(), qindices.begin())));
    ZipItr zend(thrust::make_zip_iterator(thrust::make_tuple(qcells.end(), qindices.end())));
    ZipItr itr = thrust::partition(zbegin, zend, greater_zero);
    //itr = thrust::partition(query_indices.begin(), query_indices.end(), less_zero);
    
    #ifdef DEBUG
    cout << "Partition: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    cout << "Indices: [";
    thrust::copy(qindices.begin(), qindices.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // If greater than zero, range for fast is begin to itr, if less, itr to end
    // Do fast case query on all query points that have points in the cell
    // Permutation from indices to actual points
    typedef thrust::permutation_iterator<Coord_Iterator_Tuple, IntItr> PermItr;
    PermItr qbegin(q->begin(), qindices.begin());
    thrust::transform(qbegin, qbegin + (itr - zbegin), qcells.begin(), g->fast_query);

    #ifdef DEBUG
    cout << "Fast from (" << 0 << ", " << (itr - zbegin) << ")" << endl;
    cout << "Fast Query Results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Slow case query
    thrust::transform(qbegin + (itr - zbegin), qbegin + nqpts,
                      qcells.begin() + (itr - zbegin), g->slow_query);
    //cout << slowtest(g, *(qbegin+(itr-zbegin))) << endl;
    
    #ifdef DEBUG
    cout << "Slow from (" << (itr - zbegin) << ", " << nqpts << ")" << endl;
    cout << "Slow Query Results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Any slow case queries that returned -1 need to be done exhaustively.
    // It will be faster to do parallel search over all points, rather than parallel exhaustive searches
    greater_functor<thrust::tuple<int, int> > positive(-1);
    itr = thrust::partition(zbegin, zend, positive);

    #ifdef DEBUG
    cout << "Repartition: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    for (int i=(itr - zbegin); i<nqpts; ++i) {
      qcells[i] = g->exhaustive_query((*q)[i]);
    }
    
    #ifdef DEBUG
    cout << "Exhaustive from (" << (itr - zbegin) << ", " << nqpts << ")" << endl;
    cout << "Exhaustive Query Results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Resort query points by index, could be skipped if order of queries doesn't matter
    thrust::sort_by_key(qindices.begin(), qindices.end(), qcells.begin());
    
    #ifdef DEBUG
    cout << "Resorted Query results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Copy results to the closest points array
    thrust::copy(qcells.begin(), qcells.end(), closest->begin());
  }
};