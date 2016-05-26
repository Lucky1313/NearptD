#pragma once

#ifdef DEBUG
#define PROFILE
#endif

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
#include "functors.cu"
#include "cellsearchorder.cu"
#include "grid.cu"
#include "utils.cpp"
#include "timer.cpp"
#include "tuple_utility.cu"

using namespace std;

namespace nearptd {
  
	double ng_factor = 1.6;

  // Process all fixed points into a uniform grid
  template<typename Coord_T, size_t Dim>
  Grid_T<Coord_T, Dim>*
  Preprocess(const int nfixpts, Point_Vector<Coord_T, Dim>* pts) {
    // Typedefs derived from Grid class
    typedef typename Grid_T<Coord_T, Dim>::Coord_Tuple Coord_Tuple;
    typedef typename Grid_T<Coord_T, Dim>::Coord_Tuple_Iterator Coord_Tuple_Iterator;
    typedef typename Grid_T<Coord_T, Dim>::Double_Tuple Double_Tuple;

    #ifdef PROFILE
    Timer pptimer = Timer(true);
    #endif
    
    Grid_T<Coord_T, Dim> *g;
    g = new Grid_T<Coord_T, Dim>;
    
    g->nfixpts = nfixpts;
    int &ng = g->ng;
    // Multiply by the dim root of the number of points
    ng = static_cast<int> (ng_factor * pow(static_cast<double>(nfixpts), 1.0 / static_cast<double>(Dim)));

    ng = min(2000, max(1, ng));
    g->ngd = pow(ng, Dim);
    g->pts = pts;

    // Get min and max of data range
    thrust::pair<Coord_Tuple, Coord_Tuple> minmax = pts->minmax();
    Coord_Tuple lo = thrust::get<0>(minmax);
    Coord_Tuple hi = thrust::get<1>(minmax);

    // Calculate scale for each dimension
    scale<Coord_T> sc(ng);
    tuple_binary_apply<Coord_Tuple, Coord_Tuple, Double_Tuple, scale<Coord_T>, Dim> make_scale;
    Double_Tuple s = make_scale(lo, hi, sc);

    // Find minimum scale of all dimensions
    thrust::minimum<double> min;
    tuple_reduce<Double_Tuple, double, thrust::minimum<double>, Dim> minimum;
    g->r_cell = minimum(s, min);
    // Calculate size of cell for each dimension
    cell_dim<Coord_T> cd(ng, g->r_cell);
    tuple_binary_apply<Coord_Tuple, Coord_Tuple, Double_Tuple, cell_dim<Coord_T>, Dim> make_cell_dim;
    g->d_cell = make_cell_dim(lo, hi, cd);

    #ifdef PROFILE
    pptimer("Create grid parameters");
    #endif
    
    // Create device vectors (Must be before functors)
    g->base = thrust::device_vector<int>(g->ngd+1, 1);
    g->cells = thrust::device_vector<int>(g->nfixpts);

    // Create cell search order
    g->cellsearch = cellsearchorder<Dim>();

    #ifdef PROFILE
    pptimer("Create cell search order");
    #endif
    
    // Define grid functors
    g->check_cell = check_cell_functor<Dim>(g->ng);
    g->clip_cell = clip_cell_functor<Dim>(g->ng);
    g->cell_containing_point = cell_containing_point_functor<Coord_T, Dim>(g->r_cell, g->d_cell);
    g->cell_to_id = cell_to_id_functor<Dim>(g->ng);
    g->point_to_id = point_to_id_functor<Coord_T, Dim>(g->cell_containing_point, g->cell_to_id);
    g->num_points_in_cell_id = num_points_in_cell_id_functor(g->base.data());
    g->query_cell = query_cell_functor<Coord_T, Dim>(g->num_points_in_cell_id,
                                                     g->point_to_id,
                                                     g->base.data(),
                                                     g->cells.data(),
                                                     pts->get_ptrs());
    g->fast_query = fast_query_functor<Coord_T, Dim>(g->clip_cell,
                                                     g->cell_containing_point,
                                                     g->query_cell);
    g->slow_query = slow_query_functor<Coord_T, Dim>(g->cellsearch.size,
                                                     g->cellsearch.cells->get_ptrs(),
                                                     g->cellsearch.stop.data(),
                                                     g->check_cell,
                                                     g->cell_containing_point,
                                                     g->query_cell);
    
    #ifdef DEBUG
    cout << "Grid info:";
    cout << "\nng: " << g->ng;
    cout << "\nngd: " << g->ngd;
    cout << "\nr_cell: " << g->r_cell;
    cout << endl;
    #endif

    #ifdef PROFILE
    pptimer("Create grid functors");
    #endif

    // Create index of cells to reorder cells later
    thrust::device_vector<int> cell_indices(g->nfixpts);
    thrust::sequence(cell_indices.begin(), cell_indices.end());
    
    // Calculate cell id from point
    thrust::transform(pts->begin(), pts->end(), g->cells.begin(), g->point_to_id);
    
    #ifdef PROFILE
    pptimer("Calculate point ids");
    #endif

    // Ensure no cells are -1 (outside range)
    if (thrust::find(g->cells.begin(), g->cells.end(), -1) != g->cells.end()) {
      throw "Bad cell";
    }

    // Sort by ids of cells
    thrust::stable_sort_by_key(g->cells.begin(), g->cells.end(), cell_indices.begin());
    
    // Find where cells start and stop
    thrust::counting_iterator<int> count(0);
    thrust::lower_bound(g->cells.begin(), g->cells.end(),
                        count, count + g->ngd + 1,
                        g->base.begin());

    #ifdef DEBUG
    cout << "Base: [";
    thrust::copy(g->base.begin(), g->base.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    #ifdef PROFILE
    pptimer("Calculate base vector");
    #endif

    if (g->base[g->ngd] != nfixpts) {
      cout << "ERROR: Internal inconsistency; wrong point count: " << g->base[g->ngd];
      throw "Internal inconsistency";
    }

    // Transform iterator to compute point ids 
    typedef thrust::transform_iterator<point_to_id_functor<Coord_T, Dim>, Coord_Tuple_Iterator> IdItr;
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
    thrust::sort_by_key(cell_indices.begin(), cell_indices.end(), g->cells.begin());

    // Offset calculated base indices from permutation iterator by point per cell count
    thrust::transform(g->cells.begin(), g->cells.end(), base_begin, cell_indices.begin(), thrust::plus<int>());

    // Fill cells with increasing count
    thrust::sequence(g->cells.begin(), g->cells.end());

    // Reorder indices by offset base indices
    thrust::stable_sort_by_key(cell_indices.begin(), cell_indices.end(), g->cells.begin());
    
    #ifdef DEBUG
    cout << "Cells: [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    #ifdef PROFILE
    pptimer("Calculate cells vector");
    #endif

    #ifdef STATS
    g->Num_Points_Per_Cell.resize(g->ngd, 0);
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
    g->Avg_Points_Per_Cell = static_cast<float>(nfixpts) / static_cast<float>(g->ngd);
    #endif

    return g;
  }

  // Perform a single query on a coordinate tuple
  template<typename Coord_T, size_t Dim>
  void Query(Grid_T<Coord_T, Dim>* g, const typename ntuple<Coord_T, Dim>::tuple& q, int& closest) {
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

  // Perform a single query on coordinate array, rather than tuple
  template<typename Coord_T, size_t Dim>
  void Query(Grid_T<Coord_T, Dim>* g, Coord_T* q, int& closest) {
    typedef typename Grid_T<Coord_T, Dim>::Coord_Tuple Coord_Tuple;
    ntuple<Coord_T, Dim> Coord_Ntuple;
    Coord_Tuple qt = Coord_Ntuple.make(q);
    Query(g, qt, closest);
  }
  

  // Perform parallel queries
  template<typename Coord_T, size_t Dim>
  void Query(Grid_T<Coord_T, Dim>* g, Point_Vector<Coord_T, Dim>* q, thrust::host_vector<int>* closest) {
    // Typedefs derived from Grid class
    typedef typename Grid_T<Coord_T, Dim>::Coord_Tuple Coord_Tuple;
    typedef typename Grid_T<Coord_T, Dim>::Coord_Tuple_Iterator Coord_Tuple_Iterator;

    #ifdef PROFILE
    Timer qtimer = Timer(true);
    #endif
    
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

    #ifdef PROFILE
    qtimer("Calculate point ids");
    #endif
    
    // Calculate number of points in each query point's cell
    thrust::transform(qcells.begin(), qcells.end(), qcells.begin(), g->num_points_in_cell_id);
    
    #ifdef DEBUG
    cout << "Number of points in cells: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    #ifdef PROFILE
    qtimer("Calculate number of points in cell");
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
    
    #ifdef PROFILE
    qtimer("Partition points");
    #endif

    // Permutation from indices to actual points
    typedef thrust::permutation_iterator<Coord_Tuple_Iterator, IntItr> QueryItr;
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
    
    #ifdef PROFILE
    qtimer("Calculate fast queries");
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
    
    #ifdef PROFILE
    qtimer("Calculate slow queries");
    #endif

    // Any slow case queries that returned -1 need to be done exhaustively.
    // Faster to do parallel search over all points, rather than parallel exhaustive searches
    greater_functor<thrust::tuple<int, int> > positive(-1);
    index_split = thrust::partition(index_begin, index_end, positive);
    split = index_split - index_begin;

    #ifdef DEBUG
    cout << "Repartition: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    #ifdef PROFILE
    qtimer("Repartition");
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
    
    #ifdef PROFILE
    qtimer("Calculate exhaustive queries");
    #endif

    // Resort queries back to given order
    thrust::sort_by_key(qindices.begin(), qindices.end(), qcells.begin());

    #ifdef DEBUG
    cout << "Resorted Query results: [";
    thrust::copy(qcells.begin(), qcells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    // Copy back to host
    thrust::copy(qcells.begin(), qcells.end(), closest->begin());
    
    #ifdef PROFILE
    qtimer("Sort and copy");
    #endif
  }
};