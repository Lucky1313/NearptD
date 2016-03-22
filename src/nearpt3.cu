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
  const static int  cellsearchorder[][4] = {
#include "cellsearchorder"
  };
  // Number of cells in cellsearchorder (before expanding symmetries).
  const static int ncellsearchorder = 
    sizeof(nearpt3::cellsearchorder) / sizeof(nearpt3::cellsearchorder[0][0])/4;

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

    point_to_id_functor<Coord_Tuple> g_point_to_id(g->ng, g->r_cell, g->d_cell[0],
                                                   g->d_cell[1], g->d_cell[2]);

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
    g->cell_indices = thrust::device_vector<int>(g->nfixpts);
    thrust::sequence(g->cell_indices.begin(), g->cell_indices.end());
    
    // Calculate cell id from point
    thrust::transform(pts->begin(), pts->end(), g->cells.begin(), g_point_to_id);

    #ifdef DEBUG
    cout << "Cell IDs (cells): [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif
    
    // Ensure no cells are -1 (outside range)
    if (thrust::find(g->cells.begin(), g->cells.end(), -1) != g->cells.end()) {
      throw "Bad cell";
    }

    // Keep track of the indices of cells during sorting
    thrust::stable_sort_by_key(g->cells.begin(), g->cells.end(), g->cell_indices.begin());

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

    // Transform iterator to compute point ids then permutation iterator to get value from base
    typedef thrust::transform_iterator<point_to_id_functor<Coord_Tuple>, Coord_Iterator_Tuple> ptid_itr;
    ptid_itr ptid_begin(pts->begin(), g_point_to_id);
    ptid_itr ptid_end(pts->end(), g_point_to_id);

    typedef thrust::device_vector<int>::iterator IntItr;
    typedef thrust::permutation_iterator<IntItr, ptid_itr> PermItr;
    PermItr cbbegin(g->base.begin(), ptid_begin);

    // Exclusive scan by key to get count for number of points in cell
    thrust::constant_iterator<int> one(1);
    thrust::exclusive_scan_by_key(g->cells.begin(), g->cells.end(), one, g->cells.begin());

    // 'Undo' previous sort to have an increasing point count per cell
    thrust::stable_sort_by_key(g->cell_indices.begin(), g->cell_indices.end(), g->cells.begin());

    // Offset calculated base indices from permutation iterator by point per cell count
    thrust::plus<int> plus_op;
    thrust::transform(g->cells.begin(), g->cells.end(), cbbegin, g->cell_indices.begin(), plus_op);

    // Fill cells with increasing count
    thrust::sequence(g->cells.begin(), g->cells.end());

    // Reorder indices by offset base indices
    thrust::stable_sort_by_key(g->cell_indices.begin(), g->cell_indices.end(), g->cells.begin());
    
    #ifdef DEBUG
    cout << "Cell indices: [";
    thrust::copy(g->cell_indices.begin(), g->cell_indices.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    cout << "ID (cells iterator): [";
    thrust::copy(ptid_begin, ptid_end, ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    cout << "Permutation (cells): [";
    thrust::copy(cbbegin, cbbegin + g->nfixpts, ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    cout << "Point to cell (cells): [";
    thrust::copy(g->cells.begin(), g->cells.end(), ostream_iterator<int>(cout, ", "));
    cout << "]" << endl;
    #endif

    return g;
  }

  template<typename Coord_T> int
  Query(Grid_T<Coord_T>* g, const array<Coord_T, 3> q) {
    typedef typename Grid_T<Coord_T>::Coord_Tuple Coord_Tuple;
    typedef typename Grid_T<Coord_T>::Coord_Iterator_Tuple Coord_Iterator_Tuple;

    int closestpt(g->Query_Fast_Case(q));
    if (closestpt>=0) {
      return closestpt;
    }
    
    Cell3 querycell(g->Compute_Cell_Containing_Point(q));

    double dist(numeric_limits<double>::max());
    //int closecell(-1);
    //int goodsortnum;
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
          //goodsortnum = isort;
          g->querythiscell(c2, q, thisclosest, thisdist);
          if (thisclosest < 0) continue;

          // If two fixed points are the same distance from the query, then return the one with the
          // smallest index.  This removes ambiguities, but complicates the code in several places.
          
          if (thisdist<dist || (thisdist==dist && thisclosest<closestpt)) {
            dist = thisdist;
            closestpt = thisclosest;
            //closecell = g->cellid_to_int(c2);
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
    typedef thrust::device_vector<int>::iterator IntItr;
    typedef thrust::transform_iterator<distance2_functor<Coord_Tuple>, Coord_Iterator_Tuple> dist2_itr;
    distance2_functor<Coord_Tuple> distance2(q[0], q[1], q[2]);
    dist2_itr begin(g->pts->begin(), distance2);
    dist2_itr end(g->pts->end(), distance2);
    dist2_itr result = thrust::min_element(begin, end);
    closestpt = g->cells[result - begin];
    
    return closestpt;
  }
  
};