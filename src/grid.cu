#pragma once

#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/extrema.h>

#include "point_vector.cu"
#include "functors.cu"
#include "tuple_utility.cu"

using namespace std;

namespace nearpt3 {

  // Check if this is a legal cell.
  template<size_t Dim>
  struct check_cell_functor
  {
    int ng;

    check_cell_functor() : ng(-1) {}
    check_cell_functor(int ng) : ng(ng) {}

    __host__ __device__
    bool operator()(const Cell<Dim>& a) const {
      for (int i=0; i<Dim; ++i) {
        if (a[i] < 0 || a[i] >= ng) return false;
      }
      return true;
    }
  };
  
  // clip is needed because roundoff errors may cause a number to be slightly outside the legal range
  template<size_t Dim>
  struct clip_cell_functor
  {
    int ng;

    clip_cell_functor() : ng(-1) {}
    clip_cell_functor(int ng) : ng(ng) {}

    __host__ __device__
    void operator()(Cell<Dim>& a) {
      for (size_t i=0; i<Dim; ++i) {
        if (a[i] < 0) a[i] = 0;
        if (a[i] >= ng) a[i] = ng-1;
      }
    }
  };

  template<typename Coord_T, size_t Dim>
  struct cell_containing_point_functor
    : public thrust::unary_function<typename ntuple<Coord_T, Dim>::tuple, Cell<Dim> >
  {
    typedef typename Cell<Dim>::Cell_Index_T Cell_Index_T;
    typedef typename ntuple<double, Dim>::tuple Double_Tuple;
    typedef typename ntuple<Coord_T, Dim>::tuple Coord_Tuple;
    typedef typename ntuple<Cell_Index_T, Dim>::tuple Cell_Tuple;
    double r_cell;
    Double_Tuple d_cell;
    // Only used for making tuple
    ntuple<double, Dim> Double_Ntuple;
    coord_to_cell_index<Coord_T, Cell_Index_T> cts;
    tuple_binary_apply<Coord_Tuple, Double_Tuple, Cell_Tuple,
      coord_to_cell_index<Coord_T, Cell_Index_T>, Dim> make_cell;

    cell_containing_point_functor() : r_cell(-1) {
      double d[Dim];
      for (int i=0; i<Dim; ++i) {d[i] = -1;}
      d_cell = Double_Ntuple.make_tuple(d);
    }

    cell_containing_point_functor(double r_cell, Double_Tuple d_cell)
      : r_cell(r_cell), d_cell(d_cell) {
      cts = coord_to_cell_index<Coord_T, Cell_Index_T>(r_cell);
    }

    __host__ __device__
    Cell<Dim> operator()(const Coord_Tuple& a) const {
      // short int i[Dim];
      // i[0] = static_cast<short int>(static_cast<double>(thrust::get<0>(a))*r_cell+thrust::get<0>(d_cell));
      // i[1] = static_cast<short int>(static_cast<double>(thrust::get<1>(a))*r_cell+thrust::get<1>(d_cell));
      // i[2] = static_cast<short int>(static_cast<double>(thrust::get<2>(a))*r_cell+thrust::get<2>(d_cell));
      // Cell<Dim> c(i);
      Cell<Dim> c(make_cell(a, d_cell, cts));
      return c;
    }
  };

  template<size_t Dim>
  struct cell_to_id_functor : public thrust::unary_function<Cell<Dim>, int>
  {
    int ng;

    cell_to_id_functor() : ng(-1) {}
    cell_to_id_functor(int ng) : ng(ng) {}
  
    __host__ __device__
    int operator()(const Cell<Dim>& c) const {
      int id = 0;
      for (int i=0; i<Dim; ++i) {
        if (c[i] < 0 || c[i] >= ng) return -1;
        id = static_cast<int>(c[i]) + ng * id;
      }
      return id;
    }
  };
  
  template<typename Coord_T, size_t Dim>
  struct point_to_id_functor : public thrust::unary_function<typename ntuple<Coord_T, Dim>::tuple, int>
  {
    typedef typename ntuple<Coord_T, Dim>::tuple Coord_Tuple;
    
    cell_containing_point_functor<Coord_T, Dim> cell_containing_point;
    cell_to_id_functor<Dim> cell_to_id;

    point_to_id_functor() : cell_containing_point(cell_containing_point_functor<Coord_T, Dim>()),
                            cell_to_id(cell_to_id_functor<Dim>()) {}
    
    point_to_id_functor(cell_containing_point_functor<Coord_T, Dim> cell_containing_point,
                        cell_to_id_functor<Dim> cell_to_id)
      : cell_containing_point(cell_containing_point), cell_to_id(cell_to_id) {}

    __host__ __device__
    int operator()(const Coord_Tuple& a) const {
      return cell_to_id(cell_containing_point(a));
    }
  };

  struct num_points_in_cell_id_functor : public thrust::unary_function<int, int>
  {
    thrust::device_ptr<int> base;

    num_points_in_cell_id_functor() : base(thrust::device_ptr<int>()) {}
    num_points_in_cell_id_functor(thrust::device_ptr<int> base) : base(base) {}

    __host__ __device__
    int operator()(const int& id) const {
      if (id < 0) return 0;
      return base[id+1] - base[id];
    }
  };


  template<typename Coord_T, size_t Dim>
  struct query_cell_functor
  {
    typedef typename ntuple<Coord_T, Dim>::tuple Coord_Tuple;
    typedef thrust::device_ptr<Coord_T> Coord_Ptr;
    typedef typename ntuple<Coord_Ptr, Dim>::tuple Coord_Ptr_Tuple;
    typedef typename ntuple<int, Dim>::tuple Int_Tuple;
    ntuple<Coord_Ptr, Dim> Coord_Ptr_Ntuple;
    ntuple<int, Dim> Int_Ntuple;
    get_point<Coord_Ptr, Coord_T> point;
    tuple_binary_apply<Coord_Ptr_Tuple, Int_Tuple,
      Coord_Tuple, get_point<Coord_Ptr, Coord_T>, Dim> pget;
    
    num_points_in_cell_id_functor num_points_in_cell_id;
    point_to_id_functor<Coord_T, Dim> point_to_id;
    thrust::device_ptr<int> base;
    thrust::device_ptr<int> cells;
    Coord_Ptr_Tuple pts;

    query_cell_functor() : num_points_in_cell_id(num_points_in_cell_id_functor()),
                           point_to_id(point_to_id_functor<Coord_T, Dim>()),
                           base(thrust::device_ptr<int>()),
                           cells(thrust::device_ptr<int>()) {
      Coord_Ptr ptrs[Dim];
      for (size_t i=0; i<Dim; ++i) {ptrs[i] = Coord_Ptr();}
      pts = Coord_Ptr_Ntuple.make_tuple(ptrs);
    }
    
    query_cell_functor(num_points_in_cell_id_functor num_points_in_cell_id,
                       point_to_id_functor<Coord_T, Dim> point_to_id,
                       thrust::device_ptr<int> base,
                       thrust::device_ptr<int> cells,
                       Coord_Ptr_Tuple pts)
      : num_points_in_cell_id(num_points_in_cell_id), point_to_id(point_to_id),
        base(base), cells(cells), pts(pts) {}

    __host__ __device__
    Coord_Tuple point_at(int i) {
      int id[Dim];
      for (int d=0; d<Dim; ++d) {id[d] = i;}
      Int_Tuple id_tup = Int_Ntuple.make_tuple(id);
      return pget(pts, id_tup, point);
    }
    
    __host__ __device__
    void operator()(const int &cell_id, const Coord_Tuple& q,
                    int &closest, double &dist2) {
      const int num_points(num_points_in_cell_id(cell_id));
      if (num_points <= 0) {
        closest = -1;
        dist2 = -1;
        return;
      }
      
      const int queryint(point_to_id(q));
      distance2_functor<Coord_T, Dim> distance2(q);
      int i = base[cell_id];
      closest = cells[i];
      dist2 = distance2(point_at(closest));
      while (i < base[cell_id+1]) {
        const double d2 = distance2(point_at(cells[i]));
        if (d2 < dist2 || (d2 == dist2 && cells[i] < closest)) {
          dist2 = d2;
          closest = cells[i];
        }
        ++i;
      }
    }
  };

  template<typename Coord_T, size_t Dim>
  struct fast_query_functor : public thrust::unary_function<typename ntuple<Coord_T, Dim>::tuple, int>
  {
    typedef typename ntuple<Coord_T, Dim>::tuple Coord_Tuple;
    tuple_unary_apply<Coord_Tuple, Coord_Tuple, near_cell<Coord_T>, Dim> dc;
    
    clip_cell_functor<Dim> clip_cell;
    cell_containing_point_functor<Coord_T, Dim> cell_containing_point;
    query_cell_functor<Coord_T, Dim> query_cell;

    fast_query_functor() : clip_cell(clip_cell_functor<Dim>()),
                           cell_containing_point(cell_containing_point_functor<Coord_T, Dim>()),
                           query_cell(query_cell_functor<Coord_T, Dim>()) {}

    fast_query_functor(clip_cell_functor<Dim> clip_cell,
                       cell_containing_point_functor<Coord_T, Dim> cell_containing_point,
                       query_cell_functor<Coord_T, Dim> query_cell)
      : clip_cell(clip_cell), cell_containing_point(cell_containing_point), query_cell(query_cell) {}
    
    __host__ __device__
    int operator()(const Coord_Tuple& q) {
      int queryint = query_cell.point_to_id(q);
      int closestpt = -1;
      double dist2 = -1;
      query_cell(queryint, q, closestpt, dist2);
      const double distf = sqrt(dist2) * 1.00001;
      // Coord_Tuple lopt(thrust::make_tuple(clamp_USI(static_cast<double>(thrust::get<0>(q)) - distf),
      //                                     clamp_USI(static_cast<double>(thrust::get<1>(q)) - distf),
      //                                     clamp_USI(static_cast<double>(thrust::get<2>(q)) - distf)));
      // Coord_Tuple hipt(thrust::make_tuple(clamp_USI(static_cast<double>(thrust::get<0>(q)) + distf + 1.0),
      //                                     clamp_USI(static_cast<double>(thrust::get<1>(q)) + distf + 1.0),
      //                                     clamp_USI(static_cast<double>(thrust::get<2>(q)) + distf + 1.0)));
      near_cell<Coord_T> near_cell_lo(distf, true);
      near_cell<Coord_T> near_cell_hi(distf, false);
      
      Coord_Tuple lopt = dc(q, near_cell_lo);
      Coord_Tuple hipt = dc(q, near_cell_hi);
      
      Cell<Dim> locell(cell_containing_point(lopt));
      Cell<Dim> hicell(cell_containing_point(hipt));

      clip_cell(locell);
      clip_cell(hicell);

      Cell<Dim> qcell(cell_containing_point(q));
      if (locell == qcell && hicell == qcell) {
        return closestpt;
      }
      int close2 = -1;
      double d2 = -1;
      // for (Coord_T x=locell[0]; x<=hicell[0]; ++x) {
      //   for (Coord_T y=locell[1]; y<=hicell[1]; ++y) {
      //     for (Coord_T z=locell[2]; z<=hicell[2]; ++z) {
      //       queryint = query_cell.point_to_id.cell_to_id(Cell<Dim>(x, y, z));
      //       query_cell(queryint, q, close2, d2);
      //       if (close2 != -1 && (d2 < dist2 || (d2 == dist2 && close2 < closestpt))) {
      //         closestpt = close2;
      //         dist2 = d2;
      //       }
      //     }
      //   }
      // }
      
      // Nested loop traversal, modified from this code:
      // http://stackoverflow.com/questions/18732974/c-dynamic-number-of-nested-for-loops-without-recursion
      short int coords[Dim];
      size_t index = 0;

      // While loop causes major slowdown, precompute number of iterations
      int itrs = 1;
      for (size_t i=0; i<Dim; ++i) {
        coords[i] = locell[i];
        itrs *= (hicell[i] - locell[i] + 1);
      }
      for (int i=0; i<itrs; ++i) {
        queryint = query_cell.point_to_id.cell_to_id(Cell<Dim>(coords));
        query_cell(queryint, q, close2, d2);
        if (close2 != -1 && (d2 < dist2 || (d2 == dist2 && close2 < closestpt))) {
          closestpt = close2;
          dist2 = d2;
        }
        coords[0]++;

        while (coords[index] == hicell[index]) {
          if (index != Dim - 1) {
            coords[index] = locell[index];
            index++;
            coords[index]++;
          }
          else {
            break;
          }
        }
        index = 0;
      }

      return closestpt;
    }
  };

  template<typename Coord_T, size_t Dim>
  struct slow_query_functor : public thrust::unary_function<typename ntuple<Coord_T, Dim>::tuple, int>
  {
    typedef typename ntuple<Coord_T, Dim>::tuple Coord_Tuple;
    typedef typename Cell<Dim>::Cell_Index_T Cell_Index_T;
    typedef typename ntuple<Cell_Index_T, Dim>::tuple Cell_Tuple;
    typedef thrust::device_ptr<Cell_Index_T> Cell_Ptr;
    typedef typename ntuple<Cell_Ptr, Dim>::tuple Cell_Ptr_Tuple;
    typedef typename ntuple<int, Dim>::tuple Int_Tuple;
    ntuple<int, Dim> Int_Ntuple;
    ntuple<Cell_Ptr, Dim> Cell_Ptr_Ntuple;
    
    int ncellsearch;
    Cell_Ptr_Tuple cellsearch;
    thrust::device_ptr<int> cellstop;
    sign<Dim> signs;
    perm<Dim> perms;
    Cell_Ptr_Tuple pts;
    get_point<Cell_Ptr, Cell_Index_T> point;
    tuple_binary_apply<Cell_Ptr_Tuple, Int_Tuple,
      Cell_Tuple, get_point<Cell_Ptr, Cell_Index_T>, Dim> pget;

    check_cell_functor<Dim> check_cell;
    cell_containing_point_functor<Coord_T, Dim> cell_containing_point;
    query_cell_functor<Coord_T, Dim> query_cell;

    slow_query_functor() : ncellsearch(0),
                           cellstop(thrust::device_ptr<int>()),
                           check_cell(check_cell_functor<Dim>()),
                           cell_containing_point(cell_containing_point_functor<Coord_T, Dim>()),
                           query_cell(query_cell_functor<Coord_T, Dim>()) {
      signs = sign<Dim>();
      perms = perm<Dim>();
      Cell_Ptr ptrs[Dim];
      for (size_t i=0; i<Dim; ++i) {ptrs[i] = Cell_Ptr();}
      pts = Cell_Ptr_Ntuple.make_tuple(ptrs);
    }

    slow_query_functor(int ncellsearch,
                       Cell_Ptr_Tuple cellsearch,
                       thrust::device_ptr<int> cellstop,
                       check_cell_functor<Dim> check_cell,
                       cell_containing_point_functor<Coord_T, Dim> cell_containing_point,
                       query_cell_functor<Coord_T, Dim> query_cell)
      : ncellsearch(ncellsearch), cellsearch(cellsearch), cellstop(cellstop), check_cell(check_cell),
        cell_containing_point(cell_containing_point), query_cell(query_cell) {
      signs = sign<Dim>();
      perms = perm<Dim>();
    }

    __host__ __device__
    Cell_Tuple point_at(int i) {
      int id[Dim];
      for (int d=0; d<Dim; ++d) {id[d] = i;}
      Int_Tuple id_tup = Int_Ntuple.make_tuple(id);
      return pget(cellsearch, id_tup, point);
    }

    __host__ __device__
    int operator()(const Coord_Tuple& q) {
      // const int sign3[8][3] = {{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},
      //                          {-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}};
      // const int perm3[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};

      Cell<Dim> qcell = cell_containing_point(q);
      int queryint = query_cell.point_to_id.cell_to_id(qcell);
      int closestpt = -1;
      double dist2 = -1;

      int nstop(ncellsearch);
      bool found(false);

      for (int isort=0; isort<nstop; ++isort) {
        int close2;
        double d2;
        const Cell<Dim> s(point_at(isort));

        for (int isign=0; isign<1<<Dim; ++isign) {
          bool skip(false);
          for (size_t i=0; i<Dim; ++i) {
            if (s[i]==0 && signs[isign][i]==-1) {
              skip = true;
              break;
            }
          }
          if (skip) continue;
          
          const Cell<Dim> s2(s*signs[isign]);

          for (int iperm=0; iperm<factorial<Dim>::value; ++iperm) {
            // switch (iperm) {
            // case 1:
            //   if (s[1]==s[2]) continue;
            //   break;
            // case 2: 
            //   if (s[0]==s[1]) continue;
            //   break;
            // case 3:
            // case 4:
            //   if (s[0]==s[1] && s[0]==s[2]) continue;
            //   break;
            // case 5:
            //   if (s[0]==s[2]) continue;
            //   break;
            // }
            if (iperm > 0 && (s[iperm % Dim] == s[(iperm-1) % Dim])) continue;
            
            Cell<Dim> s3;
            //(s2[perms[iperm][0]], s2[perms[iperm][1]], s2[perms[iperm][2]]);
            for (size_t i=0; i<Dim; ++i) {
              s3[i] = s2[perms[iperm][i]];
            }
            
            const Cell<Dim> c2(qcell+s3);
            if (!check_cell(c2)) continue;
            int cell_id(query_cell.point_to_id.cell_to_id(c2));
            query_cell(cell_id, q, close2, d2);
            if (close2 < 0) continue;

            if (dist2 == -1 || d2 < dist2 || (d2 == dist2 && close2 < closestpt)) {
              dist2 = d2;
              closestpt = close2;
              if (!found) {
                found = true;
                nstop = cellstop[isort];
                if (nstop >= ncellsearch) {
                  iperm = 1<<Dim;
                  isign = factorial<Dim>::value;
                  isort = nstop;
                }
              }
            }
          }
        }
      }
      return closestpt;
    }
  };

  template<typename Coord_T, size_t Dim>
  class Grid_T {
  public:
    // Typedefs from Point_Vector class
    typedef typename Point_Vector<Coord_T, Dim>::Coord_Tuple Coord_Tuple;
    typedef typename Point_Vector<Coord_T, Dim>::Coord_Tuple_Iterator Coord_Tuple_Iterator;
    typedef typename ntuple<double, Dim>::tuple Double_Tuple;

    int ng;
    int ng3;
    double r_cell;
    Double_Tuple d_cell;
    int nfixpts;
    Point_Vector<Coord_T, Dim>* pts;
    thrust::device_vector<int> cells;
    thrust::device_vector<int> base;
    //thrust::device_vector<int> cellsearch;
    cellsearchcreate<Dim> cellsearch;

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
    check_cell_functor<Dim> check_cell;
    clip_cell_functor<Dim> clip_cell;
    cell_containing_point_functor<Coord_T, Dim> cell_containing_point;
    cell_to_id_functor<Dim> cell_to_id;
    point_to_id_functor<Coord_T, Dim> point_to_id;
    num_points_in_cell_id_functor num_points_in_cell_id;
    query_cell_functor<Coord_T, Dim> query_cell;
    fast_query_functor<Coord_T, Dim> fast_query;
    slow_query_functor<Coord_T, Dim> slow_query;

    int exhaustive_query(const Coord_Tuple& q) {
      typedef thrust::transform_iterator<distance2_functor<Coord_T, Dim>, Coord_Tuple_Iterator> dist2_itr;
      distance2_functor<Coord_T, Dim> distance2(q);
      dist2_itr begin(pts->begin(), distance2);
      dist2_itr end(pts->end(), distance2);
      dist2_itr result = thrust::min_element(begin, end);
      int closestpt = result - begin;
      return closestpt;
    }
  };
};