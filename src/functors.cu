#include <thrust/functional.h>
#include <limits>

#include "cell.cu"

namespace nearpt3 {

  
  // clamp_USI     Convert to an unsigned short int while clamping

  template <typename T> __host__ __device__
  unsigned short int clamp_USI(T a) {
    const T mm(static_cast<T>(USHRT_MAX));
    return  
      static_cast<unsigned short int>(a > mm ? mm : (a > 0 ? static_cast<unsigned short int>(a) : 0));
  }

  // Check if this is a legal cell.
  struct check_cell_functor
  {
    int ng;

    check_cell_functor() : ng(-1) {}
    check_cell_functor(int ng) : ng(ng) {}

    __host__ __device__
    bool operator()(const Cell3& a) const {
      if (a[0] < 0 || a[0] >= ng ) return false;
      if (a[1] < 0 || a[1] >= ng ) return false;
      if (a[2] < 0 || a[2] >= ng ) return false;
      return true;
    }
  };

  // clip is needed because roundoff errors may cause a number to be slightly outside the legal range
  struct clip_cell_functor
  {
    int ng;

    clip_cell_functor() : ng(-1) {}
    clip_cell_functor(int ng) : ng(ng) {}

    __host__ __device__
    void operator()(Cell3& a) {
      if (a[0] < 0) a[0] = 0;
      if (a[0] >= ng) a[0] = ng-1;
      if (a[1] < 0) a[1] = 0;
      if (a[1] >= ng) a[1] = ng-1;
      if (a[2] < 0) a[2] = 0;
      if (a[2] >= ng) a[2] = ng-1;
    }
  };

  template<typename Coord_Tuple>
  struct cell_containing_point_functor : public thrust::unary_function<Coord_Tuple, Cell3>
  {
    double r_cell;
    double d0;
    double d1;
    double d2;

    cell_containing_point_functor() : r_cell(-1), d0(-1), d1(-1), d2(-1) {}

    cell_containing_point_functor(double r_cell, double d0, double d1, double d2)
      : r_cell(r_cell), d0(d0), d1(d1), d2(d2) {}

    __host__ __device__
    Cell3 operator()(const Coord_Tuple& a) const {
      int ix = static_cast<short int>(static_cast<double>(thrust::get<0>(a))*r_cell+d0);
      int iy = static_cast<short int>(static_cast<double>(thrust::get<1>(a))*r_cell+d1);
      int iz = static_cast<short int>(static_cast<double>(thrust::get<2>(a))*r_cell+d2);
      Cell3 c(ix, iy, iz);
      return c;
    }
  };

  struct cell_to_id_functor : public thrust::unary_function<Cell3, int>
  {
    int ng;

    cell_to_id_functor() : ng(-1) {}
    cell_to_id_functor(int ng) : ng(ng) {}
  
    __host__ __device__
    int operator()(const Cell3& c) const {
      if (c[0]<0 || c[0] >=ng || c[1]<0 || c[1] >=ng || c[2]<0 || c[2] >=ng) return -1;
      return (static_cast<int> (c[0])*ng + static_cast<int>(c[1]))*ng + c[2];
    }
  };
  
  template<typename Coord_Tuple>
  struct point_to_id_functor : public thrust::unary_function<Coord_Tuple, int>
  {
    cell_containing_point_functor<Coord_Tuple> cell_containing_point;
    cell_to_id_functor cell_to_id;

    point_to_id_functor() : cell_containing_point(cell_containing_point_functor<Coord_Tuple>()),
                            cell_to_id(cell_to_id_functor()) {}

    point_to_id_functor(cell_containing_point_functor<Coord_Tuple> cell_containing_point,
                        cell_to_id_functor cell_to_id)
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

  template<typename Coord_Tuple>
  struct distance2_functor : public thrust::unary_function<Coord_Tuple, double>
  {
    const Coord_Tuple q;

    __host__ __device__
    distance2_functor(Coord_Tuple q) : q(q) { }

    __host__ __device__
    double square(const double t) const {
      return t * t;
    }
  
    __host__ __device__
    double operator()(const Coord_Tuple& a) const {
      return (square((thrust::get<0>(a)-thrust::get<0>(q))) +
              square((thrust::get<1>(a)-thrust::get<1>(q))) +
              square((thrust::get<2>(a)-thrust::get<2>(q))));
    }
  };

  template<typename Coord_T>
  struct query_cell_functor
  {
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord_Tuple;
    typedef thrust::device_ptr<Coord_T> Coord_Ptr;
    typedef thrust::tuple<Coord_Ptr, Coord_Ptr, Coord_Ptr> Coord_Ptr_Tuple;
    
    num_points_in_cell_id_functor num_points_in_cell_id;
    point_to_id_functor<Coord_Tuple> point_to_id;
    thrust::device_ptr<int> base;
    thrust::device_ptr<int> cells;
    Coord_Ptr_Tuple pts;

    query_cell_functor() : num_points_in_cell_id(num_points_in_cell_id_functor()),
                           point_to_id(point_to_id_functor<Coord_Tuple>()),
                           base(thrust::device_ptr<int>()),
                           cells(thrust::device_ptr<int>()),
                           pts(thrust::make_tuple(Coord_Ptr(), Coord_Ptr(), Coord_Ptr())) {}
    
    query_cell_functor(num_points_in_cell_id_functor num_points_in_cell_id,
                       point_to_id_functor<Coord_Tuple> point_to_id,
                       thrust::device_ptr<int> base,
                       thrust::device_ptr<int> cells,
                       Coord_Ptr_Tuple pts)
      : num_points_in_cell_id(num_points_in_cell_id), point_to_id(point_to_id),
        base(base), cells(cells), pts(pts) {}

    __host__ __device__
    Coord_Tuple point_at(int i) {
      return thrust::make_tuple(thrust::get<0>(pts)[i],
                                thrust::get<1>(pts)[i],
                                thrust::get<2>(pts)[i]);
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
      distance2_functor<Coord_Tuple> distance2(q);
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

  template<typename Coord_T>
  struct fast_query_functor : public thrust::unary_function<thrust::tuple<Coord_T, Coord_T, Coord_T>, int>
  {
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord_Tuple;
    typedef thrust::device_ptr<Coord_T> Coord_Ptr;
    typedef thrust::tuple<Coord_Ptr, Coord_Ptr, Coord_Ptr> Coord_Ptr_Tuple;
    
    clip_cell_functor clip_cell;
    cell_containing_point_functor<Coord_Tuple> cell_containing_point;
    query_cell_functor<Coord_T> query_cell;

    fast_query_functor() : clip_cell(clip_cell_functor()),
                           cell_containing_point(cell_containing_point_functor<Coord_Tuple>()),
                           query_cell(query_cell_functor<Coord_T>()) {}

    fast_query_functor(clip_cell_functor clip_cell,
                       cell_containing_point_functor<Coord_Tuple> cell_containing_point,
                       query_cell_functor<Coord_T> query_cell)
      : clip_cell(clip_cell), cell_containing_point(cell_containing_point), query_cell(query_cell) {}

    __host__ __device__
    Coord_Tuple point_at(int i) {
      Coord_Ptr_Tuple pts(query_cell.pts);
      return thrust::make_tuple(thrust::get<0>(pts)[i],
                                thrust::get<1>(pts)[i],
                                thrust::get<2>(pts)[i]);
    }
    
    __host__ __device__
    int operator()(const Coord_Tuple& q) {
      int queryint = query_cell.point_to_id(q);
      int closestpt = -1;
      double dist2 = -1;
      query_cell(queryint, q, closestpt, dist2);
      const double distf = sqrt(dist2) * 1.00001;
      Coord_Tuple lopt(thrust::make_tuple(clamp_USI(static_cast<double>(thrust::get<0>(q)) - distf),
                                          clamp_USI(static_cast<double>(thrust::get<1>(q)) - distf),
                                          clamp_USI(static_cast<double>(thrust::get<2>(q)) - distf)));
      Coord_Tuple hipt(thrust::make_tuple(clamp_USI(static_cast<double>(thrust::get<0>(q)) + distf + 1.0),
                                          clamp_USI(static_cast<double>(thrust::get<1>(q)) + distf + 1.0),
                                          clamp_USI(static_cast<double>(thrust::get<2>(q)) + distf + 1.0)));
      
      Cell3 locell(cell_containing_point(lopt));
      Cell3 hicell(cell_containing_point(hipt));

      clip_cell(locell);
      clip_cell(hicell);

      Cell3 qcell(cell_containing_point(q));
      if (locell == qcell && hicell == qcell) {
        return closestpt;
      }
      int close2 = -1;
      double d2 = -1;
      for (Coord_T x=locell[0]; x<=hicell[0]; ++x) {
        for (Coord_T y=locell[1]; y<=hicell[1]; ++y) {
          for (Coord_T z=locell[2]; z<=hicell[2]; ++z) {
            queryint = query_cell.point_to_id.cell_to_id(Cell3(x, y, z));
            query_cell(queryint, q, close2, d2);
            if (close2 != -1 && (d2 < dist2 || (d2 == dist2 && close2 < closestpt))) {
              closestpt = close2;
              dist2 = d2;
            }
          }
        }
      }
      return closestpt;
    }
  };

  template<typename Coord_T>
  struct slow_query_functor : public thrust::unary_function<thrust::tuple<Coord_T, Coord_T, Coord_T>, int>
  {
    typedef thrust::tuple<Coord_T, Coord_T, Coord_T> Coord_Tuple;
    typedef thrust::device_ptr<Coord_T> Coord_Ptr;
    typedef thrust::tuple<Coord_Ptr, Coord_Ptr, Coord_Ptr> Coord_Ptr_Tuple;

    int ncellsearch;
    thrust::device_ptr<int> cellsearch;

    check_cell_functor check_cell;
    cell_containing_point_functor<Coord_Tuple> cell_containing_point;
    query_cell_functor<Coord_T> query_cell;

    slow_query_functor() : ncellsearch(0),
                           cellsearch(thrust::device_ptr<int>()),
                           check_cell(check_cell_functor()),
                           cell_containing_point(cell_containing_point_functor<Coord_Tuple>()),
                           query_cell(query_cell_functor<Coord_T>()) {}

    slow_query_functor(int ncellsearch,
                       thrust::device_ptr<int> cellsearch,
                       check_cell_functor check_cell,
                       cell_containing_point_functor<Coord_Tuple> cell_containing_point,
                       query_cell_functor<Coord_T> query_cell)
      : ncellsearch(ncellsearch), cellsearch(cellsearch), check_cell(check_cell),
        cell_containing_point(cell_containing_point), query_cell(query_cell) {}

    
    __host__ __device__
    Coord_Tuple point_at(int i) {
      Coord_Ptr_Tuple pts(query_cell.pts);
      return thrust::make_tuple(thrust::get<0>(pts)[i],
                                thrust::get<1>(pts)[i],
                                thrust::get<2>(pts)[i]);
    }
    
    __host__ __device__
    int operator()(const Coord_Tuple& q) {
      const int sign3[8][3] = {{1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},
                               {-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}};
      const int perm3[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};

      Cell3 qcell = cell_containing_point(q);
      int queryint = query_cell.point_to_id.cell_to_id(qcell);
      int closestpt = -1;
      double dist2 = -1;

      int nstop(ncellsearch);
      bool found(false);

      for (int isort=0; isort<nstop; ++isort) {
        int close2;
        double d2;
        Cell3 s (cellsearch[isort*4], cellsearch[isort*4+1], cellsearch[isort*4+2]);

        for (int isign=0; isign<8; ++isign) {
          if (s[0]==0 && sign3[isign][0]== -1) continue;
          if (s[1]==0 && sign3[isign][1]== -1) continue;
          if (s[2]==0 && sign3[isign][2]== -1) continue;

          const Cell3 s2(s*sign3[isign]);

          for (int iperm=0; iperm<6; ++iperm) {
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
            const Cell3 s3(s2[perm3[iperm][0]], s2[perm3[iperm][1]], s2[perm3[iperm][2]]);
            const Cell3 c2(qcell+s3);
            if (!check_cell(c2)) continue;
            int cell_id(query_cell.point_to_id.cell_to_id(c2));
            query_cell(cell_id, q, close2, d2);
            if (close2 < 0) continue;

            if (dist2 == -1 || d2 < dist2 || (d2 == dist2 && close2 < closestpt)) {
              dist2 = d2;
              closestpt = close2;
              if (!found) {
                found = true;
                nstop = cellsearch[isort*4+3];
                if (nstop >= ncellsearch) {
                  iperm = 6;
                  isign = 8;
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
  

  template<typename T>
  struct greater_functor : public thrust::unary_function<T, bool>
  {
    const int b;

    greater_functor(int b) : b(b) {}

    __host__ __device__
    bool operator()(const T& a) const {
      return thrust::get<0>(a) > b;
    }
  };
};