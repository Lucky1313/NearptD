#include <boost/multi_array.hpp>
#include <iostream>

using boost::array;

namespace nearpt3 {

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

  void write(std::ostream &o, const Cell3& c) {
    o << '(' << c[0] << ',' << c[1] << ',' << c[2] << ") ";
  }
};
