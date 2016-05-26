#pragma once

#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace nearptd {

  // Typedef for tuple of size N and type T
  template<typename T, size_t N>
  struct ntuple {};

  template<typename T>
  struct ntuple<T, 1> {
    typedef thrust::tuple<T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a);
    }
  };

  template<typename T>
  struct ntuple<T, 2> {
    typedef thrust::tuple<T, T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0],a[1]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a); b[1] = thrust::get<1>(a);
    }
  };
  
  template<typename T>
  struct ntuple<T, 3> {
    typedef thrust::tuple<T, T, T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0],a[1],a[2]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a); b[1] = thrust::get<1>(a); b[2] = thrust::get<2>(a);
    }
  };

  template<typename T>
  struct ntuple<T, 4> {
    typedef thrust::tuple<T, T, T, T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0],a[1],a[2],a[3]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a); b[1] = thrust::get<1>(a); b[2] = thrust::get<2>(a);
      b[3] = thrust::get<3>(a);
    }
  };

  template<typename T>
  struct ntuple<T, 5> {
    typedef thrust::tuple<T, T, T, T, T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0],a[1],a[2],a[3],a[4]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a); b[1] = thrust::get<1>(a); b[2] = thrust::get<2>(a);
      b[3] = thrust::get<3>(a); b[4] = thrust::get<4>(a);
    }
  };

  template<typename T>
  struct ntuple<T, 6> {
    typedef thrust::tuple<T, T, T, T, T, T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0],a[1],a[2],a[3],a[4],a[5]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a); b[1] = thrust::get<1>(a); b[2] = thrust::get<2>(a);
      b[3] = thrust::get<3>(a); b[4] = thrust::get<4>(a); b[5] = thrust::get<5>(a);
    }
  };

  template<typename T>
  struct ntuple<T, 7> {
    typedef thrust::tuple<T, T, T, T, T, T, T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0],a[1],a[2],a[3],a[4],a[5],a[6]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a); b[1] = thrust::get<1>(a); b[2] = thrust::get<2>(a);
      b[3] = thrust::get<3>(a); b[4] = thrust::get<4>(a); b[5] = thrust::get<5>(a);
      b[6] = thrust::get<6>(a);
    }
  };

  template<typename T>
  struct ntuple<T, 8> {
    typedef thrust::tuple<T, T, T, T, T, T, T, T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a); b[1] = thrust::get<1>(a); b[2] = thrust::get<2>(a);
      b[3] = thrust::get<3>(a); b[4] = thrust::get<4>(a); b[5] = thrust::get<5>(a);
      b[6] = thrust::get<6>(a); b[7] = thrust::get<7>(a);
    }
  };

  template<typename T>
  struct ntuple<T, 9> {
    typedef thrust::tuple<T, T, T, T, T, T, T, T, T> tuple;
    __host__ __device__
    tuple make_tuple(T* a) {return thrust::make_tuple(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8]);}
    __host__ __device__
    void make_array(const tuple& a, T* b) {
      b[0] = thrust::get<0>(a); b[1] = thrust::get<1>(a); b[2] = thrust::get<2>(a);
      b[3] = thrust::get<3>(a); b[4] = thrust::get<4>(a); b[5] = thrust::get<5>(a);
      b[6] = thrust::get<6>(a); b[7] = thrust::get<7>(a); b[8] = thrust::get<8>(a);
    }
  };

  // Apply function across a tuple of T1s, returning a new tuple of T2s
  template<typename T1, typename T2, typename UnFunc, size_t N>
  struct tuple_unary_apply {};

  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 1> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)));
    }
  };
  
  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 2> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)),func(thrust::get<1>(a)));
    }
  };
  
  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 3> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)),func(thrust::get<1>(a)),
                                func(thrust::get<2>(a)));
    }
  };
  
  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 4> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)),func(thrust::get<1>(a)),
                                func(thrust::get<2>(a)),func(thrust::get<3>(a)));
    }
  };
  
  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 5> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)),func(thrust::get<1>(a)),
                                func(thrust::get<2>(a)),func(thrust::get<3>(a)),
                                func(thrust::get<4>(a)));
    }
  };
            
  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 6> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)),func(thrust::get<1>(a)),
                                func(thrust::get<2>(a)),func(thrust::get<3>(a)),
                                func(thrust::get<4>(a)),func(thrust::get<5>(a)));
    }
  };
  
  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 7> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)),func(thrust::get<1>(a)),
                                func(thrust::get<2>(a)),func(thrust::get<3>(a)),
                                func(thrust::get<4>(a)),func(thrust::get<5>(a)),
                                func(thrust::get<6>(a)));
    }
  };
  
  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 8> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)),func(thrust::get<1>(a)),
                                func(thrust::get<2>(a)),func(thrust::get<3>(a)),
                                func(thrust::get<4>(a)),func(thrust::get<5>(a)),
                                func(thrust::get<6>(a)),func(thrust::get<7>(a)));
    }
  };
  
  template<typename T1, typename T2, typename UnFunc>
  struct tuple_unary_apply<T1, T2, UnFunc, 9> {
    __host__ __device__
    T2 operator()(const T1&a, UnFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a)),func(thrust::get<1>(a)),
                                func(thrust::get<2>(a)),func(thrust::get<3>(a)),
                                func(thrust::get<4>(a)),func(thrust::get<5>(a)),
                                func(thrust::get<6>(a)),func(thrust::get<7>(a)),
                                func(thrust::get<8>(a)));
    }
  };

  // Apply function across two tuples of T1s and T2s, returning a tuple of T3s
  template<typename T1, typename T2, typename T3, typename BinFunc, size_t N>
  struct tuple_binary_apply {};
  
  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 1> {
    __host__ __device__
    T3 operator()(const T1& a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)));
    }
  };
  
  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 2> {
    __host__ __device__
    T3 operator()(const T1&a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 3> {
    __host__ __device__
    T3 operator()(const T1&a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 4> {
    __host__ __device__
    T3 operator()(const T1&a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 5> {
    __host__ __device__
    T3 operator()(const T1&a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)),
                                func(thrust::get<4>(a), thrust::get<4>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 6> {
    __host__ __device__
    T3 operator()(const T1&a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)),
                                func(thrust::get<4>(a), thrust::get<4>(b)),
                                func(thrust::get<5>(a), thrust::get<5>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 7> {
    __host__ __device__
    T3 operator()(const T1&a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)),
                                func(thrust::get<4>(a), thrust::get<4>(b)),
                                func(thrust::get<5>(a), thrust::get<5>(b)),
                                func(thrust::get<6>(a), thrust::get<6>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 8> {
    __host__ __device__
    T3 operator()(const T1&a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)),
                                func(thrust::get<4>(a), thrust::get<4>(b)),
                                func(thrust::get<5>(a), thrust::get<5>(b)),
                                func(thrust::get<6>(a), thrust::get<6>(b)),
                                func(thrust::get<7>(a), thrust::get<7>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct tuple_binary_apply<T1, T2, T3, BinFunc, 9> {
    __host__ __device__
    T3 operator()(const T1&a, const T2& b, BinFunc func) const {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)),
                                func(thrust::get<4>(a), thrust::get<4>(b)),
                                func(thrust::get<5>(a), thrust::get<5>(b)),
                                func(thrust::get<6>(a), thrust::get<6>(b)),
                                func(thrust::get<7>(a), thrust::get<7>(b)),
                                func(thrust::get<8>(a), thrust::get<8>(b)));
    }
  };

  // Apply function across a tuple of T1s as a reduction, returning a single T2 value
  template<typename T1, typename T2, typename Function, size_t N>
  struct tuple_reduce {
    __host__ __device__
    T2 operator()(const T1& a, Function func) {
      tuple_reduce<T1, T2, Function, N-1> op;
      return func(thrust::get<N-1>(a), op(a, func));
    }
  };

  template<typename T1, typename T2, typename Function>
  struct tuple_reduce<T1, T2, Function, 1> {
    __host__ __device__
    T2 operator()(const T1& a, Function func) {
      return static_cast<T2>(thrust::get<0>(a));
    }
  };  
};
