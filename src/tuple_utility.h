#pragma once

#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace nearpt3 {

  // Typedef for tuple of size N and type T
  template<typename T, size_t N>
  struct ntuple {};

  template<typename T>
  struct ntuple<T, 1> {
    typedef thrust::tuple<T> tuple;  
    tuple make(T* a) { return thrust::make_tuple(a[0]);}
  };

  template<typename T>
  struct ntuple<T, 2> {
    typedef thrust::tuple<T, T> tuple;
    tuple make(T* a) { return thrust::make_tuple(a[0], a[1]);}
  };
  
  template<typename T>
  struct ntuple<T, 3> {
    typedef thrust::tuple<T, T, T> tuple; 
    tuple make(T* a) { return thrust::make_tuple(a[0], a[1], a[2]);}
  };

  template<typename T>
  struct ntuple<T, 4> {
    typedef thrust::tuple<T, T, T, T> tuple;
    tuple make(T* a) { return thrust::make_tuple(a[0], a[1], a[2], a[3]);}
  };

  template<typename T>
  struct ntuple<T, 5> {
    typedef thrust::tuple<T, T, T, T, T> tuple;
    tuple make(T* a) { return thrust::make_tuple(a[0], a[1], a[2], a[3], a[4]);}
  };

  template<typename T>
  struct ntuple<T, 6> {
    typedef thrust::tuple<T, T, T, T, T, T> tuple;
    tuple make(T* a) { return thrust::make_tuple(a[0], a[1], a[2], a[3], a[4], a[5]);}
  };

  template<typename T>
  struct ntuple<T, 7> {
    typedef thrust::tuple<T, T, T, T, T, T, T> tuple;
    tuple make(T* a) { return thrust::make_tuple(a[0], a[1], a[2], a[3], a[4], a[5], a[6]);}
  };

  template<typename T>
  struct ntuple<T, 8> {
    typedef thrust::tuple<T, T, T, T, T, T, T, T> tuple;
    tuple make(T* a) { return thrust::make_tuple(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);}
  };

  template<typename T>
  struct ntuple<T, 9> {
    typedef thrust::tuple<T, T, T, T, T, T, T, T, T> tuple;
    tuple make(T* a) { return thrust::make_tuple(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);}
  };
  
  
  template<typename T1, typename T2, typename T3, typename BinFunc, size_t N>
  struct transform_functor {};
  
  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct transform_functor<T1, T2, T3, BinFunc, 1> {
    T3 operator()(const T1& a, const T2& b, BinFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)));
    }
  };
  
  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct transform_functor<T1, T2, T3, BinFunc, 2> {
    T3 operator()(const T1&a, const T2& b, BinFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct transform_functor<T1, T2, T3, BinFunc, 3> {
    T3 operator()(const T1&a, const T2& b, BinFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct transform_functor<T1, T2, T3, BinFunc, 4> {
    T3 operator()(const T1&a, const T2& b, BinFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct transform_functor<T1, T2, T3, BinFunc, 5> {
    T3 operator()(const T1&a, const T2& b, BinFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)),
                                func(thrust::get<4>(a), thrust::get<4>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct transform_functor<T1, T2, T3, BinFunc, 6> {
    T3 operator()(const T1&a, const T2& b, BinFunc func) {
      return thrust::make_tuple(func(thrust::get<0>(a), thrust::get<0>(b)),
                                func(thrust::get<1>(a), thrust::get<1>(b)),
                                func(thrust::get<2>(a), thrust::get<2>(b)),
                                func(thrust::get<3>(a), thrust::get<3>(b)),
                                func(thrust::get<4>(a), thrust::get<4>(b)),
                                func(thrust::get<5>(a), thrust::get<5>(b)));
    }
  };

  template<typename T1, typename T2, typename T3, typename BinFunc>
  struct transform_functor<T1, T2, T3, BinFunc, 7> {
    T3 operator()(const T1&a, const T2& b, BinFunc func) {
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
  struct transform_functor<T1, T2, T3, BinFunc, 8> {
    T3 operator()(const T1&a, const T2& b, BinFunc func) {
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
  struct transform_functor<T1, T2, T3, BinFunc, 9> {
    T3 operator()(const T1&a, const T2& b, BinFunc func) {
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

  template<typename T1, typename T2, typename Function, size_t N>
  struct reduce_functor {
    T2 operator()(const T1& a, Function func) {
      reduce_functor<T1, T2, Function, N-1> op;
      return func(thrust::get<N-1>(a), op(a, func));
    }
  };

  template<typename T1, typename T2, typename Function>
  struct reduce_functor<T1, T2, Function, 1> {
    T2 operator()(const T1& a, Function func) {
      return thrust::get<0>(a);
    }
  };  
};
