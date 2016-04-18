#pragma once

#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace nearpt3 {

  // Typedef for tuple of size N and type T
  template<typename T, size_t N>
  struct ntuple {};

  template<typename T>
  struct ntuple<T, 1> { typedef thrust::tuple<T> tuple;};

  template<typename T>
  struct ntuple<T, 2> { typedef thrust::tuple<T, T> tuple;};
  
  template<typename T>
  struct ntuple<T, 3> { typedef thrust::tuple<T, T, T> tuple;};

  template<typename T>
  struct ntuple<T, 4> { typedef thrust::tuple<T, T, T, T> tuple;};

  template<typename T>
  struct ntuple<T, 5> { typedef thrust::tuple<T, T, T, T, T> tuple;};

  template<typename T>
  struct ntuple<T, 6> { typedef thrust::tuple<T, T, T, T, T, T> tuple;};

  template<typename T>
  struct ntuple<T, 7> { typedef thrust::tuple<T, T, T, T, T, T, T> tuple;};

  template<typename T>
  struct ntuple<T, 8> { typedef thrust::tuple<T, T, T, T, T, T, T, T> tuple;};

  template<typename T>
  struct ntuple<T, 9> { typedef thrust::tuple<T, T, T, T, T, T, T, T, T> tuple;};
  
  
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

  template<typename T1, typename T2, typename Function, size_t N>
  struct reduce_functor {
    T2 operator()(const T1& a, Function func) {
      reduce_functor<T1, T2, Function, N-1> op;
      return func(thrust::get<N>(a), op(a, func));
    }
  };

  template<typename T1, typename T2, typename Function>
  struct reduce_functor<T1, T2, Function, 0> {
    T2 operator()(const T1& a, Function func) {
      return thrust::get<0>(a);
    }
  };  
};
