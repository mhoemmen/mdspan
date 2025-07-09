//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#include <mdspan/mdspan.hpp>
#include <gtest/gtest.h>

#if ! defined(MDSPAN_ENABLE_P3663)
#  error "This test requires that MDSPAN_ENABLE_P3663 be defined."
#endif

struct some_struct {};

template<class T>
constexpr bool my_is_constant_wrapper = false;

//template<auto Value, class Type>
//constexpr bool my_is_constant_wrapper<
//  ::std::constant_wrapper<Value, Type>> = true;

template<auto Value>
constexpr bool my_is_constant_wrapper<
  ::std::constant_wrapper<Value>> = true;

template<class Type, Type Value>
constexpr bool my_is_constant_wrapper<
    ::std::constant_wrapper<
      ::std::exposition_only::cw_fixed_value<Type>{Value},
      Type
    >
  > = true;

template<class T>
struct printer {};

TEST(IsConstantWrapper, Test0) {
  using ::MDSPAN_IMPL_STANDARD_NAMESPACE::detail::is_constant_wrapper;

  static_assert(! is_constant_wrapper<int>);
  static_assert(! is_constant_wrapper<size_t>);
  static_assert(! is_constant_wrapper<some_struct>);

  static_assert(! my_is_constant_wrapper<int>);
  static_assert(! my_is_constant_wrapper<size_t>);
  static_assert(! my_is_constant_wrapper<some_struct>);

  [[maybe_unused]] auto forty_two = ::std::cw<42>;
  //static_assert(is_constant_wrapper< decltype(forty_two) >);
  static_assert(my_is_constant_wrapper< decltype(forty_two) >);

  //using type = printer<decltype(forty_two)>::type;

  static_assert(my_is_constant_wrapper<
      ::std::constant_wrapper<
        ::std::exposition_only::cw_fixed_value<int>{42}, int
      >
    >);

#if 0
  [[maybe_unused]] auto forty_two_a = ::std::constant_wrapper<42>{};
  static_assert(is_constant_wrapper< decltype(forty_two_a) >);
  static_assert(my_is_constant_wrapper< decltype(forty_two_a) >);

  [[maybe_unused]] auto forty_two_b = ::std::constant_wrapper<42, int>{};
  static_assert(is_constant_wrapper< decltype(forty_two_b) >);
  static_assert(my_is_constant_wrapper< decltype(forty_two_b) >);

  [[maybe_unused]] auto forty_two_c = ::std::constant_wrapper<
    ::std::exposition_only::cw_fixed_value<int>(42), int>{};
  static_assert(is_constant_wrapper< decltype(forty_two_c) >);
  static_assert(my_is_constant_wrapper< decltype(forty_two_c) >);

  //static_assert(! is_constant_wrapper< decltype(::std::cw<size_t(42)>) >);
#endif // 0
}
