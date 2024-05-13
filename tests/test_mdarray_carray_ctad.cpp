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
#include <mdspan/mdarray.hpp>
#include <vector>

#include <gtest/gtest.h>
#include "offload_utils.hpp"

namespace KokkosEx = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;

_MDSPAN_INLINE_VARIABLE constexpr auto dyn = MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent;


template<class ValueType, size_t Extent>
_MDSPAN_HOST_DEVICE
void fill_values(ValueType values[Extent])
{
  for (std::size_t k = 0; k < Extent; ++k) {
    values[k] = static_cast<ValueType>(1) + static_cast<ValueType>(k);
  }
}

struct ErrorBufferDeleter {
  void operator() (std::size_t* ptr) const {
    free_array(ptr);
  }
};

std::unique_ptr<std::size_t, ErrorBufferDeleter> allocate_error_buffer() {
  return {allocate_array<std::size_t>(1u), ErrorBufferDeleter{}};
}

template<class ValueType, size_t Extent>
void test_mdarray_ctad_carray_rank1() {
  using MDSPAN_IMPL_STANDARD_NAMESPACE::layout_right;

  auto error_buffer = allocate_error_buffer();
  {
    size_t* errors = error_buffer.get();
    errors[0] = 0;
    dispatch([errors] _MDSPAN_HOST_DEVICE () {
      ValueType values[Extent];
      fill_values<ValueType, Extent>(values);
    
      KokkosEx::mdarray m{values};
      static_assert(std::is_same_v<typename decltype(m)::layout_type,
                    layout_right>);
      static_assert(std::is_same_v<typename decltype(m)::container_type,
                    std::array<ValueType, Extent>>);

      __MDSPAN_DEVICE_ASSERT_EQ(m.rank(), 1);
      __MDSPAN_DEVICE_ASSERT_EQ(m.rank_dynamic(), 0);
      __MDSPAN_DEVICE_ASSERT_EQ(m.extent(0), Extent);
      __MDSPAN_DEVICE_ASSERT_EQ(m.static_extent(0), Extent);
    });
    ASSERT_EQ(errors[0], 0);
  }
}

TEST(TestMdarrayCtorDataCArray, test_mdarray_carray_ctad) {
  __MDSPAN_TESTS_RUN_TEST((test_mdarray_ctad_carray_rank1<float, 5>()))
}

