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
#pragma once

#include <mdspan/mdspan.hpp>
#include <benchmark/benchmark.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>

namespace submdspan_benchmark {

#if defined(MDSPAN_CONSTANT_WRAPPER_WORKAROUND)

template<class ElementType, class Extents, class Layout, class Accessor, size_t... Indices>
constexpr typename Kokkos::mdspan<ElementType, Extents, Layout, Accessor>::reference
get_broadcast_element_impl(
  const Kokkos::mdspan<ElementType, Extents, Layout, Accessor>& x,
  typename Extents::index_type broadcast_index,
  std::index_sequence<Indices...>)
{
#if defined(MDSPAN_USE_BRACKET_OPERATOR) && (MDSPAN_USE_BRACKET_OPERATOR != 0)
  return x[((void) Indices, 0)...];
#else
  return x(((void) Indices, 0)...);
#endif
}

template<class ElementType, class Extents, class Layout, class Accessor>
constexpr typename Kokkos::mdspan<ElementType, Extents, Layout, Accessor>::reference
get_broadcast_element(
  const Kokkos::mdspan<ElementType, Extents, Layout, Accessor>& x,
  typename Extents::index_type broadcast_index)
{
  return get_broadcast_element_impl(x, broadcast_index, std::make_index_sequence<Extents::rank()>());
}

#else

template<class ElementType, class IndexType, size_t... Exts, class Layout, class Accessor>
constexpr typename Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, Exts...>, Layout, Accessor>::reference
get_broadcast_element(
  const Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, Exts...>, Layout, Accessor>& x,
  typename Kokkos::extents<IndexType, Exts...>::index_type broadcast_index)
{
#if defined(MDSPAN_USE_PAREN_OPERATOR) && (MDSPAN_USE_PAREN_OPERATOR != 0)
  return x(((void) Exts, broadcast_index)...);
#else
  return x[((void) Exts, broadcast_index)...];  
#endif
}

#endif

template<class IndexType, size_t... Exts>
using nonconst_test_mdspan =
  Kokkos::mdspan<std::uint8_t, Kokkos::extents<IndexType, Exts...>>;

template<class IndexType, size_t... Exts>
using const_test_mdspan =
  Kokkos::mdspan<const std::uint8_t, Kokkos::extents<IndexType, Exts...>>;

class random_state_t {
public:
  using seed_type = std::mt19937::result_type;

  random_state_t() : gen_(default_seed) {}
  random_state_t(seed_type seed) : gen_(seed) {}

  std::mt19937& generator() noexcept { return gen_; }

private:
  static constexpr seed_type default_seed = 1234u;
  std::mt19937 gen_;
};

template<class ExecutionSpace, class ValueType>
struct array_deleter {};

template<class ExecutionSpace, class ValueType>
using array_deleter_t = typename array_deleter<ExecutionSpace, ValueType>::type;

struct host_execution_space {};

template<class ValueType>
struct array_deleter<host_execution_space, ValueType> {
  using type = std::default_delete<ValueType[]>;
};

template<class ValueType>
std::unique_ptr<ValueType[], array_deleter_t<host_execution_space, ValueType>>
allocate_buffer(host_execution_space, size_t num_elements) {
  return std::make_unique<ValueType[]>(num_elements);
}

template<class ValueType>
requires(std::is_trivially_copyable_v<ValueType>)
void
copy_buffer(host_execution_space, const ValueType in[], ValueType out[], std::size_t num_elements)
{
  (void) std::memcpy(out, in, num_elements * sizeof(ValueType));
}

template<class ExecutionSpace, class IndexType, size_t... Exts>
class benchmark_buffer {
private:
  static constexpr bool is_host =
    std::is_same_v<ExecutionSpace, host_execution_space>;
  
public:
  using value_type = std::uint8_t;

  benchmark_buffer(ExecutionSpace exec_space, Kokkos::extents<IndexType, Exts...> exts) :
    exec_space_{exec_space},
    mapping_{exts},
    buffer_{allocate_buffer<value_type>(exec_space, mapping_.required_span_size())}
  {
    if constexpr (! is_host) {
      host_buffer_ = allocate_buffer<value_type>(
        host_execution_space{}, mapping_.required_span_size());
    }
  }

  friend void copy(const benchmark_buffer& src, benchmark_buffer& dst) {
    if (&dst != &src) {
      if (src.size() != dst.size()) {
        throw std::runtime_error("benchmark_buffer::copy: src.size() != dst.size()");
      }
      if constexpr (! is_host) {
        copy_buffer(dst.exec_space_, src.host_buffer_.get(), dst.host_buffer_.get(), dst.size());
      }
      copy_buffer(dst.exec_space_, src.buffer_.get(), dst.buffer_.get(), dst.size());
    }
  }

  benchmark_buffer(const benchmark_buffer& rhs) :
    exec_space_{rhs.exec_space_},
    mapping_{rhs.mapping_},
    buffer_{allocate_buffer<value_type>(rhs.exec_space_, mapping_.required_span_size())}
  {
    if constexpr (! is_host) {
      host_buffer_ = allocate_buffer<value_type>(
        host_execution_space{}, mapping_.required_span_size());
    }
    copy(rhs, *this);
  }

  benchmark_buffer& operator=(const benchmark_buffer& rhs) {
    if (this != &rhs) {
      exec_space_ = rhs.exec_space_;
      mapping_ = rhs.mapping_;
      buffer_ = allocate_buffer<value_type>(rhs.exec_space_, mapping_.required_span_size());

      if constexpr (! is_host) {
        host_buffer_ = allocate_buffer<value_type>(
          host_execution_space{}, mapping_.required_span_size());
      }
      copy(rhs, *this);
    }
    return *this;
  }
  
  size_t size() const {
    return mapping_.required_span_size();
  }

  nonconst_test_mdspan<IndexType, Exts...> get_mdspan() {
    return {buffer_.get(), mapping_};
  }

  const_test_mdspan<IndexType, Exts...> get_mdspan() const {
    return {static_cast<const value_type*>(buffer_.get()), mapping_};
  }

  void sync_to_device() {
    if constexpr (! is_host) {
      copy_buffer(exec_space_, host_buffer_.get(), buffer_.get(), size());
    }
  }

  void sync_to_host() {
    if constexpr (! is_host) {
      copy_buffer(exec_space_, buffer_.get(), host_buffer_.get(), size());
    }
  }

  nonconst_test_mdspan<IndexType, Exts...> get_host_mdspan() {
    if constexpr (is_host) {
      return {buffer_.get(), mapping_};
    }
    else {
      return {host_buffer_.get(), mapping_};
    }
  }

  const_test_mdspan<IndexType, Exts...> get_host_mdspan() const {
    if constexpr (is_host) {
      return {static_cast<const value_type*>(buffer_.get()), mapping_};
    }
    else {
      return {static_cast<const value_type*>(host_buffer_.get()), mapping_};
    }
  }
  
private:
  ExecutionSpace exec_space_{};
  Kokkos::layout_right::template mapping<Kokkos::extents<IndexType, Exts...>> mapping_;
  std::unique_ptr<value_type[], array_deleter_t<ExecutionSpace, value_type>> buffer_;
  std::unique_ptr<value_type[], array_deleter_t<host_execution_space, value_type>> host_buffer_;
};

template <class ExecutionSpace, class IndexType, size_t... Exts>
void fill_with_random_values(
  ExecutionSpace exec,
  random_state_t& state,
  benchmark_buffer<ExecutionSpace, IndexType, Exts...>& s)
{
  auto val_dist = std::uniform_int_distribution<std::uint8_t>(0u, 255u);
  auto next = [&] () {
    return val_dist(state.generator());
  };
  auto s_host = s.get_host_mdspan();
  std::generate(s_host.data_handle(), s_host.data_handle() + s.size(), next);
  s.sync_to_device();
}

// Index or slice type that's convertible to IndexType,
// but neither integral nor integral-constant-like.
MDSPAN_TEMPLATE_REQUIRES(
  class IndexType,
  /* requires */ (
    std::is_signed_v<IndexType> || std::is_unsigned_v<IndexType>
  )
)
class index_holder {
public:
  constexpr MDSPAN_FUNCTION index_holder(IndexType i) : i_{i} {}
  constexpr MDSPAN_FUNCTION operator IndexType() const noexcept { return i_; }
  constexpr MDSPAN_FUNCTION index_holder& operator++() noexcept {
    ++i_;
    return *this;
  }
#if defined(__cpp_impl_three_way_comparison)
  constexpr /* MDSPAN_FUNCTION */ auto operator<=>(const index_holder&) const noexcept = default;
#else
  friend constexpr MDSPAN_FUNCTION bool operator<(const index_holder& x, const index_holder& y) noexcept {
    return x.i_ < y.i_;
  }
  friend constexpr MDSPAN_FUNCTION bool operator==(const index_holder& x, const index_holder& y) noexcept {
    return x.i_ == y.i_;
  }
#endif

private:
  IndexType i_;
};
static_assert(std::is_convertible_v<index_holder<int>, int>);
static_assert(std::is_convertible_v<index_holder<size_t>, size_t>);
static_assert(std::is_nothrow_constructible_v<int, index_holder<int>>);
static_assert(std::is_nothrow_constructible_v<size_t, index_holder<size_t>>);

// Slice type that's convertible to full_extent_t, but is not full_extent_t.
struct full_extent_wrapper_t {
  constexpr operator Kokkos::full_extent_t() const noexcept{
    return Kokkos::full_extent;
  }
};

template<class ElementType, class Layout, class Accessor, class Slice, class IndexType, size_t... Exts>
constexpr MDSPAN_FUNCTION auto slice_one_extent(
  Kokkos::mdspan<ElementType, Kokkos::extents<IndexType, Exts...>, Layout, Accessor> x, Slice slice)
{
  if constexpr (sizeof...(Exts) == 0) {
#if defined(MDSPAN_CONSTANT_WRAPPER_WORKAROUND)
    static_assert(sizeof...(Exts) != 0, "slice_one_extent called with no extents");
#else
    static_assert(false, "slice_one_extent called with no extents");
#endif
  }
  else if constexpr (sizeof...(Exts) == 1) {
    return Kokkos::submdspan(x, slice);
  }
  else {
    return [&] <size_t... Inds> (std::index_sequence<Inds...>) {
      return Kokkos::submdspan(x, slice, ((void) Inds, full_extent_wrapper_t{})...);
    } (std::make_index_sequence<sizeof...(Exts) - 1u>());
  }
}

// Elements of x are uint8_t, so computations happen modulo 256.
// For each element x_e of x, on output, result is
//
//   (x_e * 3^count) mod 256
// = ((x_e mod 256) * (3^count mod 256)) mod 256.
//
// If count is a power of two, we can compute (3^count) mod 256
// by divide and conquer.
//
//   (3^count) mod 256
// = ((3^(count/2)) mod 256) * ((3^(count/2)) mod 256) mod 256.

constexpr MDSPAN_INLINE_FUNCTION size_t
base_to_the_exponent_mod_modulus(size_t base, size_t exponent, size_t modulus)
{
  if (modulus == 1u) {
    return 0u;
  }
  // modulus - 1u) * (modulus - 1u) must not overflow base
  size_t result = 1u;
  base = base % modulus;
  while (exponent > 0u) {
    if (exponent % 2u == 1u) {
      result = (result * base) % modulus;
    }
    exponent = exponent >> 1u;
    base = (base * base) % modulus;
  }
  return result;
}

constexpr MDSPAN_INLINE_FUNCTION size_t
expected_element(size_t original_element, size_t count) {
  constexpr size_t base = 3u;
  constexpr size_t modulus = 256u;
  return ((original_element % modulus) * base_to_the_exponent_mod_modulus(base, count, modulus)) % modulus;
}

// Multiply elements by 3, using 1-D slices.
template<class ExecutionSpace,
  class IndexType, size_t... Exts,
  class Layout>
MDSPAN_INLINE_FUNCTION void
benchmark2_loop(ExecutionSpace exec_space,
  Kokkos::mdspan<std::uint8_t, Kokkos::extents<IndexType, Exts...>, Layout> out)
{
  using mdspan_type = Kokkos::mdspan<std::uint8_t,
    Kokkos::extents<IndexType, Exts...>, Layout>;

  if constexpr (mdspan_type::rank() == 0) {
    return;
  }
  else if constexpr (mdspan_type::rank() == 1) {
    const IndexType ext0 = out.extent(0);
    for (IndexType k = 0; k < ext0; ++k) {
      out[k] *= 3u;
    }
  }
  else {
    const auto ext0 = index_holder{out.extent(0)};
    for (auto k = index_holder{IndexType(0)}; k < ext0; ++k) {
      benchmark2_loop(exec_space, slice_one_extent(out, k));
    }
  }
}

template<class IndexType, size_t... Exts>
size_t benchmark2_impl(host_execution_space exec_space,
  benchmark::State& state,
  nonconst_test_mdspan<IndexType, Exts...> out,
  size_t inner_count)
{
  size_t count = 0;
  for (auto _ : state) {
    for (size_t c = 0; c < inner_count; ++c) {
      benchmark2_loop(exec_space, out);
    }
    count += inner_count;
  }
  benchmark::DoNotOptimize(count);
  return count;
}

template<class ExecutionSpace, class IndexType, size_t... Exts>
void benchmark2(ExecutionSpace exec_space,
  benchmark::State& state,
  Kokkos::extents<IndexType, Exts...> exts,
  size_t inner_count = 100u)
{
  auto in_buf = benchmark_buffer{exec_space, exts};
  random_state_t random_state{};
  fill_with_random_values(exec_space, random_state, in_buf);
  auto out_buf = benchmark_buffer{in_buf}; // deep copy

  const size_t count = benchmark2_impl(exec_space, state, out_buf.get_mdspan(), inner_count);
  {
    in_buf.sync_to_host();
    out_buf.sync_to_host();
    auto in = in_buf.get_host_mdspan().data_handle();
    auto out = out_buf.get_host_mdspan().data_handle();
    const size_t num_elements = in_buf.size();
    for (size_t i = 0; i < num_elements; ++i) {
      const auto original = in[i];
      const auto expected = expected_element(original, count);
      if (out[i] != expected) {
        std::ostringstream os;
        os << "benchmark2 failed: out[" << i << "] = "
           << static_cast<unsigned>(out[i]) << " != "
           << expected << "\n";
        throw std::runtime_error(os.str());
      }
    }
  }
}

} // namespace submdspan_benchmark
