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

#include "submdspan_generic.hpp"

// This benchmark measures the overhead of submdspan slice
// canonicalization as proposed by P3663R2.
//
// Slice canonicalization happens in the submdspan function,
// before slices reach the layout mapping's submdspan_mapping
// customization.  Thus, we need to call submdspan itself,
// but the layout mapping type does not matter.
// We do want to exercise a Standard layout mapping, though.
//
// The mdspan's value type doesn't matter either,
// so we can use a char-sized type to minimize storage.
// Using unsigned char makes overflow defined behavior.

inline void
cuda_internal_safe_call(cudaError e, const char* name,
  const char* file, int line_number)
{
  if (cudaSuccess != e) {
    std::ostringstream out;
    out << name << " error( " << cudaGetErrorName(e)
        << "): " << cudaGetErrorString(e);
    if (file) {
      out << " " << file << ":" << line_number;
    }
    throw std::runtime_error(out.str());
  }
}

#define CUDA_SAFE_CALL(call) \
  cuda_internal_safe_call(call, #call, __FILE__, __LINE__)

namespace submdspan_benchmark {

struct cuda_execution_space {};

template<class ValueType>
struct cuda_array_deleter {
  void operator() (ValueType* ptr) const {
    CUDA_SAFE_CALL(cudaFree(ptr));
  }
};

template<class ValueType>
struct array_deleter<cuda_execution_space, ValueType> {
  using type = cuda_array_deleter<ValueType>;
};

template<class ValueType>
std::unique_ptr<ValueType[], cuda_array_deleter<ValueType>>
allocate_buffer(cuda_execution_space, size_t num_elements) {
  ValueType* buf = nullptr;
  CUDA_SAFE_CALL(cudaMalloc(&buf, num_elements * sizeof(ValueType)));
  return std::unique_ptr<ValueType[], cuda_array_deleter<ValueType>>{buf, {}};
}

template<class ValueType>
requires(std::is_trivially_copyable_v<ValueType>)
void
copy_buffer(cuda_execution_space, const ValueType in[], ValueType out[], std::size_t num_elements)
{
  if (in != out) {
    const std::size_t num_bytes = num_elements * sizeof(ValueType);
    CUDA_SAFE_CALL(cudaMemcpy(out, in, num_bytes, cudaMemcpyDefault));
  }
}

template<class IndexType, class Layout, std::size_t... Exts>
__global__ void
benchmark2_loop_kernel(
  Kokkos::mdspan<std::uint8_t, Kokkos::extents<IndexType, Exts...>, Layout> out)
{
  benchmark2_loop(cuda_execution_space{}, out);
}

template<class IndexType, size_t... Exts>
size_t benchmark2_impl(cuda_execution_space exec_space,
  benchmark::State& state,
  nonconst_test_mdspan<IndexType, Exts...> out)
{
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));
  
  size_t count = 0;

  CUDA_SAFE_CALL(cudaEventRecord(start));  
  for (auto _ : state) {
    benchmark2_loop_kernel<<< 1, 1 >>>(out);
    ++count;
  }
  CUDA_SAFE_CALL(cudaEventRecord(stop));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));

  std::cout << "Count: " << count << "\nTime (ms): " << milliseconds << "\n";
  benchmark::DoNotOptimize(count);
  return count;
}

} // namespace submdspan_benchmark

template<class IndexType, size_t... Exts>
void cuda_benchmark2(benchmark::State& state,
  Kokkos::extents<IndexType, Exts...> exts)
{
  return submdspan_benchmark::benchmark2(submdspan_benchmark::cuda_execution_space{}, state, exts);
}

BENCHMARK_CAPTURE(cuda_benchmark2, int_6d, (Kokkos::extents<int, 2, 2, 2, 2, 2, 2>{}));
BENCHMARK_CAPTURE(cuda_benchmark2, int_6d, (Kokkos::dextents<int, 6>{2, 2, 2, 2, 2, 2}));
BENCHMARK_CAPTURE(cuda_benchmark2, size_t_6d, (Kokkos::extents<size_t, 2, 2, 2, 2, 2, 2>{}));
BENCHMARK_CAPTURE(cuda_benchmark2, size_t_6d, (Kokkos::dextents<size_t, 6>{2, 2, 2, 2, 2, 2}));

BENCHMARK_MAIN();

namespace test {

dim3 get_bench_thread_block(size_t y,size_t z) {
  cudaDeviceProp cudaProp;
  size_t dim_z = 1;
  while(dim_z*3<z && dim_z<32) dim_z*=2;
  CUDA_SAFE_CALL(cudaGetDeviceProperties(&cudaProp, 1));
  size_t dim_y = 16;
  while(dim_y*3<y && dim_y<32) dim_y*=2;

  return dim3(1, static_cast<int>(dim_y), static_cast<int>(dim_z));
}

template <class F, class... Args>
__global__
void do_run_kernel(F f, Args... args) {
  f(args...);
}

template <class F, class... Args>
float run_kernel_timed(size_t N, size_t M, size_t K, F&& f, Args&&... args) {
  cudaEvent_t start, stop;
  CUDA_SAFE_CALL(cudaEventCreate(&start));
  CUDA_SAFE_CALL(cudaEventCreate(&stop));

  CUDA_SAFE_CALL(cudaEventRecord(start));
  do_run_kernel<<<N, get_bench_thread_block(M,K)>>>(
    (F&&)f, ((Args&&) args)...
  );
  CUDA_SAFE_CALL(cudaEventRecord(stop));
  CUDA_SAFE_CALL(cudaEventSynchronize(stop));
  float milliseconds = 0;
  CUDA_SAFE_CALL(cudaEventElapsedTime(&milliseconds, start, stop));
  return milliseconds;
}

} // namespace test
