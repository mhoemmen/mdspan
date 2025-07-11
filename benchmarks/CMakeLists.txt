
function(mdspan_add_benchmark EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cpp)
  target_link_libraries(${EXENAME} mdspan benchmark::benchmark)
  target_include_directories(${EXENAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks>
  )
  # Set flag to build with parenthesis enabled
  target_compile_definitions(${EXENAME} PRIVATE MDSPAN_USE_PAREN_OPERATOR=1)
  if(MDSPAN_ENABLE_P3663)
    target_compile_definitions(${EXENAME} PUBLIC MDSPAN_ENABLE_P3663=1)
  endif()
  if(MDSPAN_CONSTANT_WRAPPER_WORKAROUND)
    target_compile_definitions(${EXENAME}
      PUBLIC
      MDSPAN_CONSTANT_WRAPPER_WORKAROUND=1
    )
  endif()
endfunction()

if(MDSPAN_USE_SYSTEM_BENCHMARK)
  find_package(benchmark REQUIRED)
else()
  set(BENCHMARK_ENABLE_TESTING OFF)
  include(FetchContent)
  
  fetchcontent_declare(
      benchmark
      GIT_REPOSITORY https://github.com/google/benchmark.git
      GIT_TAG v1.9.1
  )

  # TODO CMake 3.28, we can pass EXCLUDE_FROM_ALL directly to fetchcontent_makeavailable(benchmark)
  fetchcontent_getproperties(benchmarkgoogletest)
  if (NOT benchmark_POPULATED)
    fetchcontent_populate(benchmark)
    add_subdirectory(${benchmark_SOURCE_DIR} ${benchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

function(mdspan_add_cuda_benchmark EXENAME)
  add_executable(${EXENAME} ${EXENAME}.cu)
  target_link_libraries(${EXENAME} PUBLIC mdspan benchmark::benchmark)
  target_include_directories(${EXENAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks>
  )
  # Set flag to build with parenthesis enabled
  target_compile_definitions(${EXENAME} PRIVATE MDSPAN_USE_PAREN_OPERATOR=1)
  if(MDSPAN_ENABLE_P3663)
    target_compile_definitions(${EXENAME} PUBLIC MDSPAN_ENABLE_P3663=1)
  endif()
  if(MDSPAN_CONSTANT_WRAPPER_WORKAROUND)
    target_compile_definitions(${EXENAME}
      PUBLIC
      MDSPAN_CONSTANT_WRAPPER_WORKAROUND=1
    )
  endif()
endfunction()

if(MDSPAN_ENABLE_OPENMP)
  find_package(OpenMP)
endif()

function(mdspan_add_openmp_benchmark EXENAME)
  if(MDSPAN_ENABLE_OPENMP)
    if(OpenMP_CXX_FOUND)
      add_executable(${EXENAME} ${EXENAME}.cpp)
      target_link_libraries(${EXENAME} mdspan benchmark::benchmark OpenMP::OpenMP_CXX)
      target_include_directories(${EXENAME} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/benchmarks>
      )
      target_compile_definitions(${EXENAME} PRIVATE MDSPAN_USE_PAREN_OPERATOR=1)
      if(MDSPAN_ENABLE_P3663)
        target_compile_definitions(${EXENAME} PUBLIC MDSPAN_ENABLE_P3663=1)
      endif()
      if(MDSPAN_CONSTANT_WRAPPER_WORKAROUND)
        target_compile_definitions(${EXENAME}
          PUBLIC
          MDSPAN_CONSTANT_WRAPPER_WORKAROUND=1
        )
      endif() 
    else()
      message(WARNING "Not adding target ${EXENAME} because OpenMP was not found")
    endif()
  endif()
endfunction()

add_subdirectory(sum)
add_subdirectory(matvec)
add_subdirectory(copy)
add_subdirectory(stencil)
add_subdirectory(tiny_matrix_add)
add_subdirectory(submdspan)
