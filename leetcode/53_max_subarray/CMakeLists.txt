cmake_minimum_required (VERSION 3.16)

project (lc_53)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)
message(STATUS "Always set to CMAKE_BUILD_TYPE=Release")

add_library(maxsubarray max_subarray.cpp)

# Benchmarks
find_package(benchmark REQUIRED 
    HINTS ${CMAKE_SOURCE_DIR}/../../benchmark/build/install
)
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark maxsubarray benchmark::benchmark)
