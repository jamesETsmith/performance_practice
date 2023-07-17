cmake_minimum_required(VERSION 3.20)

project(sparse_mat)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -march=native")

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)

CPMAddPackage("gh:doctest/doctest@2.4.11")

find_package(OpenMP REQUIRED)

add_library(sparse_mat INTERFACE)
target_include_directories(sparse_mat INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sparse_mat INTERFACE OpenMP::OpenMP_CXX)

add_executable(sparse_mat_exe src/sparse_mat.cpp)
target_include_directories(sparse_mat_exe PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sparse_mat_exe PRIVATE sparse_mat)


enable_testing()
add_subdirectory(test)