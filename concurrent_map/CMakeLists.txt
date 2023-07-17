cmake_minimum_required(VERSION 3.20)

project(concurrent_map)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -march=native")

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)

find_package(OpenMP REQUIRED)

add_executable(concurrent_map concurrent_map.cpp)
target_link_libraries(concurrent_map PUBLIC OpenMP::OpenMP_CXX)


CPMAddPackage("gh:greg7mdp/parallel-hashmap#v1.3.11")

add_executable(phmap_test phmap_test.cpp)
target_link_libraries(phmap_test PUBLIC phmap OpenMP::OpenMP_CXX)