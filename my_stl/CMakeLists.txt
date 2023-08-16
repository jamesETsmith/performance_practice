cmake_minimum_required(VERSION 3.20)

project(my_stl)

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE CACHE BOOL "Export compile commands to build directory" FORCE)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -Wpedantic")
# if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fanalyzer")
# endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)

CPMAddPackage("gh:doctest/doctest@2.4.11")

find_package(OpenMP REQUIRED)

add_library(my_stl INTERFACE)
target_include_directories(my_stl INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(my_stl INTERFACE OpenMP::OpenMP_CXX)

enable_testing()
add_subdirectory(test)