cmake_minimum_required(VERSION 3.20)

project(url_downloader)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-g -O3 -march=native")

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)
CPMAddPackage("gh:jpbarrette/curlpp#1d8c7876cc81d7d125b663066282b207d9cbfe9a")

find_package(OpenMP REQUIRED)

add_executable(url_downloader url_downloader.cpp)
target_link_libraries(url_downloader PUBLIC curlpp OpenMP::OpenMP_CXX)
