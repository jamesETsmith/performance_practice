cmake_minimum_required(VERSION 3.20)

project(url_downloader)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)
CPMAddPackage("gh:jpbarrette/curlpp#1d8c7876cc81d7d125b663066282b207d9cbfe9a")

add_executable(url_downloader url_downloader.cpp)
target_link_libraries(url_downloader curlpp)