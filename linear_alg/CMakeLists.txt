cmake_minimum_required(VERSION 3.27)

set(CMAKE_EXPORT_COMPILE_COMMANDS
    TRUE
    CACHE BOOL "Export compile commands to build directory" FORCE)

include(cmake/CPM.cmake)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Need to enforce -fPIC across whole project to build shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#
# Project specific configuration
#

# Dependencies
cpmaddpackage("gh:doctest/doctest@2.4.11")
cpmaddpackage("gh:fmtlib/fmt#10.2.1")
cpmaddpackage("gh:kokkos/mdspan#b885a2c60ad42f9e1aaa0d317a38105b950cbed0")
cpmaddpackage("gh:kokkos/stdBLAS#7988045feeb179d294fb2697ce7c8ba155f060fd")

# Project
project(tensor LANGUAGES CXX)

# Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")

add_library(tensor INTERFACE)
target_link_libraries(tensor INTERFACE fmt::fmt std::linalg mdspan)
target_include_directories(
  tensor INTERFACE include ${stdBLAS_BUILD_DIR}/include/experimental)

#

add_subdirectory(test)
