cmake_minimum_required(VERSION 3.27)

set(CMAKE_EXPORT_COMPILE_COMMANDS
    TRUE
    CACHE BOOL "Export compile commands to build directory" FORCE)

include(cmake/CPM.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS
    TRUE
    CACHE BOOL "Export compile commands to build directory" FORCE)

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
project(einsum LANGUAGES CXX)

# Build type
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
                                               "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3")

# Main library
add_library(einsum INTERFACE)
target_link_libraries(einsum INTERFACE std::linalg fmt::fmt mdspan)
target_include_directories(
  einsum INTERFACE
  include
  ${stdBLAS_BINARY_DIR}/include/experimental
)

# Enable testing
enable_testing()

# Add test directory
add_subdirectory(test)

# # Add examples directory if it exists
# if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples")
#   add_subdirectory(examples)
# endif()

# # Install targets
# install(TARGETS einsum
#         EXPORT einsumTargets
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib
#         RUNTIME DESTINATION bin
#         INCLUDES DESTINATION include)

# install(DIRECTORY include/
#         DESTINATION include
#         FILES_MATCHING PATTERN "*.hpp")

# # Export targets
# install(EXPORT einsumTargets
#         FILE einsumTargets.cmake
#         NAMESPACE einsum::
#         DESTINATION lib/cmake/einsum)

