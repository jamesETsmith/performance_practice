#include <doctest/doctest.h>

#include <experimental/mdspan>
#include <vector>

#include "einsum/einsum.hpp"

using namespace einsum;
namespace stdex = std::experimental;

TEST_CASE("Einsum interface") {
  SUBCASE("Matrix multiplication") {
    // Create test matrices
    std::vector<double> a_data = {1, 2, 3, 4};
    std::vector<double> b_data = {5, 6, 7, 8};

    auto a = stdex::mdspan<double, stdex::extents<size_t, 2, 2>>(a_data.data());
    auto b = stdex::mdspan<double, stdex::extents<size_t, 2, 2>>(b_data.data());

    // Perform einsum
    auto result = einsum("ij,jk->ik", a, b);

    // Expected result: [[19, 22], [43, 50]]
    // This is a placeholder test - the actual implementation will need to be
    // tested
  }

  SUBCASE("Dot product") {
    // Create test vectors
    std::vector<double> a_data = {1, 2, 3};
    std::vector<double> b_data = {4, 5, 6};

    auto a = stdex::mdspan<double, stdex::extents<size_t, 3>>(a_data.data());
    auto b = stdex::mdspan<double, stdex::extents<size_t, 3>>(b_data.data());

    // Perform einsum
    auto result = einsum("i,i->", a, b);

    // Expected result: 32 (1*4 + 2*5 + 3*6)
    // This is a placeholder test - the actual implementation will need to be
    // tested
  }

  SUBCASE("Trace") {
    // Create test matrix
    std::vector<double> a_data = {1, 2, 3, 4};

    auto a = stdex::mdspan<double, stdex::extents<size_t, 2, 2>>(a_data.data());

    // Perform einsum
    auto result = einsum("ii->", a);

    // Expected result: 5 (1 + 4)
    // This is a placeholder test - the actual implementation will need to be
    // tested
  }

  SUBCASE("Tensor contraction with ellipsis") {
    // Create test tensors
    std::vector<double> a_data(24); // 2x3x4 tensor
    std::vector<double> b_data(12); // 4x3 tensor

    // Fill with some test data
    for (size_t i = 0; i < a_data.size(); ++i)
      a_data[i] = i + 1;
    for (size_t i = 0; i < b_data.size(); ++i)
      b_data[i] = i + 1;

    auto a =
        stdex::mdspan<double, stdex::extents<size_t, 2, 3, 4>>(a_data.data());
    auto b = stdex::mdspan<double, stdex::extents<size_t, 4, 3>>(b_data.data());

    // Perform einsum
    auto result = einsum("...ij,...ji->...", a, b);

    // This is a placeholder test - the actual implementation will need to be
    // tested
  }
}

TEST_CASE("Error handling") {
  SUBCASE("Dimension mismatch") {
    // Create test matrices with incompatible dimensions
    std::vector<double> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<double> b_data = {7, 8, 9, 10};

    auto a = stdex::mdspan<double, stdex::extents<size_t, 2, 3>>(a_data.data());
    auto b = stdex::mdspan<double, stdex::extents<size_t, 2, 2>>(b_data.data());

    // This should throw because the contracted dimension 'j' has different
    // sizes (3 vs 2)
    CHECK_THROWS_AS(einsum("ij,jk->ik", a, b),
                    exceptions::DimensionMismatchError);
  }

  SUBCASE("Invalid subscripts") {
    std::vector<double> a_data = {1, 2, 3, 4};
    auto a = stdex::mdspan<double, stdex::extents<size_t, 2, 2>>(a_data.data());

    // Invalid subscript notation
    CHECK_THROWS_AS(einsum("ij->i$", a), exceptions::InvalidSubscriptError);
  }
}