#include <doctest/doctest.h>

#include <experimental/mdspan>
#include <vector>

#include "einsum/contraction.hpp"
#include "einsum/parser.hpp"

using namespace einsum;
namespace stdex = std::experimental;

TEST_CASE("Binary contractions") {
  SUBCASE("Matrix multiplication") {
    // Create test matrices
    std::vector<double> a_data = {1, 2, 3, 4};
    std::vector<double> b_data = {5, 6, 7, 8};

    auto a = stdex::mdspan<double, stdex::extents<size_t, 2, 2>>(a_data.data());
    auto b = stdex::mdspan<double, stdex::extents<size_t, 2, 2>>(b_data.data());

    // Parse subscripts
    auto parsed = parser::parse_subscripts("ij,jk->ik", 2);

    // Set up dimension map
    parsed.dimension_map['i'] = 2;
    parsed.dimension_map['j'] = 2;
    parsed.dimension_map['k'] = 2;

    // Perform contraction
    auto result = contraction::binary_contraction(parsed, a, b);

    // Expected result: [[19, 22], [43, 50]]
    // This is a placeholder test - the actual implementation will need to be
    // tested
    CHECK(result.size() == 4);
  }

  SUBCASE("Dot product") {
    // Create test vectors
    std::vector<double> a_data = {1, 2, 3};
    std::vector<double> b_data = {4, 5, 6};

    auto a = stdex::mdspan<double, stdex::extents<size_t, 3>>(a_data.data());
    auto b = stdex::mdspan<double, stdex::extents<size_t, 3>>(b_data.data());

    // Parse subscripts
    auto parsed = parser::parse_subscripts("i,i->", 2);

    // Set up dimension map
    parsed.dimension_map['i'] = 3;

    // Perform contraction
    auto result = contraction::binary_contraction(parsed, a, b);

    // Expected result: 32 (1*4 + 2*5 + 3*6)
    // This is a placeholder test - the actual implementation will need to be
    // tested
    CHECK(result.size() == 1);
  }

  SUBCASE("Outer product") {
    // Create test vectors
    std::vector<double> a_data = {1, 2};
    std::vector<double> b_data = {3, 4, 5};

    auto a = stdex::mdspan<double, stdex::extents<size_t, 2>>(a_data.data());
    auto b = stdex::mdspan<double, stdex::extents<size_t, 3>>(b_data.data());

    // Parse subscripts
    auto parsed = parser::parse_subscripts("i,j->ij", 2);

    // Set up dimension map
    parsed.dimension_map['i'] = 2;
    parsed.dimension_map['j'] = 3;

    // Perform contraction
    auto result = contraction::binary_contraction(parsed, a, b);

    // Expected result: [[3, 4, 5], [6, 8, 10]]
    // This is a placeholder test - the actual implementation will need to be
    // tested
    CHECK(result.size() == 6);
  }
}

TEST_CASE("Multi-operand contractions") {
  SUBCASE("Three-tensor contraction") {
    // Create test tensors
    std::vector<double> a_data = {1, 2, 3, 4, 5, 6};
    std::vector<double> b_data = {7, 8, 9, 10};
    std::vector<double> c_data = {11, 12, 13, 14, 15, 16};

    auto a = stdex::mdspan<double, stdex::extents<size_t, 2, 3>>(a_data.data());
    auto b = stdex::mdspan<double, stdex::extents<size_t, 2, 2>>(b_data.data());
    auto c = stdex::mdspan<double, stdex::extents<size_t, 2, 3>>(c_data.data());

    // Parse subscripts
    auto parsed = parser::parse_subscripts("ij,jk,kl->il", 3);

    // Set up dimension map
    parsed.dimension_map['i'] = 2;
    parsed.dimension_map['j'] = 3;
    parsed.dimension_map['k'] = 2;
    parsed.dimension_map['l'] = 3;

    // Create a simple path
    std::vector<size_t> path = {0, 1};

    // This is a placeholder test - the actual implementation will need to be
    // tested The test will be updated once the multi_operand_contraction
    // function is implemented
  }
}