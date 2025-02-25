#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "einsum/exceptions.hpp"
#include "einsum/parser.hpp"

using namespace einsum;

TEST_CASE("Parse basic subscripts") {
  SUBCASE("Matrix multiplication") {
    auto parsed = parser::parse_subscripts("ij,jk->ik", 2);

    CHECK(parsed.input_indices.size() == 2);
    CHECK(parsed.input_indices[0] == std::vector<char>{'i', 'j'});
    CHECK(parsed.input_indices[1] == std::vector<char>{'j', 'k'});
    CHECK(parsed.output_indices == std::vector<char>{'i', 'k'});
    CHECK(parsed.contracted_indices == std::vector<char>{'j'});
    CHECK_FALSE(parsed.has_ellipsis);
  }

  SUBCASE("Dot product") {
    auto parsed = parser::parse_subscripts("i,i->", 2);

    CHECK(parsed.input_indices.size() == 2);
    CHECK(parsed.input_indices[0] == std::vector<char>{'i'});
    CHECK(parsed.input_indices[1] == std::vector<char>{'i'});
    CHECK(parsed.output_indices.empty());
    CHECK(parsed.contracted_indices == std::vector<char>{'i'});
    CHECK_FALSE(parsed.has_ellipsis);
  }

  SUBCASE("Outer product") {
    auto parsed = parser::parse_subscripts("i,j->ij", 2);

    CHECK(parsed.input_indices.size() == 2);
    CHECK(parsed.input_indices[0] == std::vector<char>{'i'});
    CHECK(parsed.input_indices[1] == std::vector<char>{'j'});
    CHECK(parsed.output_indices == std::vector<char>{'i', 'j'});
    CHECK(parsed.contracted_indices.empty());
    CHECK_FALSE(parsed.has_ellipsis);
  }

  SUBCASE("Trace") {
    auto parsed = parser::parse_subscripts("ii->", 1);

    CHECK(parsed.input_indices.size() == 1);
    CHECK(parsed.input_indices[0] == std::vector<char>{'i', 'i'});
    CHECK(parsed.output_indices.empty());
    CHECK(parsed.contracted_indices == std::vector<char>{'i'});
    CHECK_FALSE(parsed.has_ellipsis);
  }

  SUBCASE("Tensor contraction") {
    auto parsed = parser::parse_subscripts("abc,cd,def->abef", 3);

    CHECK(parsed.input_indices.size() == 3);
    CHECK(parsed.input_indices[0] == std::vector<char>{'a', 'b', 'c'});
    CHECK(parsed.input_indices[1] == std::vector<char>{'c', 'd'});
    CHECK(parsed.input_indices[2] == std::vector<char>{'d', 'e', 'f'});
    CHECK(parsed.output_indices == std::vector<char>{'a', 'b', 'e', 'f'});
    CHECK(parsed.contracted_indices.size() == 2);
    CHECK_FALSE(parsed.has_ellipsis);
  }
}

TEST_CASE("Parse subscripts with ellipsis") {
  SUBCASE("Basic ellipsis") {
    auto parsed = parser::parse_subscripts("...ij,...jk->...ik", 2);

    CHECK(parsed.input_indices.size() == 2);
    CHECK(parsed.has_ellipsis);
    CHECK(parsed.ellipsis_positions.size() == 2);
    CHECK(parsed.ellipsis_positions[0].has_value());
    CHECK(parsed.ellipsis_positions[0].value() == 0);
    CHECK(parsed.ellipsis_positions[1].has_value());
    CHECK(parsed.ellipsis_positions[1].value() == 0);
  }

  SUBCASE("Ellipsis in different positions") {
    auto parsed = parser::parse_subscripts("ij...,jk...->i...k", 2);

    CHECK(parsed.input_indices.size() == 2);
    CHECK(parsed.has_ellipsis);
    CHECK(parsed.ellipsis_positions.size() == 2);
    CHECK(parsed.ellipsis_positions[0].has_value());
    CHECK(parsed.ellipsis_positions[0].value() == 2);
    CHECK(parsed.ellipsis_positions[1].has_value());
    CHECK(parsed.ellipsis_positions[1].value() == 2);
  }
}

TEST_CASE("Invalid subscripts") {
  SUBCASE("Mismatched operands") {
    CHECK_THROWS_AS(parser::parse_subscripts("ij,jk->ik", 3),
                    exceptions::InvalidSubscriptError);
  }

  SUBCASE("Invalid characters") {
    CHECK_THROWS_AS(parser::parse_subscripts("ij,j$->i", 2),
                    exceptions::InvalidSubscriptError);
  }

  SUBCASE("Missing output") {
    CHECK_THROWS_AS(parser::parse_subscripts("ij,jk", 2),
                    exceptions::InvalidSubscriptError);
  }

  SUBCASE("Multiple ellipses") {
    CHECK_THROWS_AS(parser::parse_subscripts("...ij...->ik", 1),
                    exceptions::InvalidSubscriptError);
  }

  SUBCASE("Output indices not in input") {
    CHECK_THROWS_AS(parser::parse_subscripts("ij,jk->il", 2),
                    exceptions::InvalidSubscriptError);
  }
}

TEST_CASE("Ellipsis expansion") {
  SUBCASE("Basic expansion") {
    auto parsed = parser::parse_subscripts("...ij,...jk->...ik", 2);
    std::vector<std::vector<size_t>> operand_dims = {
        {2, 3, 4, 5}, // First operand has shape (2,3,4,5)
        {4, 5, 6}     // Second operand has shape (4,5,6)
    };

    parser::expand_ellipsis(parsed, operand_dims);

    // After expansion, input_indices should be:
    // [['0', '1', 'i', 'j'], ['0', '1', 'j', 'k']]
    // where '0' and '1' are placeholder indices for the expanded ellipsis
    CHECK(parsed.input_indices[0].size() == 4);
    CHECK(parsed.input_indices[1].size() == 4);
    CHECK(parsed.output_indices.size() == 4);
  }
}