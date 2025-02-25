#pragma once

#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <experimental/linalg>
#include <mdspan/mdspan.hpp>

#include "einsum/contraction.hpp"
#include "einsum/exceptions.hpp"
#include "einsum/parser.hpp"
#include "einsum/utils.hpp"

namespace einsum {

/**
 * @brief Performs tensor contractions using Einstein summation convention
 *
 * @param subscripts String specifying the subscripts for summation as
 * comma-separated list of subscript labels for each operand, followed by '->'
 * and the output subscript labels
 * @param operands   One or more tensor-like objects that can be wrapped in
 * mdspan
 * @return           Result of the contraction as an mdspan
 *
 * Example:
 *   auto result = einsum::einsum("ij,jk->ik", matrix_a, matrix_b); // Matrix
 * multiplication
 */
template <typename... MdspanTypes>
auto einsum(std::string_view subscripts, MdspanTypes &&...operands) {
  // Parse the subscript notation
  auto parsed = parser::parse_subscripts(subscripts, sizeof...(operands));

  // Validate the inputs against the parsed subscripts
  utils::validate_inputs(parsed, std::forward<MdspanTypes>(operands)...);

  // Perform the contraction
  return contraction::contract(parsed, std::forward<MdspanTypes>(operands)...);
}

} // namespace einsum