#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include <experimental/linalg>
#include <mdspan/mdspan.hpp>

#include "einsum/parser.hpp"
#include "einsum/utils.hpp"

namespace einsum::contraction {

/**
 * @brief Performs the tensor contraction based on parsed subscripts
 *
 * @param parsed The parsed subscript information
 * @param operands The tensor operands
 * @return Result of the contraction as an mdspan
 */
template <typename... MdspanTypes>
auto contract(const parser::ParsedSubscripts &parsed,
              MdspanTypes &&...operands) {
  // Determine the optimal contraction path
  auto path = optimize_contraction_path(parsed, operands...);

  // Perform the contraction following the optimized path
  return execute_contraction_path(path, parsed,
                                  std::forward<MdspanTypes>(operands)...);
}

/**
 * @brief Determines the optimal contraction path to minimize computational cost
 */
template <typename... MdspanTypes>
auto optimize_contraction_path(const parser::ParsedSubscripts &parsed,
                               const MdspanTypes &...operands) {
  // For now, use a simple greedy algorithm
  // In the future, implement more sophisticated optimization strategies
  return utils::greedy_path(parsed, operands...);
}

/**
 * @brief Executes the contraction following the specified path
 */
template <typename... MdspanTypes>
auto execute_contraction_path(const auto &path,
                              const parser::ParsedSubscripts &parsed,
                              MdspanTypes &&...operands) {
  // Implementation will depend on the path representation
  // This is a placeholder for the actual implementation
  if constexpr (sizeof...(operands) == 2) {
    // Special case for binary operations (common case)
    return binary_contraction(parsed, std::forward<MdspanTypes>(operands)...);
  } else {
    // General case for multiple operands
    return multi_operand_contraction(path, parsed,
                                     std::forward<MdspanTypes>(operands)...);
  }
}

/**
 * @brief Specialized implementation for binary contractions (two operands)
 */
template <typename MdspanType1, typename MdspanType2>
auto binary_contraction(const parser::ParsedSubscripts &parsed, MdspanType1 &&a,
                        MdspanType2 &&b) {
  // Implement common binary contractions efficiently
  // This will handle cases like matrix multiplication, dot products, etc.
  // For now, this is a placeholder
  return utils::create_output_mdspan<
      typename std::decay_t<MdspanType1>::value_type>(parsed.output_indices,
                                                      parsed.dimension_map);
}

/**
 * @brief General implementation for contractions with more than two operands
 */
template <typename... MdspanTypes>
auto multi_operand_contraction(const auto &path,
                               const parser::ParsedSubscripts &parsed,
                               MdspanTypes &&...operands) {
  // Implement the general case by following the contraction path
  // This is a placeholder for the actual implementation
  return utils::create_output_mdspan<typename std::decay_t<
      std::tuple_element_t<0, std::tuple<MdspanTypes...>>>::value_type>(
      parsed.output_indices, parsed.dimension_map);
}

} // namespace einsum::contraction