#pragma once

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <mdspan/mdspan.hpp>

#include "einsum/parser.hpp"

namespace einsum::utils {

/**
 * @brief Validates that the input operands match the parsed subscripts
 *
 * @param parsed The parsed subscript information
 * @param operands The tensor operands
 * @throws DimensionMismatchError if dimensions don't match the subscripts
 */
template <typename... MdspanTypes>
void validate_inputs(parser::ParsedSubscripts &parsed,
                     const MdspanTypes &...operands) {
  // Check that the number of operands matches the parsed subscripts
  if (sizeof...(operands) != parsed.input_indices.size()) {
    throw exceptions::InvalidInputError(
        "Number of operands doesn't match subscripts");
  }

  // Extract dimensions from operands and validate against subscripts
  std::vector<std::vector<size_t>> operand_dims = {get_dimensions(operands)...};

  // If ellipsis is used, expand it based on actual dimensions
  if (parsed.has_ellipsis) {
    parser::expand_ellipsis(parsed, operand_dims);
  }

  // Validate dimensions and build dimension map
  validate_dimensions(parsed, operand_dims);
}

/**
 * @brief Extracts the dimensions of an mdspan
 */
template <typename MdspanType>
std::vector<size_t> get_dimensions(const MdspanType &mdspan) {
  std::vector<size_t> dims;
  dims.reserve(mdspan.rank());

  for (size_t i = 0; i < mdspan.rank(); ++i) {
    dims.push_back(mdspan.extent(i));
  }

  return dims;
}

/**
 * @brief Validates dimensions across operands and builds dimension map
 */
void validate_dimensions(parser::ParsedSubscripts &parsed,
                         const std::vector<std::vector<size_t>> &operand_dims);

/**
 * @brief Creates a greedy contraction path
 */
template <typename... MdspanTypes>
auto greedy_path(const parser::ParsedSubscripts &parsed,
                 const MdspanTypes &...operands) {
  // Simple greedy algorithm for determining contraction order
  // This is a placeholder for a more sophisticated implementation
  std::vector<size_t> path(sizeof...(operands) - 1);
  std::iota(path.begin(), path.end(), 0);
  return path;
}

/**
 * @brief Creates an output mdspan with the appropriate dimensions
 */
template <typename ElementType>
auto create_output_mdspan(
    const std::vector<char> &output_indices,
    const std::unordered_map<char, size_t> &dimension_map) {
  // Determine output dimensions from output indices and dimension map
  std::vector<size_t> extents;
  extents.reserve(output_indices.size());

  for (char idx : output_indices) {
    extents.push_back(dimension_map.at(idx));
  }

  // Create and return the output mdspan
  // This is a simplified version; the actual implementation will depend on the
  // mdspan API
  if (extents.empty()) {
    // Scalar output
    return std::vector<ElementType>{ElementType{}};
  } else {
    // Tensor output
    size_t total_size = std::accumulate(extents.begin(), extents.end(),
                                        size_t{1}, std::multiplies<>());
    return std::vector<ElementType>(total_size);
  }
}

} // namespace einsum::utils