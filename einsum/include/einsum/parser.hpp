#pragma once

#include <algorithm>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "einsum/exceptions.hpp"

namespace einsum::parser {

/**
 * @brief Represents a parsed einsum subscript specification
 */
struct ParsedSubscripts {
  // Input operand indices (e.g., for "ij,jk->ik", this would be [['i','j'],
  // ['j','k']])
  std::vector<std::vector<char>> input_indices;

  // Output indices (e.g., for "ij,jk->ik", this would be ['i','k'])
  std::vector<char> output_indices;

  // Indices that are contracted (summed over)
  std::vector<char> contracted_indices;

  // Map from index label to its dimension size (determined during validation)
  std::unordered_map<char, size_t> dimension_map;

  // Whether ellipsis was used in the notation
  bool has_ellipsis = false;

  // Positions of ellipsis in each operand, if present
  std::vector<std::optional<size_t>> ellipsis_positions;
};

/**
 * @brief Parses an einsum subscript string into a structured representation
 *
 * @param subscripts The subscript string (e.g., "ij,jk->ik")
 * @param num_operands Expected number of operands
 * @return ParsedSubscripts Structure containing the parsed information
 * @throws InvalidSubscriptError if the subscript string is invalid
 */
ParsedSubscripts parse_subscripts(std::string_view subscripts,
                                  size_t num_operands);

/**
 * @brief Expands ellipsis notation in the parsed subscripts based on operand
 * dimensions
 *
 * @param parsed The parsed subscripts to update
 * @param operand_dims Dimensions of each operand
 */
void expand_ellipsis(ParsedSubscripts &parsed,
                     const std::vector<std::vector<size_t>> &operand_dims);

} // namespace einsum::parser