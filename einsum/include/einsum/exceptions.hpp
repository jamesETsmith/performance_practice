#pragma once

#include <stdexcept>
#include <string>

namespace einsum::exceptions {

/**
 * @brief Base class for all einsum-related exceptions
 */
class EinsumError : public std::runtime_error {
public:
  explicit EinsumError(const std::string &message)
      : std::runtime_error(message) {}
};

/**
 * @brief Exception thrown when the subscript string is invalid
 */
class InvalidSubscriptError : public EinsumError {
public:
  explicit InvalidSubscriptError(const std::string &message)
      : EinsumError("Invalid subscript: " + message) {}
};

/**
 * @brief Exception thrown when dimensions don't match across operands
 */
class DimensionMismatchError : public EinsumError {
public:
  explicit DimensionMismatchError(const std::string &message)
      : EinsumError("Dimension mismatch: " + message) {}
};

/**
 * @brief Exception thrown when input validation fails
 */
class InvalidInputError : public EinsumError {
public:
  explicit InvalidInputError(const std::string &message)
      : EinsumError("Invalid input: " + message) {}
};

} // namespace einsum::exceptions