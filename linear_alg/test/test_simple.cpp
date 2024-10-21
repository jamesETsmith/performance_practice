#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <fmt/core.h>

#include "tensor.hpp"

TEST_CASE("Tensor simple") {
  Tensor<double, 2> t;
  CHECK(true);
}

TEST_CASE("Tensor ctors") {
  std::vector<int> data = {0, 1, 2, 3, 4, 5, 6, 7};
  std::array<size_t, 3> shape = {2, 2, 2};

  Tensor<int, 3> t(data, shape);
  fmt::println("Here's T[0,1,1] = {}", t[0, 1, 1]);
}