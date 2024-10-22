#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <fmt/core.h>

#include "linalg/qr.hpp"
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

TEST_CASE("QR") {
  std::vector<double> data = {0.65037424, 0.50545337, 0.87860147,
                              0.18184023, 0.85223307, 0.75013629,
                              0.66610167, 0.98789545, 0.25696842};
  std::array<size_t, 2> shape = {3, 3};
  Tensor<double, 2> A(data, shape);

  Tensor<double, 2> Q(shape);
  Tensor<double, 2> R(shape);

  linalg::qr_mgs(A.view(), Q.view(), R.view());

  std::vector<double> data_Q_np = {-0.68565218, 0.44836503,  -0.57345434,
                                   -0.1917037,  -0.87120003, -0.45195154,
                                   -0.70223271, -0.19994825, 0.68329344};
  Tensor<double, 2> Q_np(data_Q_np, shape);
}
