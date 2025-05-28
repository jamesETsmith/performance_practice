#ifndef TENSOR_QR_HPP
#define TENSOR_QR_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <numeric>
// #include <experimental/linalg>
#include <experimental/mdspan>
#include <span>
#include <vector>

#include "tensor.hpp"

namespace linalg {

template <class mds_t, class T = typename mds_t::element_type> T norm(mds_t v) {
  T res = 0;
  for (size_t i = 0; i < v.extent(0); ++i) {
    res += v[i] * v[i];
  }
  return std::sqrt(res);
}

template <class T, class E, class L, class A>
T col_norm(std::mdspan<T, E, L, A> v, size_t col) {
  T res = 0;
  for (size_t i = 0; i < v.extent(0); ++i) {
    res += v[i, col] * v[i, col];
  }
  return std::sqrt(res);
}

template <class T, class E, class L, class A>
T dot_cols(std::mdspan<T, E, L, A> m1, size_t col1, std::mdspan<T, E, L, A> m2,
           size_t col2) {
  T res = 0;

  for (size_t i = 0; i < m1.extent(0); ++i) {
    res += m1[i, col1] * m2[i, col2];
  }

  return res;
}

/**
 * @brief QR decomposition with modified Gram-Schmidt
 *
 * From Trefethen and Bau, Numerical Linear Algebra
 * Alg. 8.1
 *
 * @tparam mds_t
 * @param A
 * @param Q
 * @param R
 */
template <class mds_t> void qr_mgs(mds_t A, mds_t Q, mds_t R) {
  size_t const n = A.extent(0);

  // Copy A to Q
  std::copy(A.data_handle(), A.data_handle() + A.size(), Q.data_handle());

  for (size_t i = 0; i < n; ++i) {
    R[i, i] = col_norm(Q, i);

    for (size_t j = 0; j < n; ++j) {
      Q[j, i] /= R[i, i];
    }

    for (size_t j = i + 1; j < n; ++j) {
      R[i, j] = dot_cols(Q, i, Q, j);
      for (size_t k = 0; k < n; ++k) {
        Q[k, j] -= R[i, j] * Q[k, i];
      }
    }
  }
}
} // namespace linalg

#endif