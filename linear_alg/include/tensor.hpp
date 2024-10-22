#ifndef __TENSOR_HPP
#define __TENSOR_HPP

#include <array>
#include <cstddef>
// #include <experimental/linalg>
#include <experimental/mdspan>
#include <span>
#include <vector>

using namespace std::experimental;

template <typename T, size_t R, class LayoutPolicy = std::layout_right,
          class AccesorPolicy = std::default_accessor<T>>
class Tensor {
  std::vector<T> __data;
  std::array<size_t, R> __shape;

  using mds_t =
      std::mdspan<T, std::dextents<size_t, R>, LayoutPolicy, AccesorPolicy>;
  mds_t __view;

public:
  Tensor() = default;

  Tensor(std::span<size_t> shape) {
    std::copy(shape.begin(), shape.end(), __shape.begin());
  }

  Tensor(std::span<T> data, std::array<size_t, R> shape) {
    __shape = shape;
    __data.reserve(data.size());
    std::copy(data.begin(), data.end(), __data.begin());
    __view = mds_t(__data.data(), shape);
  }

  // Use view
  template <class... SizeTypes>
  inline T &operator[](SizeTypes... indices) const {
    return __view[std::forward<SizeTypes>(indices)...];
  }

  // Return a view
  mds_t view() const { return __view; }
};

#endif
