#ifndef __TENSOR_HPP
#define __TENSOR_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
// #include <experimental/linalg>
#include <experimental/mdspan>
#include <fmt/core.h>
#include <ranges>
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

  Tensor(std::array<size_t, R> shape) {
    __shape = shape;
    __data.resize(
        std::reduce(shape.begin(), shape.end(), 1, std::multiplies{}));
    __view = mds_t(__data.data(), __shape);
  }

  Tensor(std::span<T> data, std::array<size_t, R> shape) {
    __shape = shape;
    __data.resize(data.size());
    std::copy(data.begin(), data.end(), __data.begin());
    __view = mds_t(__data.data(), shape);
  }

  bool operator==(const Tensor &other) const {
    return std::ranges::equal(__shape, other.__shape) &&
           std::ranges::equal(__data, other.__data);
  }

  bool isclose(const Tensor &other, double rtol = 1e-5,
               double atol = 1e-8) const {
    return std::ranges::equal(
        __data, other.__data, [rtol, atol](const auto &a, const auto &b) {
          return std::abs(a - b) <= atol + rtol * std::abs(b);
        });
  }

  // Use view
  template <class... SizeTypes>
  inline T &operator[](SizeTypes... indices) const {
    return __view[std::forward<SizeTypes>(indices)...];
  }

  // Return a view
  mds_t view() const { return __view; }
};

/** Print tensor (mostly from cursorai)*/
template <typename T, size_t R, class LayoutPolicy, class AccesorPolicy>
struct fmt::formatter<Tensor<T, R, LayoutPolicy, AccesorPolicy>> {
  constexpr auto parse(format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const Tensor<T, R, LayoutPolicy, AccesorPolicy> &tensor,
              FormatContext &ctx) {
    auto out = ctx.out();
    auto shape = tensor.view().extents();
    out = fmt::format_to(out, "Tensor(shape=[");
    for (size_t i = 0; i < R; ++i) {
      out = fmt::format_to(out, "{}", shape.extent(i));
      if (i < R - 1) {
        out = fmt::format_to(out, ", ");
      }
    }
    out = fmt::format_to(out, "], data=\n[\n");

    size_t total_size = tensor.view().size();
    std::string indent(R, ' ');

    for (size_t i = 0; i < total_size; ++i) {
      for (size_t r = 1; r < R; ++r) {
        if (i % shape.extent(r) == 0 && i != 0) {
          out = fmt::format_to(out, " ] ");
          if (r == R - 1) {
            out = fmt::format_to(out, "\n");
          }
        }
      }
      for (size_t r = 1; r < R; ++r) {
        if (i % shape.extent(r) == 0) {
          out = fmt::format_to(out, "{}[ ", indent);
        }
      }
      out = fmt::format_to(out, " {} ", tensor.view().data_handle()[i]);
    }

    for (size_t r = 0; r < R - 1; ++r) {
      out = fmt::format_to(out, " ] ");
    }
    out = fmt::format_to(out, "\n]\n");

    return out;
  }
};

#endif
