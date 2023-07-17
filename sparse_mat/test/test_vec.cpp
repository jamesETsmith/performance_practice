#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#include "sparse_mat.hpp"

TEST_CASE("test_vec") {
  sparse_vec<double> v(10);
  CHECK(v.size() == 10);

  std::vector<size_t> x = {1, 2, 3, 9};
  std::vector<double> z = {-1.2, -1, 1.1, 9e10};
  v.add_elements(x, z);

  CHECK(v.get_val(1) == -1.2);
  CHECK(v.get_val(2) == -1);
  CHECK(v.get_val(3) == 1.1);
  CHECK(v.get_val(9) == 9e10);
  CHECK(v.get_val(8).has_value() != true);
}