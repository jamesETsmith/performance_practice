#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <string>

#include "doctest/doctest.h"
#include "vector.hpp"

TEST_CASE("test_vec") {
  my_stl::vector<double> v1;

  CHECK(v1.first == nullptr);

  my_stl::vector<int> v2(100);
  CHECK(v2.size() == 100);
  CHECK(v2.capacity() == 100);

  my_stl::vector<double> v3(2);
  v3[0] = 1.0;
  v3[1] = 2.0;
  CHECK(v3[0] == 1.0);
  CHECK(v3[1] == 2.0);

  size_t const old_capacity = v3.capacity();
  v3.push_back(-1.5);

  CHECK(v3.capacity() == old_capacity * 2);
  CHECK(v3[2] == -1.5);
}

TEST_CASE("test sting") {
  my_stl::vector<std::string> v1;
  my_stl::vector<std::string> v2(10);

  v2[0] = std::string("hello world");
  CHECK(v2[0] == "hello world");
  CHECK(v2[1] == "");
}

TEST_CASE("test struct") {
  struct A {};
  my_stl::vector<A> v1;
}