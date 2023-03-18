#include "max_subarray.hpp"

// Test helpers

// From :
// https://www.fluentcpp.com/2019/05/24/how-to-fill-a-cpp-collection-with-random-values/
auto randomNumberBetween = [](int low, int high) {
  auto randomFunc =
      [distribution_ = std::uniform_int_distribution<int>(low, high),
       random_engine_ = std::mt19937{std::random_device{}()}]() mutable {
        return distribution_(random_engine_);
      };
  return randomFunc;
};

std::vector<std::vector<int>> make_test_cases() {
  // Keep things the same
  srand(20);
  int const n_tests = 200;
  int const element_max = 1e4;
  int const element_min = -1e4;
  int const length_max = 1e5;

  std::vector<std::vector<int>> tests(n_tests);
  for (int i = 0; i < n_tests; i++) {
    tests[i].resize(randomNumberBetween(1, length_max)());
    std::generate(tests[i].begin(), tests[i].end(),
                  randomNumberBetween(element_min, element_max));
  }

  return tests;
}

// Solutions
int maxSubArray_5(std::vector<int>& nums) {
  int curMax = 0, maxTillNow = std::numeric_limits<int>::min();
  for (auto c : nums)
    curMax = std::max(c, curMax + c), maxTillNow = std::max(maxTillNow, curMax);
  return maxTillNow;
}