#include <benchmark/benchmark.h>

#include "max_subarray.hpp"

static void bench_5(benchmark::State& state) {
  std::vector<std::vector<int>> tests = make_test_cases();

  for (auto _ : state) {
    for (auto& t : tests) {
      maxSubArray_5(t);  // Don't worry about return for now
    }
  }
}

BENCHMARK(bench_5);

BENCHMARK_MAIN();