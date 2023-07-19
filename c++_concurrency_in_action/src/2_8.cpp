#include <algorithm>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

template <typename Iterator, typename T> struct accumulate_block {
  void operator()(Iterator first, Iterator last, T &result) {
    result = std::accumulate(first, last, result);
  }
};

template <typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init) {
  size_t const length = std::distance(first, last);

  // 1) Input check
  if (length == 0) {
    return init;
  }

  // 2)/3)/4) Pick appropriate number of threads
  size_t const min_per_thread = 25;
  size_t const max_threads = (length + min_per_thread - 1) / min_per_thread;
  size_t const hardware_threads = std::thread::hardware_concurrency();
  size_t const num_threads =
      std::min(hardware_threads != 0 ? hardware_threads : 2, max_threads);
  std::cout << "Using " << num_threads << " threads" << std::endl;
  size_t const block_size = length / num_threads;

  std::vector<T> results(num_threads);
  std::vector<std::thread> threads(num_threads - 1);
  Iterator block_start = first;
  for (size_t i = 0; i < num_threads - 1; ++i) {
    Iterator block_end = block_start;
    std::advance(block_end, block_size);

    // 7) do the work
    threads[i] = std::thread(accumulate_block<Iterator, T>(), block_start,
                             block_end, std::ref(results[i]));
    block_start = block_end;
  }
  // 9) take into account any uneven division in this last block
  accumulate_block<Iterator, T>()(block_start, last, results[num_threads - 1]);

  // 10) Joing everything
  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));
  // 11) Gather the results from the threads, accumulate, and return
  return std::accumulate(results.begin(), results.end(), init);
}

int main() {
  std::vector<int> v1(100);
  std::iota(v1.begin(), v1.end(), 0);
  int res = parallel_accumulate(v1.begin(), v1.end(), 0);
  std::cout << res << std::endl;
}