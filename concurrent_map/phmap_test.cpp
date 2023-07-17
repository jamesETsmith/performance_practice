#include <array>
#include <numeric>
#include <parallel_hashmap/phmap.h>
#include <string>
#include <vector>

#define PHMAP_ALLOCATOR_NOTHROW 1

int main(int argc, char **argv) {

  //
  // User args
  //

  size_t const N = std::stoi(argv[1]);

  phmap::parallel_flat_hash_map<int, std::vector<int>> mymap;

#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < N; i++) {
      std::vector<int> vi(10);
      std::iota(vi.begin(), vi.end(), 0);
      // mymap.insert({i, vi});
      mymap[i] = vi;
    }

  } //

  return 0;
}