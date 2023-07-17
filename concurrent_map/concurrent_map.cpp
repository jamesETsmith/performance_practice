#include "concurrent_map.hpp"

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

/*
typedef enum omp_sync_hint_t {
  omp_sync_hint_none = 0x0,
  omp_lock_hint_none = omp_sync_hint_none,
  omp_sync_hint_uncontended = 0x1,
  omp_lock_hint_uncontended = omp_sync_hint_uncontended,
  omp_sync_hint_contended = 0x2,
  omp_lock_hint_contended = omp_sync_hint_contended,
  omp_sync_hint_nonspeculative = 0x4,
  omp_lock_hint_nonspeculative = omp_sync_hint_nonspeculative,
  omp_sync_hint_speculative = 0x8
  omp_lock_hint_speculative = omp_sync_hint_speculative
} omp_sync_hint_t;
*/
int main(int argc, char **argv) {

  //
  // User args
  //
  omp_lock_hint_t hint;
  //    = 1 << 20;

  //   if (argc != 2) {
  //     hint = omp_sync_hint_speculative;
  //   } else {
  hint = static_cast<omp_lock_hint_t>(std::stoi(argv[1]));
  size_t const N = std::stoi(argv[2]);
  //   }

  concurrent_hash_map<int, std::vector<int>> mymap(hint);

#pragma omp parallel
  {
#pragma omp for
    for (int i = 0; i < N; i++) {
      std::vector<int> vi(10);
      std::iota(vi.begin(), vi.end(), 0);
      mymap.insert({i, vi});
    }

  } //

  return 0;
}