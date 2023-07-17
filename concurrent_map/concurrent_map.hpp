#include <iostream>
#include <omp.h>
#include <optional>
#include <unordered_map>

template <typename key, typename T> class concurrent_hash_map {
private:
  using map_type = std::unordered_map<key, T>;
  map_type data_;
  omp_lock_t lock_;

public:
  concurrent_hash_map(omp_lock_hint_t hint = omp_lock_hint_speculative) {
    omp_init_lock_with_hint(&lock_, hint);
  }

  void insert(typename map_type::value_type kv) {
    omp_set_lock(&lock_);
    data_.insert(kv);
    omp_unset_lock(&lock_);
  }

  std::optional<T> lookup(key &k) {
    omp_set_lock(&lock_);
    auto res = data_.find(k);
    omp_unset_lock(&lock_);
    return res == data_.end() ? res->second : std::nullopt;
  }
};