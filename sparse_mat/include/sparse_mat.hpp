#include <assert.h>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <typeinfo>
#include <unordered_map>
#include <vector>

template <typename T> class sparse_vec {
  using row_t = std::unordered_map<size_t, T>;
  using val_t = T;

  size_t const m_;
  size_t nnz_;
  row_t data_;

public:
  sparse_vec(size_t const m) : m_(m) {}
  sparse_vec(size_t const m, std::span<size_t> x, std::span<T> z) : m_(m) {
    add_elements(x, z);
  }

  void add_elements(std::span<size_t> x, std::span<T> z) {
    assert(x.size() == z.size());

    nnz_ = x.size();

    for (size_t i = 0; i < nnz_; i++) {
      data_[x[i]] = z[i];
    }
  }

  std::optional<T> get_val(size_t i) {
    assert(i < m_);
    auto res = data_.find(i);
    return res != data_.end() ? std::optional<T>{res->second} : std::nullopt;
  }

  friend std::ostream &operator<<(std::ostream &out, sparse_vec const &me) {
    out << "sparse_vec:\n\ttype: " << typeid(T).name();
    out << "\n\tm: " << me.m_ << "   nnz: " << me.nnz_;
    out << std::endl;

    for (auto it = me.data_.begin(); it != me.data_.end(); ++it) {
      out << "\t"
          << " " << it->first << " " << it->second << std::endl;
    }
    return out;
  }

  size_t size() { return m_; }
};

template <typename T> class sparse_mat {

  using row_t = std::unordered_map<size_t, T>;
  using val_t = T;

  size_t const m_;
  size_t const n_;
  size_t nnz_;
  std::vector<row_t> data_;

public:
  sparse_mat(size_t const m, size_t const n) : m_(m), n_(n), nnz_(0) {
    data_.resize(m);
  }

  sparse_mat(size_t const m, size_t const n, std::span<size_t> x,
             std::span<size_t> y, std::span<T> z)
      : m_(m), n_(n) {

    // Input checking
    assert(x.size() == y.size());
    assert(y.size() == z.size());

    nnz_ = x.size();
    data_.resize(m);

    //
    for (size_t i = 0; i < nnz_; i++) {
      data_[x[i]][y[i]] = z[i];
    }
  }

  void print() { std::cout << *this << std::endl; }

  friend std::ostream &operator<<(std::ostream &out, sparse_mat const &me) {
    out << "sparse_mat:\n\ttype: " << typeid(T).name();
    out << "\n\tm: " << me.m_ << "   n: " << me.n_ << "   nnz: " << me.nnz_;
    out << std::endl;

    for (size_t r = 0; r < me.m_; r++) {
      for (auto it = me.data_[r].begin(); it != me.data_[r].end(); ++it) {
        out << "\t" << r << " " << it->first << " " << it->second << std::endl;
      }
    }
    return out;
  }
};