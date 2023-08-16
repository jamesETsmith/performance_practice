#include <memory>
#include <new>

namespace my_stl {

/*
| T | T | T | T | T | T | ? | ? | ? |
^                       ^           ^
  first                   last        end
*/

template <typename T>
struct vector {
  T* first = nullptr;
  T* last = nullptr;
  T* end = nullptr;

  vector() = default;
  vector(size_t size) {
    first = new T[size];
    last = first + size;
    end = first + size;
  }

  ~vector() {
    if (first != nullptr) {
      delete[] first;
    }
  }

  size_t size() const { return last - first; }
  size_t capacity() const { return end - first; }

  T& operator[](size_t pos) { return *(first + pos); }

  void push_back(T const& e) {
    size_t const old_size = this->size();
    size_t const old_cap = this->capacity();

    if (old_size == old_cap) {
      T* new_first = new T[old_cap * 2];
      std::memcpy(new_first, first, sizeof(T) * old_cap);
      delete[] first;
      first = new_first;
      last = new_first + old_size;
      end = first + old_cap * 2;
    }

    *(first + old_cap) = e;
    last++;
  }
};

}  // namespace my_stl
