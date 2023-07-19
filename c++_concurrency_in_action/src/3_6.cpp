#include <lock>
#include <thread>
#include <vector>

class some_big_object;

void swap(some_big_object &lhs, some_big_object &rhs);

class X {
private
  some_big_object some_detail;
  std::mutex m;

public:
  X(some_big_object const &sd) : some_detail(sd) {}

  friend void swap(X &lhs, X &rhs) {
    // 0) Make sure we aren't dealing with the same instance
    if (&lhs == &rhs) {
      return;
    }

    // 1) Lock both std::mutex at once
    std::lock(lhs.m, rhs.m);
    // 2)/3) create std::lock_guard and specify the locks are already lock with
    // std::adopt_lock
    std::lock_guard<std::mutex> lock_a(lhs.m, std::adopt_lock);
    std::lock_guard<std::mutex> lock_b(rhs.m, std::adopt_lock);
    swap(lhs.some_detail, rhs.some_detail);
  }
};