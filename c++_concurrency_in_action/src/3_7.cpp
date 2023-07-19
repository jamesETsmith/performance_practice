#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

class hierarchical_mutex {
  std::mutex internal_mutex;
  size_t const hierarchy_value;
  // Need to keep track of previous value for walking back up/out of mutex
  // hierarchy
  size_t previous_hierarchy_value;
  // Global tracker of where we are in the hierarchy as we work our way down
  static thread_local size_t this_thread_hierarchy_value;

  void check_for_hierarchy_violation() {
    if (this_thread_hierarchy_value <= hierarchy_value) {
      throw std::logic_error("mutex hierarchy violated");
    }
  }

  void update_hierarchy_value() {
    previous_hierarchy_value = this_thread_hierarchy_value;
    this_thread_hierarchy_value = hierarchy_value;
  }

public:
  explicit hierarchical_mutex(size_t value)
      : hierarchy_value(value), previous_hierarchy_value(0) {}

  void lock() {
    check_for_hierarchy_violation();
    internal_mutex.lock();
    update_hierarchy_value();
  }

  void unlock() {
    this_thread_hierarchy_value = previous_hierarchy_value;
    internal_mutex.unlock();
  }

  bool try_lock() {
    check_for_hierarchy_violation();
    if (!internal_mutex.try_lock()) {
      return false;
    }
    update_hierarchy_value();
    return true;
  }
};

thread_local size_t hierarchical_mutex::this_thread_hierarchy_value(SIZE_MAX);

hierarchical_mutex high_level_mutex(10000);
hierarchical_mutex low_level_mutex(5000);

int low_level_thing = 0;
int high_level_thing = 0;

int do_low_level_stuff() { return low_level_thing++; };

int low_level_func() {
  std::lock_guard<hierarchical_mutex> ll(low_level_mutex);
  return do_low_level_stuff();
}

void high_level_stuff(int some_param) { high_level_thing += some_param; }
void high_level_func() {
  std::lock_guard<hierarchical_mutex> hl(high_level_mutex);
  high_level_stuff(low_level_func());
}

void thread_a() { high_level_func(); }

//
// Anit-pattern below
//
hierarchical_mutex other_mutex(100);
int lowest_level_stuff = 0;
void do_other_stuff() { lowest_level_stuff++; }
void other_stuff() {
  high_level_func();
  do_other_stuff();
}

// Bad code
void thread_b() {
  // locks lowest level data
  std::lock_guard<hierarchical_mutex> ll(other_mutex);

  // This function calls high_level_stuff() which violates the hierarchy
  // may throw exception or abort the program. Deadlocks between hierarchical
  // mutexes are impossible because the mutexes themselves enforce the ordering.
  other_stuff();
}

int main()

{
  std::thread t1(thread_a);
  std::thread t2(thread_b);

  t1.join();
  t2.join();

  return 0;
}