#include <thread>

void do_something(unsigned &j) { unsigned k = j + j; }

struct func {
  int &i;
  func(int &i_) : i(i_) {}

  void operator()() {
    for (unsigned j = 0; j < 1000000; ++j) {
      do_something(j);
    }
  }
};

void oops() {
  int some_local_state = 0;
  func my_func(some_local_state);
  std::thread my_thread(my_func);
  my_thread.detach(); // dangerous
}

int main() {

  oops();

  return 0;
}