#include <algorithm>
#include <iostream>
#include <thread>
#include <vector>

void do_work(size_t i) {
  std::cout << "I'm doing work on thread " << i << std::endl;
}

void f() {
  std::vector<std::thread> threads;
  for (size_t i = 0; i < 20; ++i) {
    threads.push_back(std::thread(do_work, i));
  }

  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));
}

int main() {

  f();

  return 0;
}