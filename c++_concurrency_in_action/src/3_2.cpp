#include <algorithm>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

class some_data {
  int a;
  std::string b;

public:
  void do_something() { a++; }
};

class data_wrapper {
private:
  some_data data;
  std::mutex m;

public:
  template <typename Function> void process_data(Function func) {
    std::lock_guard<std::mutex> l(m);
    func(data);
  }
};

some_data *unprotected;
void malicious_function(some_data &protected_data) {
  unprotected = &protected_data;
}
data_wrapper x;
void foo() {
  x.process_data(malicious_function);
  unprotected->do_something();
}

int main() {

  std::vector<std::thread> threads(100);

  for (int i = 0; i < 100; i++) {
    threads[i] = std::thread(foo);
  }

  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));
}