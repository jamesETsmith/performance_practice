#include <algorithm>
#include <iostream>
#include <list>
#include <mutex>
#include <thread>

std::list<int> some_list;
std::mutex some_mutex;

void add_to_list(int new_value) {
  std::lock_guard<std::mutex> guard(some_mutex);
  some_list.push_back(new_value);
}

void list_contains(int value_to_find) {
  std::lock_guard<std::mutex> guard(some_mutex);
  bool res = std::find(some_list.begin(), some_list.end(), value_to_find) !=
             some_list.end();
  std::cout << res << std::endl;
}

int main() {

  std::thread t1(add_to_list, 1);
  std::thread t2(list_contains, 1);

  t1.join();
  t2.join();

  return 0;
}
