#include <algorithm>
#include <exception>
#include <memory>
#include <mutex>
#include <stack>
#include <thread>
#include <vector>

struct empty_stack : std::exception {
  const char *what() const throw();
};

template <typename T> class threadsafe_stack {
private:
  std::stack<T> data;
  mutable std::mutex m;

public:
  threadsafe_stack(){};
  threadsafe_stack(const threadsafe_stack &other) {
    std::lock_guard<std::mutex> lock(other.m);
    data = other.data;
  }
  threadsafe_stack &operator=(const threadsafe_stack &) = delete;

  void push(T new_value) {
    std::lock_guard<std::mutex> lock(m);
    data.push(new_value);
  }
  std::shared_ptr<T> pop() {
    std::lock_guard<std::mutex> lock(m);
    if (data.empty())
      throw empty_stack();
    std::shared_ptr<T> const res(std::make_shared<T>(data.top()));
    data.pop();
    return res;
  }

  void pop(T &value) {
    std::lock_guard<std::mutex> lock(m);
    if (data.empty())
      throw empty_stack();

    value = data.top();
    data.pop();
  }

  bool empty() const {
    std::lock_guard<std::mutex> lock(m);
    return data.empty();
  }
};

int main() {

  threadsafe_stack<double> my_stack;

  size_t const N = 12;
  std::vector<std::thread> threads(N);

  for (size_t i = 0; i < N; i++) {
    threads[i] = std::thread([&]() {
      for (size_t j = 0; j < 100; j++) {
        my_stack.push(i * j);
      }
    });
  }

  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));

  return 0;
}