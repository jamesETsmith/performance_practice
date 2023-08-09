#include <algorithm>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

template <typename T> class threadsafe_queue {
private:
  // must be marked as mutable so we can use it in `bool empty() const`;
  mutable std::mutex mut;
  std::queue<T> data_queue;
  std::condition_variable data_cond;

public:
  threadsafe_queue(){};
  threadsafe_queue(threadsafe_queue const &other) {
    std::lock_guard<std::mutex> lk(other.mut);
    data_queue = other.data_queue;
  }
  threadsafe_queue &operator=(threadsafe_queue const &) = delete;

  void push(T new_value) {
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(new_value);
    data_cond.notify_one();
  }

  // Stores the returned value in value and returns flag saying whether or not
  // it actually got the value
  bool try_pop(T &value) {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty()) {
      return false;
    }
    value = data_queue.front();
    data_queue.pop();
    return true;
  }

  // Can't directly the status here, but we can check if the returned pointer is
  // NULL
  std::shared_ptr<T> try_pop() {
    std::lock_guard<std::mutex> lk(mut);
    if (data_queue.empty()) {
      return std::shared_ptr<T>();
    }
    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }

  void wait_and_pop(T &value) {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this] { return !data_queue.empty(); });
    value = data_queue.front();
    data_queue.pop();
  }
  std::shared_ptr<T> wait_and_pop() {
    std::unique_lock<std::mutex> lk(mut);
    data_cond.wait(lk, [this] { return !data_queue.empty(); });
    std::shared_ptr<T> res(std::make_shared<T>(data_queue.front()));
    data_queue.pop();
    return res;
  }

  bool empty() const {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.empty();
  }

  size_t size() const {
    std::lock_guard<std::mutex> lk(mut);
    return data_queue.size();
  }
};

int main() {

  threadsafe_queue<double> q;

  std::vector<std::thread> threads(1000);

  std::for_each(threads.begin(), threads.end(), [&](std::thread &th) {
    th = std::thread([&]() { q.push(-1.1); });
  });

  std::for_each(threads.begin(), threads.end(),
                std::mem_fn(&std::thread::join));

  std::cout << q.size() << std::endl;

  return 0;
}