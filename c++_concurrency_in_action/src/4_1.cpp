#include <condition_variable>
#include <queue>
#include <threads>
#include <vector>

std::mutex mut;
// 1) share data between threads with queue
std::queue<data_chunk> data_queue;
std::condition_variable data_cond;

void data_preparation_thread() {
  while (more_data_to_prepare()) {
    data_chunk const data = prepare_data();

    // 2) Since queue is shared we need to make sure we don't push two at once
    std::lock_guard<std::mutex> lk(mut);
    data_queue.push(data);
    data_cond.notify_one();
  }
}

void data_processing_thread() {
  while (true) {
    // 4) Need unique lock so we can lock while we check the condition and
    // unlock while we wait
    std::unique_lock<std::mutex> lk(mut);

    // 5) Tell this thread to wait until the data_queue is non-empty
    data_cond.wait(lk, [] { return !data_queue.empty(); });
    data_chunk data = data_queue.front();
    data_queue.pop();

    lk.unlock();
    process(data);
    if (is_last_chunk(data)) {
      break;
    }
  }
}

int main() {

  //
  //

  return 0;
}
