#include <algorithm>
#include <future>
#include <iostream>
#include <list>
#include <thread>
#include <vector>

template <typename T>
std::list<T> sequential_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }

  std::list<T> result;
  result.splice(result.begin(), input, input.begin());
  T const& pivot = *result.begin();

  auto divide_point = std::partition(input.begin(), input.end(),
                                     [&](T const& t) { return t < pivot; });
  std::list<T> lower_part;
  lower_part.splice(lower_part.end(), input, input.begin(), divide_point);

  auto new_lower = sequential_quick_sort(std::move(lower_part));
  auto new_higher = sequential_quick_sort(std::move(input));
  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower);

  return result;
}

template <typename T>
std::list<T> parallel_quick_sort(std::list<T> input) {
  if (input.empty()) {
    return input;
  }
  std::list<T> result;
  result.splice(result.begin(), input, input.begin());
  T const& pivot = *result.begin();

  auto divide_point = std::partition(input.begin(), input.end(),
                                     [&](T const& t) { return t < pivot; });
  std::list<T> lower_part;
  lower_part.splice(lower_part.end(), input, input.begin(), divide_point);

  std::future<std::list<T> > new_lower(
      std::async(&parallel_quick_sort<T>, std::move(lower_part)));
  auto new_higher(parallel_quick_sort(std::move(input)));

  result.splice(result.end(), new_higher);
  result.splice(result.begin(), new_lower.get());

  return result;
}

template <typename T>
bool check_seq(std::list<T>& l) {
  bool res = true;

  std::list<T> trusted = l;
  //   std::sort(trusted.begin(), trusted.end());
  trusted.sort();

  std::list<T> my_list = sequential_quick_sort(l);

  auto it_trusted = trusted.begin();
  auto it_my = my_list.begin();

  while (it_trusted != trusted.end() || it_my != my_list.end()) {
    res &= (*it_trusted == *it_my);
    it_trusted = std::next(it_trusted);
    it_my = std::next(it_my);
  }

  if (res) {
    std::cout << "[INFO]: sequential test passed" << std::endl;
  } else {
    std::cerr << "[ERROR]: sequential test failed" << std::endl;
  }

  return res;
}

template <typename T>
bool check_parallel(std::list<T>& l) {
  bool res = true;

  std::list<T> trusted = l;
  //   std::sort(trusted.begin(), trusted.end());
  trusted.sort();

  std::list<T> my_list = parallel_quick_sort(l);

  auto it_trusted = trusted.begin();
  auto it_my = my_list.begin();

  while (it_trusted != trusted.end() || it_my != my_list.end()) {
    res &= (*it_trusted == *it_my);
    it_trusted = std::next(it_trusted);
    it_my = std::next(it_my);
  }

  if (res) {
    std::cout << "[INFO]: parallel test passed" << std::endl;
  } else {
    std::cerr << "[ERROR]: parallel test failed" << std::endl;
  }

  return res;
}

int main() {
  std::list<int> l = {
      45, 57, 85, 25, 58, 33, 80, 85, 44, 18, 11, 11, 56, 3,  30, 19, 68,
      69, 61, 54, 60, 18, 93, 63, 96, 53, 51, 45, 78, 46, 93, 5,  81, 1,
      37, 46, 6,  88, 48, 38, 26, 13, 70, 70, 15, 33, 35, 21, 35, 96, 78,
      93, 65, 11, 74, 58, 9,  16, 92, 96, 68, 51, 23, 87, 20, 52, 93, 100,
      45, 86, 70, 9,  73, 80, 69, 57, 53, 72, 13, 21, 40, 52, 59, 86, 82,
      26, 44, 2,  35, 14, 72, 57, 85, 32, 54, 60, 15, 32, 43, 12,
  };
  check_seq(l);
  check_parallel(l);

  return 0;
}