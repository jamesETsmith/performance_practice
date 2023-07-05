#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/cURLpp.hpp>

// namespace {
// const long MyPort = 80;
// }

std::stringstream download(std::string url) {
  std::stringstream ss;

  try {
    curlpp::Cleanup myCleanup;
    ss << curlpp::options::Url(url);
  }

  catch (curlpp::RuntimeError &e) {
    std::cout << e.what() << std::endl;
  }

  catch (curlpp::LogicError &e) {
    std::cout << e.what() << std::endl;
  }
  return ss;
}

using data_dict = std::unordered_map<std::string, std::stringstream>;

data_dict download_multiple(std::vector<std::string> &urls) {
  data_dict dd;

  for (std::string const &url : urls) {
    dd[url] = download(url);
  }

  return dd;
}

/**
 * This example is made to show you how you can use the Options.
 */

int main(int, char **) {
  std::stringstream ss =
      download("https://en.wikipedia.org/wiki/Sudoku_solving_algorithms");

  std::cout << ss.str() << std::endl;

  std::vector<std::string> urls = {
      "https://en.wikipedia.org/wiki/Mathematics_of_Sudoku",
      "https://en.wikipedia.org/wiki/Backtracking",
      "https://en.wikipedia.org/wiki/Depth-first_search"};

  data_dict dd = download_multiple(urls);

  return 0;
}