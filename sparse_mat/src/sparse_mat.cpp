#include "sparse_mat.hpp"

int main() {
  sparse_mat<uint32_t> sp(10, 10);
  sp.print();

  std::vector<size_t> x{0, 1, 2, 3}, y{0, 1, 2, 3};
  std::vector<double> z{3.2, 2.1, -99., 1.3};
  sparse_mat<double> sp2(4, 4, x, y, z);
  std::cout << sp2 << std::endl;

  return 0;
}