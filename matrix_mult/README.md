# Matrix Multiplication Example

All code snippets were taken from the [Algorithmica](https://en.algorithmica.org/hpc/algorithms/matmul/) website.

## Build and Testing
```
make
./benchmark # select a subset with --filter=matmul.<subset>
```

```
./benchmark
[==========] Running 6 benchmarks.
[ RUN      ] matmul.v0
[       OK ] matmul.v0 (mean 3.482s, confidence interval +- 0.306575%)
[ RUN      ] matmul.v1
[       OK ] matmul.v1 (mean 1.240s, confidence interval +- 0.283036%)
[ RUN      ] matmul.v2_manual
[       OK ] matmul.v2_manual (mean 1.251s, confidence interval +- 0.983899%)
[ RUN      ] matmul.v2
[       OK ] matmul.v2 (mean 1.229s, confidence interval +- 1.332107%)
[ RUN      ] matmul.v3
[       OK ] matmul.v3 (mean 709.670ms, confidence interval +- 1.528031%)
[ RUN      ] matmul.v4
[       OK ] matmul.v4 (mean 337.575ms, confidence interval +- 0.354333%)
[==========] 6 benchmarks ran.
[  PASSED  ] 6 benchmarks.
```