# Matrix Multiplication Example

All code snippets were taken from the [Algorithmica](https://en.algorithmica.org/hpc/algorithms/matmul/) website.

## Build and Testing
```
make
./benchmark # select a subset with --filter=matmul.<subset>
```

```
 ./benchmark
[==========] Running 7 benchmarks.
[ RUN      ] matmul.v0
[       OK ] matmul.v0 (mean 3.573s, confidence interval +- 2.433989%)
[ RUN      ] matmul.v1
[       OK ] matmul.v1 (mean 1.250s, confidence interval +- 0.491429%)
[ RUN      ] matmul.v2_manual
[       OK ] matmul.v2_manual (mean 1.256s, confidence interval +- 1.083451%)
[ RUN      ] matmul.v2
[       OK ] matmul.v2 (mean 1.225s, confidence interval +- 0.778146%)
[ RUN      ] matmul.v3
[       OK ] matmul.v3 (mean 718.140ms, confidence interval +- 0.497365%)
[ RUN      ] matmul.v4
[       OK ] matmul.v4 (mean 342.708ms, confidence interval +- 0.285434%)
[ RUN      ] matmul.v5
[       OK ] matmul.v5 (mean 147.595ms, confidence interval +- 0.712627%)
[==========] 7 benchmarks ran.
[  PASSED  ] 7 benchmarks.
```