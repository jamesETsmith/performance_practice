CXX=g++
OPT_FLAGS=-O3 -std=c++17 -march=skylake -ffast-math -funroll-loops -Wall

benchmark: benchmark.cpp
	${CXX} $^ -o $@ ${OPT_FLAGS}

clean:
	rm -f benchmark
