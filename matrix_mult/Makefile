CXX=g++
OPT_FLAGS=-O3 -std=c++17 -march=native -ffast-math -funroll-loops -Wall

benchmark: benchmark.cpp
	${CXX} -o $@ benchmark.cpp ${OPT_FLAGS}

clean:
	rm -f benchmark
