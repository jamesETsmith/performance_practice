CXX=g++
OPT_FLAGS=-O3 -march=native -ffast-math -funroll-loops

benchmark: benchmark.cpp
	${CXX} -o benchmark benchmark.cpp ${OPT_FLAGS}

clean:
	rm -f benchmark
