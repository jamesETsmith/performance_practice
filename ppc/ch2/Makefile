CXX = g++
CXX_FLAGS = -fopenmp -g -O3 -march=native -std=c++17
CXX_INCLUDE = ./


v0: v0.cpp
	$(CXX) $(CXX_FLAGS) -S -I=$(CXX_INCLUDE) v0.cpp
	$(CXX) $(CXX_FLAGS) -o $@ -I=$(CXX_INCLUDE) v0.cpp

clean:
	rm main