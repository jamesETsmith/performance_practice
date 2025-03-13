CXX = clang++
CXX_FLAGS = -fopenmp -g -O3 -Wall -march=native -std=c++17
CXX_INCLUDE = ./


v0: v0.cpp
	$(CXX) $(CXX_FLAGS) -S -I=$(CXX_INCLUDE) v0.cpp
	$(CXX) $(CXX_FLAGS) -o $@ -I=$(CXX_INCLUDE) v0.cpp

v1: v1.cpp
	$(CXX) $(CXX_FLAGS) -S -I=$(CXX_INCLUDE) v1.cpp
	$(CXX) $(CXX_FLAGS) -o $@ -I=$(CXX_INCLUDE) v1.cpp

v2: v2.cpp
	$(CXX) $(CXX_FLAGS) -S -I=$(CXX_INCLUDE) v2.cpp
	$(CXX) $(CXX_FLAGS) -o $@ -I=$(CXX_INCLUDE) v2.cpp

v3: v3.cpp
	$(CXX) $(CXX_FLAGS) -S -I=$(CXX_INCLUDE) v3.cpp
	$(CXX) $(CXX_FLAGS) -o $@ -I=$(CXX_INCLUDE) v3.cpp

clean:
	rm -rf v0 v1 v2 v3 *.s