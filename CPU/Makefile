CXX := g++

SEQUENTIAL := sequential_pi.cpp
SEQ_BIN := sequential_pi
OPENMP := openmp_pi.cpp
OPENMP_BIN := openmp_pi
DEPS := random_generator.hpp

TARGETS := $(SEQ_BIN) $(OPENMP_BIN)


all: $(TARGETS)

$(SEQ_BIN): $(SEQUENTIAL) $(DEPS)
	$(CXX) -o $@ $(SEQUENTIAL)

$(OPENMP_BIN): $(OPENMP) $(DEPS)
	$(CXX) -fopenmp -o $@ $(OPENMP)

release: $(SEQUENTIAL) $(OPENMP) $(DEPS)
	$(CXX) -O3 -o $(SEQ_BIN) $(SEQUENTIAL)
	$(CXX) -O3 -fopenmp -o $(OPENMP_BIN) $(OPENMP)

clean:
	rm $(TARGETS)
