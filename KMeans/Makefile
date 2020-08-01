NVCC := nvcc
CXX := g++

TARGETS := kmeans cpu_kmeans_opt
SOURCES := kmeans.cu map_reduce.cu
DEF_HEADER := random_generator.hpp
HEADERS := config.cuh $(DEF_HEADER)
CPU_SOURCES := cpu_kmeans.cpp
OBJECTS := $(patsubst %.cu,%.o,$(SOURCES))

all: $(TARGETS)

kmeans: $(SOURCES) $(HEADERS)
	$(NVCC) -O3 -dc $(SOURCES)
	$(NVCC) -O3 -o $@ $(OBJECTS)

cpu_kmeans: $(CPU_SOURCES) $(DEF_HEADER)
	$(CXX) -o $@ $(CPU_SOURCES)

cpu_kmeans_opt: $(CPU_SOURCES) $(DEF_HEADER)
	$(CXX) -O3 -o $@ $(CPU_SOURCES)

clean:
	rm -f *.o $(TARGETS) cpu_kmeans
