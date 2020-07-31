NVCC := nvcc

TARGETS := random_bins
SOURCES := random_bins.cu map_reduce.cu
OBJECTS := $(patsubst %.cu,%.o,$(SOURCES))
HEADERS := config.cuh random_generator.hpp

all: $(TARGETS)

$(TARGETS): $(SOURCES) $(HEADERS)
	$(NVCC) -O3 -dc $(SOURCES)
	$(NVCC) -O3 -o $@ $(OBJECTS)

clean:
	rm -f *.o $(TARGETS)

run: $(TARGETS)
	./$(TARGETS)
