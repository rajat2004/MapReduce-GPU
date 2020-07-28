NVCC := nvcc

TARGETS := pi_estimation
SOURCES := pi_estimation.cu map_reduce.cu
OBJECTS := $(patsubst %.cu,%.o,$(SOURCES))
HEADERS := config.cuh random_generator.hpp

all: $(TARGETS)

pi_estimation: $(SOURCES) $(HEADERS)
	$(NVCC) -O3 -dc $(SOURCES)
	$(NVCC) -O3 -o $@ $(OBJECTS)

# release: $(SOURCES) $(HEADERS)
# 	$(NVCC) -O3 -dc $(SOURCES)
# 	$(NVCC) -O3 -o $@ $(OBJECTS)

clean:
	rm -f *.o $(TARGETS)

run: $(TARGETS)
	./pi_estimation
