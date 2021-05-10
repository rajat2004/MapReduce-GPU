# MapReduce-GPU

Course Project for CS6023: GPU Programming course at IIT Madras

This contains implementation of a few algorithms following Map-Reduce framework, written using CUDA & Thrust library. Algorithms implemented -

1. [K Means](https://github.com/rajat2004/MapReduce-GPU/tree/master/KMeans)
2. [Monte Carlo Pi Estimation](https://github.com/rajat2004/MapReduce-GPU/tree/master/Pi-Estimation) (GPU) - ([CPU codes](https://github.com/rajat2004/MapReduce-GPU/tree/master/CPU))
3. [Sorting Random numbers in bins](https://github.com/rajat2004/MapReduce-GPU/tree/master/Random-Bins)

Results are described in the individual folders, this gives a general overview.

GPUs are great for running computations in parallel, which is similar to the premise of Map Reduce programming model. Map Reduce is used for processing large datasets with a parallel, distributed algorithm on a cluster.

The main implementation of Map-Reduce is in the `map_reduce.cu` file in the example directory, and contains a generic implementation of the model. Main steps in Map-Reduce framework -

1. **Map**: The `mapper` runs in parallel on the input data, and generates key-value pairs. Multiple input elements can be processed by the same thread, and no two threads process the same data.

2. **Shuffle**: This step sorts the Key-Value pairs based on the key, in preparation of the Reduce step where the reducer combines the values for the specific key. Thrust is used for sorting the pairs.

3. **Reduce**: All the values for the specific key are processed to generate the final output.

Implementing these in CUDA directly without following Map-Reduce framework, and with optimizations would give better results. This was just a try to see whether it's possible to implement in such a manner and what kind of results can be obtained.

This project was tested on a system with the following configuration -

OS: Ubuntu 18.04\
CUDA: 10.0\
GPU: Nvidia GTX 1060 MaxQ\
CPU: i7-8750H (6 cores, 12 threads)

**Note**:: In some system configurations, the compilation can fail due to AVX512 related errors. In that case, try removing the -O3 flag from the Makefiles.

References:

- https://github.com/gavyaggarwal/GPU-MapReduce
- Various GFG articles, SO pages
- https://github.com/aditya1601/kmeans-clustering-cpp
