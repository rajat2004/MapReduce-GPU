## K-Means Clustering

Modified version of the Map-Reduce framework used in the other two examples to implement K-Means
It's modified to run the steps for a specific number of iterations, and also pass the centroids (which are outputs) to the mapper kernel as well, since they're required to generate the mapping.

There are 2 implementations, a GPU Map-Reduce one - [`kmeans.cu`](kmeans.cu) as well as a CPU one - [`cpu_kmeans.cpp`](cpu_kmeans.cpp) for comparision
Some datasets are already present in the `datasets/` folder, downloaded from - http://cs.joensuu.fi/sipu/datasets/

KMeans generally have a tolerance parameter, to stop when the chnage goes below that. Currently, this hasn't been implemented, and it runs for the specified `ITERATIONS`.

Values are set for using `s*.txt` datasets from the above link. To use a different dataset like `birch`, you'll need to change some constants defined in `config.cuh` (for GPU) & `cpu_kmeans.cpp`.
Relevant fields are `NUM_INPUT` (lines in text file), `NUM_OUTPUT` (K or clusters), `DIMENSION` (of a single vector), `ITERATIONS`.

It saves the centroids in a text file with `.output` appended to the input text file, so for `s1.txt`, it's stored as `s1.txt.output`, & for CPU KMeans, it saves as `s1.txt.output.cpu`. This can be turned off by setting `SAVE_TO_FILE` as `false`.

To compile everything - `make`

```shell
$ make
nvcc -O3 -dc kmeans.cu map_reduce.cu
nvcc -O3 -o kmeans kmeans.o map_reduce.o
g++ -O3 -o cpu_kmeans_opt cpu_kmeans.cpp
```

By default, it compiles host code with `-O3` optimization, to build CPU code without optimizations, run `make cpu_kmeans`

### Results

Optimal clusters based on the data aren't achieved, even when using KMeans++ to initialize the centroids. This might require running it multiple times with different initial centroids, and choose the best one from them, however this hasn't been implemented yet.

Since this is a compute-intensive task with nice parallelization possible, and not memory-intensive, GPU shows much better performance than CPU.

On [`s1.txt`](datasets/s1.txt) (5000 vectors of Dimension 2, K=15 clusters, 1000 iterations)-

#### GPU

```shell
$ ./kmeans datasets/s1.txt
Centroids:
566435 483970
575116 345564
...
Time for CPU data loading + initialize: 3 milliseconds
Time for map reduce KMeans + writing outputs + free: 1245 milliseconds
Total time: 1248 milliseconds
```

#### CPU

```shell
$ ./cpu_kmeans_opt datasets/s1.txt
Centroids:
830620 117906
869991 705259
...
Time for CPU data loading + initialize: 3 milliseconds
Time for KMeans + writing output + free: 1458 milliseconds
Total time: 1461 milliseconds
```

Time taken is very similar, but CPU time increases more rapidly than GPU with increase in inputs, dimension & clusters.

For `birch1` and similar datasets, `NUM_INPUT = 100000`, `NUM_OUTPUT = 100`, `DIMENSION` remains unchanged at 2, `ITERATIONS` can be increased or decreased as desired.
On [`birch1.txt`](datasets/birch1.txt) (100,000 vectors of Dimension 2, K=100 clusters, 1000 iterations) -

#### GPU

```shell
$ ./kmeans datasets/birch1.txt
Centroids:
491674 487825
812013 62854
...
Time for CPU data loading + initialize: 73 milliseconds
Time for map reduce KMeans + writing outputs + free: 21894 milliseconds
Total time: 21967 milliseconds
```

#### CPU

```shell
$ ./cpu_kmeans_opt datasets/birch1.txt
Centroids:
239896 63426
135640 250600
...
Time for CPU data loading + initialize: 40 milliseconds
Time for KMeans + writing output + free: 54330 milliseconds
Total time: 54370 milliseconds
```
