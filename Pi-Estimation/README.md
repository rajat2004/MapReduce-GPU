## Monte Carlo Pi Estimation using Map-Reduce

A generic Map-Reduce template implemented in Cuda & applied to Pi Estimation.

This follows the Monte Carlo method of estimating Pi, where many random (x,y) points from [-1.0,1.0] are generated, and 4.0 * (the number of points which lie within a unit circle (i.e. distance from origin < 1) ) / (Total number of points) gives estimated value of Pi. More clear explanation of how it works can be found in many places, such as [this](https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/)

In this implementation, random numbrs are generated on CPU, then copied to GPU. Mapper runs over the numbers, classifying whether they lie inside the circle or not. Reducer sums the number of points inside the circle, and calculates pi.

However, this is an inefficient implementation. Only a single key is used for all the pairs, so the sorting step is useless, and the reducer runs sequentially over the entire data. This is also memory intensive, since lots of numbers are generated on CPU and then copied to GPU. A more customized impelementation is present in `v2` folder which uses `thrust` directly to sum up the values.

#### Results

```shell
$ ./pi_estimation
Generating 100000000 Test Points
Value of Pi: 3.141466720000
Time for CPU data gen: 2003 milliseconds
Time for map reduce (+free): 22936 milliseconds
Total time: 24940 milliseconds
```

Not a positive result, it runs much slower than normal CPU code, and therefore OpenMP parallelized code as well. Also, we probably don't want to increase it too much beyond 10^8 points, 10^9 is about 1GB points (16 bytes each), so will crash your system due to excessive memory usage.

CPU codes & results can be seen at [`CPU/`](https://github.com/rajat2004/MapReduce-GPU/tree/master/CPU)


10^8 points -

CPU (Sequential) -

```
$ ./sequential_pi
Sequential version:
number of samples: 100000000
real Pi: 3.141592653589...
approx Pi: 3.14165924
Time: 9796 milliseconds
```

OpenMP -

```
$ ./openmp_pi
Parallel (OpenMP) version:
Number of samples: 100000000 Threads: 12
Real Pi: 3.141592653589...
Approx Pi: 3.14144088
Time: 1981 milliseconds
```
