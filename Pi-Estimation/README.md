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

Not a very positive result, it runs much faster than normal CPU code, and bit faster than plain OpenMP parallelized code, however slower than `O3` compiled versions of both of them. These results can be seen at [`CPU/README.md`](https://github.com/rajat2004/MapReduce-GPU/tree/master/CPU)
