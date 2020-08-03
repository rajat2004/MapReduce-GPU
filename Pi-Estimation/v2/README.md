## Monte Carlo Pi Estimation using Map-Reduce (V2)

Customized Map-Rduce implementation. Improvements -

1. Replace key-value pairs with a single array of `bool` and use `thrust` for summation reduction

What this does is it removes the unnecessary sorting step (single key for all), and therefore the sequential reduce step running over all the pairs. The results from the mapping phase are stored in a boolean array (no need of keys), `true` if point lies inside circle, `false` otherwise. This also has the benefit of reducing a lot of code.

2. Generating random numbers on GPU using `cuRAND`

This didn't give good results, and initializing took much more time than the rest. Implementation and results can be seen in [`pi-curand`](https://github.com/rajat2004/MapReduce-GPU/tree/pi-curand) branch.

#### Results

For reference, using generic Map-Reduce template -

```shell
$ ./pi_estimation
Generating 100000000 Test Points
Value of Pi: 3.141894840000
Time for CPU data gen: 1962 milliseconds
Time for map reduce (+free): 22844 milliseconds
Total time: 24806 milliseconds
```

After 1st optimization -

```shell
$ ./pi_estimation
Generating 100000000 Test Points
Value of Pi: 3.141614960000
Time for CPU data gen: 2018 milliseconds
Time for map reduce (+free): 416 milliseconds
Total time: 2434 milliseconds
```

Wow, that's a huge improvement!
