## Monte Carlo Pi Estimation using Map-Reduce (V2)

Customized Map-Rduce implementation. Improvements -

1. Replace key-value pairs with a single array of `bool` and use `thrust` for summation reduction

What this does is it removes the unnecessary sorting step (single key for all), and therefore the sequential reduce step running over all the pairs. The results from the mapping phase are stored in a boolean array (no need of keys), `true` if point lies inside circle, `false` otherwise. This also has the benefit of reducing a lot of code.

2. Use `cuRAND` to generate points on GPU itself

After the first improvement, it takes more time to generate data on CPU than the rest of computation. So, let's try to reduce that.
This however results in more time taken due to cuRAND initialization. But a different benefit is it allows using 10^9 samples, since only output `bool` array is allocated on GPU, which can also be reduced to a single number per thread.

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
Faster than sequential CPU code, but a bit slower than OpenMP.

After testing out 2nd improvement -

```
$ ./pi_estimation
Value of Pi: 3.141561520000
Time for CPU data gen: 0 milliseconds
Time for map reduce (+free): 5966 milliseconds
Total time: 5966 milliseconds
```

`nvprof` gives more detailed insights and shows that most of the time is taken in `setup_kernel` (abut 99% actually). Therefore haven't added this.

For 10^9 samples -

```shell
$ ./pi_estimation
Value of Pi: 3.141633528000
No. of Samples: 1000000000
Time for CPU data gen: 0 milliseconds
Time for map reduce (+free): 6768 milliseconds
Total time: 6768 milliseconds
```

Only slower than OpenMP with O3 optimization! Increasing to 10^10 gives memory error in cuRAND initialization, but still, pretty nice. Probably using simple `rand()` won't have this limitation.
