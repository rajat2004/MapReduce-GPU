## CPU Implementations

### Monte Carlo Pi Estimation

Useful links explaining the concept & implementation -

- https://www.geeksforgeeks.org/estimating-value-pi-using-monte-carlo/
- http://jakascorner.com/blog/2016/05/monte-carlo-pi.html
- OpenMP usage - http://jakascorner.com/blog/2016/05/omp-monte-carlo-pi.html

Sequential: `g++ sequential_pi.cpp`
Parallel: `g++ -fopenmp openmp_pi.cpp`

Add `-O3` for a huge improvement in results!

Number of samples (and threads also, in case of OpenMP) can be changed in the respective files

Results:

#### Samples vs Value,Time (Sequential)

Samples: 10           Time: 0, Value: 3.6
Samples: 100          Time: 0, Value: 3.28
Samples: 1000         Time: 0, Value: 3.152
Samples: 10000        Time: 0, Value: 3.1452
Samples: 100000       Time: 9, Value: 3.14312
Samples: 1000000      Time: 96, Value: 3.141776
Samples: 10000000     Time: 1014, Value: 3.1419328
Samples: 100000000    Time: 11029, Value: 3.1415606
Samples: 1000000000   Time: 146739, Value: 3.141599232

#### Sequential

Normal:
```
approx Pi: 3.141586768
Time: 150511 milliseconds
```

`-O3` optimized
```
approx Pi: 3.141575816
Time: 16715 milliseconds
```

#### OpenMP

Number of samples: 1000000000 (1e9) Threads: 12

Normal:
```
Approx Pi: 3.141644756
Time: 29769 milliseconds
```

`-O3` optimized:
```
Approx Pi: 3.141597632
Time: 2624 milliseconds
```
