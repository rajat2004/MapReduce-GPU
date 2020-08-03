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

| Samples   |      Time      |  Value |
|----------|:-------------:|:------:|
| 10 | 0 | 3.6 |
| 100 | 0 | 3.28 |
| 1000 | 0 | 3.152 |
| 10000 | 0 | 3.1452 |
| 100000 | 9 | 3.14312 |
| 1000000 | 96 | 3.141776 |
| 10000000 | 1014 | 3.1419328 |
| 100000000 | 11029 | 3.1415606 |
| 1000000000 | 146739 | 3.141599232 |

#### Sequential

Normal:

```shell
approx Pi: 3.141586768
Time: 150511 milliseconds
```

`-O3` optimized -

```shell
approx Pi: 3.141575816
Time: 16715 milliseconds
```

#### OpenMP

Number of samples: 1000000000 (1e9) Threads: 12

Normal:

```shell
Approx Pi: 3.141644756
Time: 29769 milliseconds
```

`-O3` optimized:

```shell
Approx Pi: 3.141597632
Time: 2624 milliseconds
```
