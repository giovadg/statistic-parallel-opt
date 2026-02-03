# statistic-parallel-opt
Fast C++ library for large-scale time-series feature extraction (rolling statistics), with explicit pthread parallelism,  benchmarking against serial baselines and benchmarking against Pandas.

> **Status:** Ongoing project. Core execution engine and first numerical kernels are implemented; additional kernels and features are actively being developed.

## Requirements
This project requires the boost library for random number generation

## Overview

This project implements a small but complete **numerical analytics engine** for time-series data, focusing on:

- Quality of the computation (correlation is computed with Welford-style algo to avoid cancellation problems)
- Speed: dynamic programming (DP) is used
- explicit pthread-based parallelism (no OpenMP)
- predictable performance and low overhead
- modularity: clean separation between execution engine and numerical kernels
- reproducible benchmarking with a Notebook
  - scaling as function of vector size
  - scaling as function of rolling window size
  - scaling as function of N threads used
  - numerical outputs benchmark agains pandas
  - time-to-solution comparison with pandas

The data can be either internally generated with boost library or imported with .csv or .bin format.

The core use case is repeated feature extraction (rolling statistics) on very large arrays (10⁷–10⁸ points), where Python-level implementations become a bottleneck.

---

## Features

- Custom **pthread thread pool**
  - persistent worker threads
  - clean shutdown and join
- `parallel_for` abstraction for range-based kernels
- Numerical kernels:
  - rolling mean (serial and parallel)
  - rolling variance (serial and parallel)
  - rolling correlation (serial and parallel) with double implementation:
    -  simple DP corralation function
    -  Welford-style remove/add update on centered residuals DP algo.
    
- Deterministic results independent of thread count
- Standalone benchmark executable
- Notebook to compare time-to-solution for different threads and coparison against Pandas.

  ## Time-series input
  - Time series can be either generated internally or imported.
    ### Internal generation
    - For internally generate data the user can specify (otherwise default values) number of elements, number of time-series, length of the rolling window, feature required (ie. mean , var, corr).\
      `./test.exe n_vect=2 n=100000 w=100 num_threads=2 do_corr=1`
      
    ### Inported data
    - Time series can be inported by specifying the name of the file\
      `./test.exe path=./file.bin w=100 num_threads=2 do_corr=1`


  ## Input description:
  #TODO    

---
