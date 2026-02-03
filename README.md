# statistic-parallel-opt
Fast C++ library for large-scale time-series feature extraction (rolling statistics), with explicit pthread parallelism,  benchmarking against serial baselines and benchmarking against Pandas.

> **Status:** Ongoing project. Core execution engine and first numerical kernels are implemented; additional kernels and features are actively being developed.

## Requirements
This project requires the boost library for random number generation

## Overview

This project implements a small but complete **numerical analytics engine** for time-series data, focusing on:

- Speed: dynamic programming (DP) is used
- explicit pthread-based parallelism (no OpenMP)
- predictable performance and low overhead
- modularity: clean separation between execution engine and numerical kernels
- reproducible benchmarking with a Notebook
  - scaling as function of vector size
  - scaling as function of rolling window size
  - scaling as function of N threads used
  - numerical resutls benchmark agains pandas
  - time-to-solution comparison with pandas

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

---
