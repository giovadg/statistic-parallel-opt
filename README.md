# pthread-parallel-stats
Fast C++ library for large-scale time-series feature extraction (rolling statistics), with explicit pthread parallelism and benchmarking against serial baselines.

> **Status:** Ongoing project. Core execution engine and first numerical kernels are implemented; additional kernels and features are actively being developed.


## Overview

This project implements a small but complete **numerical analytics engine** for time-series data, focusing on:

- explicit pthread-based parallelism (no OpenMP)
- predictable performance and low overhead
- clean separation between execution engine and numerical kernels
- reproducible benchmarking with a Juppyter Nootebook
  - scaling as function of vector size
  - scaling as function of rolling window size
  - scaling as function of N threads used

The core use case is repeated feature extraction (rolling statistics) on very large arrays (10⁷–10⁸ points), where Python-level implementations become a bottleneck.

---

## Features

- Custom **pthread thread pool**
  - persistent worker threads
  - clean shutdown and join
- `parallel_for` abstraction for range-based kernels
- Numerical kernels:
  - rolling mean (serial and parallel)
  - rolling correlation (serial and parallel)
- Deterministic results independent of thread count
- Standalone benchmark executable

---
