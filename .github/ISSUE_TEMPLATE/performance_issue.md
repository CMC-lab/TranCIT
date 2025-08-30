---
name: Performance Issue
about: Report slow performance or computational efficiency problems
title: '[PERFORMANCE] '
labels: 'performance'
assignees: ''

---

## Performance Issue Type
- [ ] Unexpectedly slow execution
- [ ] Memory usage too high
- [ ] Crashes with large datasets
- [ ] Scaling poorly with data size
- [ ] Method comparison shows inefficiency
- [ ] Other: ________________

## Dataset Characteristics
**Data specifications:**
- **Data shape**: [e.g., (64, 10000, 100) - channels × time_points × trials]
- **Data size**: [e.g., ~500 MB, 2.3 GB]
- **Data type**: [e.g., np.float64, np.float32]
- **Sampling rate**: [e.g., 1000 Hz]
- **Analysis duration**: [e.g., 30 seconds, 10 minutes of data]

## Analysis Configuration
**Method and parameters:**
```python
# Your analysis configuration
from trancit import ...

calculator = ...  # Include your configuration
result = calculator.analyze(data)  # The slow operation
```

**Parameters:**
- **Method**: [DCS, Transfer Entropy, Granger Causality, etc.]
- **Model order**: [e.g., 4, auto-selection]
- **Time mode**: [e.g., inhomo, homo]
- **Bootstrap samples**: [if applicable]

## Performance Measurements
**Timing results:**
- **Execution time**: [e.g., 45 minutes for 10-second analysis]
- **Expected time**: [e.g., Should complete in ~5 minutes based on paper]
- **Memory usage**: [e.g., 32 GB RAM, crashes system]
- **CPU usage**: [e.g., Single core at 100%, multi-core at 25%]

**Measurement method:**
```python
import time
start_time = time.time()
# Your code here
execution_time = time.time() - start_time
```

## Environment Information
- **TranCIT version**: [e.g., 1.0.10]
- **Python version**: [e.g., 3.11.5]
- **Operating System**: [e.g., macOS 14.0, Ubuntu 22.04]
- **Hardware**: [e.g., Intel i7, Apple M2, 32GB RAM]
- **NumPy/SciPy backend**: [e.g., OpenBLAS, Intel MKL]

## Comparison Benchmarks
**Have you tried:**
- [ ] Different parameter settings
- [ ] Smaller subset of data
- [ ] Different time modes
- [ ] Alternative methods for comparison
- [ ] Other causality analysis packages

**Results:** [Describe performance differences if any]

## Expected Performance
**Based on:**
- [ ] Literature benchmarks
- [ ] Previous TranCIT versions
- [ ] Other software packages
- [ ] Theoretical complexity analysis

**Expected behavior:** [e.g., "Should scale O(n²) with number of channels"]

## Reproducible Example
**Minimal code to reproduce the performance issue:**
```python
# Minimal example that demonstrates the slow performance
import numpy as np
from trancit import ...

# Generate test data or load problematic dataset
data = ...

# Configuration that shows the problem
...
```

## Additional Context
- Is this a regression from previous versions?
- Are there specific scientific requirements driving the performance need?
- Would you be interested in profiling or benchmarking contributions?

## Potential Solutions Considered
- [ ] Algorithm optimization
- [ ] Parallel processing
- [ ] Memory management improvements
- [ ] Parameter tuning
- [ ] Alternative mathematical approaches
