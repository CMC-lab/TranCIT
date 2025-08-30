# Troubleshooting Guide

This guide helps you resolve common issues when using the TranCIT: Transient Causal Interaction Toolbox.

## Installation Issues

### ImportError: No module named 'trancit'

**Problem**: Package not installed or not in Python path.

**Solution**:
```bash
# Install from PyPI
pip install trancit

# Or install from source
git clone https://github.com/CMC-lab/TranCIT.git
cd TranCIT
pip install -e .
```

### Version conflicts with dependencies

**Problem**: Incompatible versions of NumPy, SciPy, or other dependencies.

**Solution**:
```bash
# Update dependencies
pip install --upgrade numpy scipy matplotlib scikit-learn

# Or create a clean environment
python -m venv trancit_env
source trancit_env/bin/activate  # On Windows: trancit_env\Scripts\activate
pip install trancit
```

## Runtime Issues

### ValueError: negative dimensions are not allowed

**Problem**: Time series too short for the configured parameters.

**Solution**:
- Increase `T` (total time points) to at least 500
- Reduce burn-in period in `generate_signals()`
- Check your data dimensions match expected format

### SingularMatrixError during VAR estimation

**Problem**: Insufficient data or highly correlated variables.

**Solution**:
- Increase number of trials (`Ntrial`)
- Reduce model order in BIC selection
- Check for constant or highly correlated time series
- Add small amount of noise to break perfect correlations

### MemoryError with large datasets

**Problem**: Insufficient memory for large time series analysis.

**Solution**:
- Process data in smaller chunks
- Reduce number of trials processed simultaneously
- Use `gc.collect()` between analyses
- Consider using a machine with more RAM

## Configuration Issues

### Pipeline fails during event detection

**Problem**: No events detected with current threshold settings.

**Solution**:
```python
# Lower the detection threshold
config.detection.thres_ratio = 2.0  # instead of 5.0

# Or provide manual event locations
config.detection.locs = np.array([100, 200, 300])  # your event times
config.options.detection = False  # disable automatic detection
```

### BIC selection returns unreasonable model orders

**Problem**: BIC selects very high or very low model orders.

**Solution**:
```python
# Adjust BIC parameters
config.bic.momax = 8  # reduce maximum model order
config.bic.morder = 3  # set reasonable default
config.bic.mode = 'unbiased'  # try different BIC mode
```

## Data Format Issues

### Wrong input data shape

**Problem**: Data not in expected format (n_vars, n_obs, n_trials).

**Solution**:
```python
# Reshape your data
if data.ndim == 2:  # (n_obs, n_vars) -> (n_vars, n_obs, 1)
    data = data.T[..., np.newaxis]
elif data.shape[0] > data.shape[1]:  # probably (n_obs, n_vars, n_trials)
    data = data.transpose(1, 0, 2)

print(f"Data shape: {data.shape}")  # Should be (n_vars, n_obs, n_trials)
```

### NaN or infinite values in data

**Problem**: Data contains NaN or infinite values.

**Solution**:
```python
# Clean your data
data = np.nan_to_num(data, nan=0.0, posinf=1e10, neginf=-1e10)

# Or remove problematic trials
valid_trials = ~np.any(np.isnan(data) | np.isinf(data), axis=(0,1))
data = data[:, :, valid_trials]
```

## Performance Issues

### Analysis running too slowly

**Problem**: Long computation times for large datasets.

**Solutions**:
- Disable bootstrap analysis if not needed: `config.options.bootstrap = False`
- Use OLS instead of ML estimation: `config.causal.estim_mode = 'OLS'`
- Reduce model order: `config.bic.momax = 5`
- Process fewer trials at once

### Memory usage growing over time

**Problem**: Memory not being released between analyses.

**Solution**:
```python
import gc

# Clear memory between analyses
for i in range(n_analyses):
    result = analyzer.analyze(data[i])
    # Process result...
    del result
    gc.collect()
```

## API Transition Issues

### ImportError: cannot import name 'snapshot_detect_analysis_pipeline'

**Problem**: Using old function-based API that has been removed.

**Solution**: Update to new class-based API:
```python
# Old way (deprecated)
# from trancit import snapshot_detect_analysis_pipeline
# results, config, snapshots = snapshot_detect_analysis_pipeline(...)

# New way
from trancit import PipelineOrchestrator
orchestrator = PipelineOrchestrator(config)
result = orchestrator.run(original_signal, detection_signal)
```

### Function signatures have changed

**Problem**: Old function calls with different parameters.

**Solution**: Check the API documentation and examples for updated signatures.

## Plotting and Visualization Issues

### Matplotlib not displaying plots

**Problem**: Plots not showing in Jupyter notebooks or scripts.

**Solution**:
```python
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

# For Jupyter notebooks
%matplotlib inline

# To show plots in scripts
plt.show()
```

### Font or rendering issues

**Problem**: Plots have rendering problems or missing fonts.

**Solution**:
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
# Or try: matplotlib.use('TkAgg')
```

## Getting Help

If you encounter issues not covered here:

1. **Check GitHub Issues**: `https://github.com/CMC-lab/TranCIT/issues`
2. **Create a New Issue**: Include:
   - Your operating system and Python version
   - TranCIT version (`import trancit; print(trancit.__version__)`)
   - Complete error message and traceback
   - Minimal code example that reproduces the issue
3. **Check Documentation**: `https://trancit.readthedocs.io`
4. **Contact**: `salr.nouri@gmail.com` for urgent issues

## Quick Debugging Checklist

When you encounter an error:

- [ ] Check data shape and format
- [ ] Verify all required parameters are provided
- [ ] Ensure data doesn't contain NaN or infinite values
- [ ] Check if time series is long enough for analysis
- [ ] Verify configuration parameters are reasonable
- [ ] Try with a smaller subset of data first
- [ ] Check Python and package versions
- [ ] Review the error message carefully for specific hints

---

**Last Updated**: Version 0.1.0
**For More Help**: See our [documentation](https://trancit.readthedocs.io) or [GitHub issues](https://github.com/CMC-lab/TranCIT/issues).
