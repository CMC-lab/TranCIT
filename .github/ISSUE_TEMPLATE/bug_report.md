---
name: Bug Report
about: Report a bug or unexpected behavior in TranCIT
title: '[BUG] '
labels: 'bug'
assignees: ''

---

## Bug Description

A clear and concise description of what the bug is.

## Steps to Reproduce

Please provide a minimal reproducible example:

```python
# Your code here that reproduces the issue
import trancit
# ... 
```

**Steps:**

1. Load data with shape: [describe data dimensions]
2. Configure analysis: [describe parameters]
3. Run method: [specify which TranCIT method]
4. Observe error

## Expected Behavior

A clear description of what you expected to happen.

## Actual Behavior

What actually happened instead?

## Error Messages

```
    Paste any error messages, tracebacks, or warnings here
```

## Environment Information

- **TranCIT version**: [e.g., 1.0.10]
- **Python version**: [e.g., 3.11.5]
- **Operating System**: [e.g., macOS 14.0, Ubuntu 22.04, Windows 11]
- **NumPy version**: [e.g., 1.24.3]
- **SciPy version**: [e.g., 1.10.1]

## Data Information

- **Data type**: [e.g., EEG, LFP, simulated signals]
- **Data shape**: [e.g., (channels, time_points, trials)]
- **Sampling rate**: [if applicable]
- **Analysis method**: [DCS, Transfer Entropy, Granger Causality, etc.]

## Additional Context

- Is this related to a specific scientific application?
- Are you following a particular research methodology?
- Any relevant literature or expected results?

## Checklist

- [ ] I have searched existing issues to avoid duplicates
- [ ] I have provided a minimal reproducible example
- [ ] I have included version information
- [ ] I have included data shape/type information
