# Dynamic Causal Strength (DCS)

![PyPI version](https://img.shields.io/pypi/v/dynamic-causal-strength.svg)
![Python version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://github.com/yourusername/dynamic-causal-strength/actions/workflows/main.yml/badge.svg)

**Dynamic Causal Strength (DCS)** is a Python package for detecting and analyzing causal relationships in time series data. It provides tools for model estimation, causality detection, and simulation, with a focus on dynamic, non-stationary systems.

---

## Table of Contents

- [Dynamic Causal Strength (DCS)](#dynamic-causal-strength-dcs)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Install from PyPI](#install-from-pypi)
    - [Install from Source](#install-from-source)
  - [Quickstart](#quickstart)
  - [File Structure](#file-structure)
    - [Key Modules](#key-modules)
  - [Usage](#usage)
    - [Detecting Causality](#detecting-causality)
    - [Estimating VAR Coefficients](#estimating-var-coefficients)
    - [Simulating Time Series](#simulating-time-series)
  - [Contributing](#contributing)
  - [License](#license)
  - [Contact](#contact)

---

## Features

- **Causality Detection**: Identify causal relationships using methods like Granger causality and transfer entropy.
- **Model Estimation**: Estimate coefficients for Vector Autoregression (VAR) models.
- **Simulation Tools**: Generate synthetic time series data for testing and validation.
- **Non-Stationary Support**: Handle time-varying and non-stationary processes.
- **Modular Design**: Organized into intuitive modules for easy extension and maintenance.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- NumPy 1.21 or higher
- SciPy 1.7 or higher

### Install from PyPI

```bash
pip install dynamic-causal-strength
```

### Install from Source

```bash
git clone https://github.com/yourusername/dynamic-causal-strength.git
cd dynamic-causal-strength
pip install .
```

---

## Quickstart

Here’s a quick example to get you started with detecting causal strength in a simulated dataset.

```python
import numpy as np
from dcs.causality import detect_causality
from dcs.simulation import generate_signals

# Generate synthetic data
data, _, _ = generate_signals(T=1000, Ntrial=10, h=0.1, gamma1=0.5, gamma2=0.5, Omega1=1, Omega2=1)

# Detect causality with lag=2
...
```

For more examples, see the [examples/](examples/) directory.

---

## File Structure

The package is organized as follows:

```bash
dynamic-causal-strength/
├── LICENSE
├── README.md
├── pyproject.toml
├── docs/
│   ├── conf.py
│   ├── index.rst
│   └── requirements.txt
├── examples/
│   ├── basic_usage.py
│   └── simulation_demo.py
├── tests/
│   ├── __init__.py
│   ├── test_causality.py
│   ├── test_models.py
│   ├── test_simulation.py
│   └── test_utils.py
└── dcs/
    ├── __init__.py
    ├── causality.py
    ├── models.py
    ├── simulation.py
    └── utils/
        ├── __init__.py
        ├── signal.py
        ├── preprocess.py
        └── residuals.py
```

### Key Modules

- **`causality.py`**: Core functions for causality detection (e.g., `detect_causality`, `time_varying_causality`).
- **`models.py`**: Tools for model estimation and selection (e.g., `estimate_coefficients`, `select_model_order`).
- **`simulation.py`**: Utilities for generating synthetic data (e.g., `generate_signals`, `simulate_var`).
- **`utils/`**: Helper functions for signal processing, preprocessing, and residual calculations.

---

## Usage

### Detecting Causality

Use `detect_causality` to compute the causal strength between time series:

```python
from dcs.causality import detect_causality

# Assuming 'data' is a 3D NumPy array (variables, time, trials)
strength = detect_causality(data, lag=3)
```

### Estimating VAR Coefficients

Estimate coefficients for a Vector Autoregression (VAR) model:

```python
from dcs.models import estimate_coefficients

# Estimate coefficients for model order 2
coeff, residual_cov = estimate_coefficients(data, morder=2)
```

### Simulating Time Series

Generate synthetic time series data for testing:

```python
from dcs.simulation import generate_signals

# Generate signals with specified parameters
X, ns_x, ns_y = generate_signals(T=1000, Ntrial=10, h=0.1, gamma1=0.5, gamma2=0.5, Omega1=1, Omega2=1)
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.

---

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/yourusername/dynamic-causal-strength) or contact the maintainer at [Salar Nouri](salr.nouri@gmail.com).

---

This README provides a comprehensive overview of the `dynamic-causal-strength` package, including all necessary information for users to install, use, and contribute to the project. It adheres to best practices for Python package documentation, ensuring clarity and accessibility. Let me know if you need any additional details or modifications!
