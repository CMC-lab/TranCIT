# dcs: Dynamic Causal Strength Package

The `dcs` package provides implementations of causal analysis methods and simulations necessary to replicate the findings of the paper "[Information Theoretic Measures of Causal Influences During Transient Neural Events](https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2023.1085347/full)" by Shao et al. (2023).

## Overview

This package focuses on quantifying causal interactions in neural time series data, particularly during transient events. It includes implementations of:

- **Transfer Entropy (TE)**
- **Dynamic Causal Strength (DCS)**
- **Relative Dynamic Causal Strength (rDCS)**

These methods are designed to assess causal influences in both simulated models and experimental neural recordings.

## Installation

To install the `dcs` package, clone the repository and install it using pip:

```bash
    git clone https://github.com/yourusername/dcs.git
    cd dcs
    pip install .
```

## Package Structure

The package is organized as follows:

```bash
    dcs/
    ├── causality
    │   ├── __init__.py
    │   ├── detecting_analysis_pipeline.py
    │   └── time_varying.py
    ├── src
    │   ├── __init__.py
    │   └── bic.py
    ├── utils
    │   ├── core
    │   │   ├── __init__.py
    │   │   ├── estimating_residuals.py
    │   │   ├── finding_best_shrinked_locs.py
    │   │   ├── finding_peak_loc.py
    │   │   ├── getting_Yt.py
    │   │   └── getting_residuals.py
    │   ├── preprocessing
    │   │   ├── __init__.py
    │   │   ├── extracting_events.py
    │   │   └── removing_artif_trials.py
    │   └── simulate
    │       ├── __init__.py
    │       ├── simulate_ar_event.py
    │       └── simulate_timefreq.py
    ├── README.md
    ├── setup.py
    ├── pyproject.toml
    ├── setup.cfg
    ├── requirements.txt
    └── LICENSE

```

## Usage

After installation, you can utilize the package to perform causal analysis as follows:

```bash
from dcs.causality import detecting_analysis_pipeline
from dcs.utils.preprocessing import extracting_events
from dcs.utils.simulate import simulate_ar_event

```

For detailed examples and tutorials, please refer to the examples directory in the repository.

## Citation

If you use this package in your research, please cite the original paper:

xxxx

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

We acknowledge the authors of the original paper for their foundational work in developing these causal analysis methods.
