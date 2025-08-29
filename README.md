
# TranCIT: Transient Causal Interaction Toolbox

[![PyPI version](https://img.shields.io/pypi/v/trancit.svg)](https://pypi.org/project/trancit/)
[![License](https://img.shields.io/github/license/CMC-lab/TranCIT)](https://github.com/CMC-lab/TranCIT/blob/main/LICENSE)
[![CI](https://github.com/CMC-lab/TranCIT/actions/workflows/ci.yml/badge.svg)](https://github.com/CMC-lab/TranCIT/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/trancit/badge/?version=latest)](https://trancit.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16998397.svg)](https://doi.org/10.5281/zenodo.16998397)

TranCIT (Transient Causal Interaction Toolbox) is a Python package for quantifying causal relationships in multivariate time series data. It provides methods for analyzing directional influences using model-based statistical tools, inspired by information-theoretic and autoregressive frameworks.

## 🚀 Features

- **Dynamic Causal Strength (DCS)**: Time-varying causal relationships
- **Transfer Entropy (TE)**: Information-theoretic causality measures
- **Granger Causality (GC)**: Linear causality detection
- **Relative Dynamic Causal Strength (rDCS)**: Event-based causality
- **VAR-based Modeling**: Vector autoregressive time series analysis
- **BIC Model Selection**: Automatic model order selection
- **Bootstrap Support**: Statistical significance testing
- **DeSnap Analysis**: Debiased statistical analysis
- **Pipeline Architecture**: Modular, stage-based analysis pipeline

## 📦 Installation

### From PyPI (Recommended)

```bash
pip install trancit
```

### From Source

```bash
git clone https://github.com/CMC-lab/TranCIT.git
cd TranCIT
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/CMC-lab/TranCIT.git
cd TranCIT
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Causality Analysis

```python
import numpy as np
from trancit import DCSCalculator, generate_signals

# Generate synthetic data
data, _, _ = generate_signals(T=1000, Ntrial=20, h=0.1, 
                             gamma1=0.5, gamma2=0.5, 
                             Omega1=1.0, Omega2=1.2)

# Create DCS calculator
calculator = DCSCalculator(model_order=4, time_mode="inhomo")

# Perform analysis
result = calculator.analyze(data)
print(f"DCS shape: {result.causal_strength.shape}")
print(f"Transfer Entropy shape: {result.transfer_entropy.shape}")
```

### Event-Based Analysis Pipeline

```python
import numpy as np
from trancit import PipelineOrchestrator, generate_signals
from trancit.config import PipelineConfig, PipelineOptions, DetectionParams, CausalParams

# Generate data
data, _, _ = generate_signals(T=1200, Ntrial=20, h=0.1, 
                             gamma1=0.5, gamma2=0.5, 
                             Omega1=1.0, Omega2=1.2)
original_signal = np.mean(data, axis=2)
detection_signal = original_signal * 1.5

# Configure pipeline
config = PipelineConfig(
    options=PipelineOptions(detection=True, causal_analysis=True),
    detection=DetectionParams(thres_ratio=2.0, align_type="peak", 
                            l_extract=150, l_start=75),
    causal=CausalParams(ref_time=75, estim_mode="OLS"),
)

# Run analysis
orchestrator = PipelineOrchestrator(config)
result = orchestrator.run(original_signal, detection_signal)

# Access results
if result.results.get("CausalOutput"):
    dcs_values = result.results["CausalOutput"]["OLS"]["DCS"]
    te_values = result.results["CausalOutput"]["OLS"]["TE"]
    print(f"DCS shape: {dcs_values.shape}")
```

### Model Selection and Validation

```python
import numpy as np
from trancit import VAREstimator, BICSelector, ModelValidator

# Generate sample data
data = np.random.randn(2, 1000, 20)  # (n_vars, n_obs, n_trials)

# BIC model selection
bic_selector = BICSelector(max_order=6, mode="biased")
bic_results = bic_selector.compute_multi_trial_BIC(data, {"Params": {"BIC": {"momax": 6, "mode": "biased"}}, "EstimMode": "OLS"})

# VAR estimation
estimator = VAREstimator(model_order=4, time_mode="inhomo")
coefficients, residuals, log_likelihood, hessian_sum = estimator.estimate_var_coefficients(
    data, model_order=4, max_model_order=6, time_mode="inhomo", lag_mode="infocrit"
)

# Model validation
validator = ModelValidator()
validation_result = validator.validate(coefficients, residuals, data)
print(f"Model stable: {validation_result.model_stability}")
```

## 📚 Documentation & Examples

For comprehensive documentation, tutorials, and API reference:

👉 **[ReadTheDocs Documentation](https://trancit.readthedocs.io)**

### Examples

- **[Basic Usage](examples/basic_usage.py)**: Simple causality analysis
- **[LFP Pipeline](examples/lfp_pipeline.py)**: Local field potential analysis
- **[DCS Introduction](examples/dcs_introduction.ipynb)**: Interactive tutorial

## 🔬 Scientific Background

This package implements methods from:

- **Shao et al. (2023)**: Information theoretic measures of causal influences during transient neural events
- **Granger Causality**: Linear causality detection in time series
- **Transfer Entropy**: Information-theoretic causality measures

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dcs --cov-report=html

# Run linting
flake8 dcs/ tests/

# Format code
black dcs/ tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/CMC-lab/TranCIT.git
cd TranCIT
pip install -e ".[dev]"
pre-commit install
```

## 📖 Citing This Work

If you use **TranCIT** in your research, please cite:

```bibtex
@article{shao2023information,
  title={Information theoretic measures of causal influences during transient neural events},
  author={Shao, Kaidi and Logothetis, Nikos K and Besserve, Michel},
  journal={Frontiers in Network Physiology},
  volume={3},
  pages={1085347},
  year={2023},
  publisher={Frontiers Media SA}
}
```

And cite this software package:

```bibtex
@software{nouri_2025_16998397,
  author       = {Nouri, Salar and
                  Shao, Kaidi and
                  Safavi, Shervin},
  title        = {TranCIT: Transient Causal Interaction Toolbox},
  month        = aug,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.16998397},
  url          = {https://doi.org/10.5281/zenodo.16998397}
}
```

## 📄 License

This project is licensed under the BSD 2-Clause License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research from the CMC-Lab
- Inspired by information-theoretic causality methods
- Built with support from the scientific Python community

## 📞 Contact

- **Maintainer**: Salar Nouri (<salr.nouri@gmail.com>)
- **Issues**: [GitHub Issues](https://github.com/CMC-lab/TranCIT/issues)
- **Documentation**: [ReadTheDocs](https://trancit.readthedocs.io)
