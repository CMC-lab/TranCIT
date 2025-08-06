
# Dynamic Causal Strength (DCS)

[![PyPI version](https://img.shields.io/pypi/v/dynamic-causal-strength.svg)](https://pypi.org/project/dynamic-causal-strength/)
[![License](https://img.shields.io/github/license/CMC-lab/dcs)](https://github.com/CMC-lab/dcs/blob/main/LICENSE)
[![CI](https://github.com/CMC-lab/dcs/actions/workflows/ci.yml/badge.svg)](https://github.com/CMC-lab/dcs/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/dynamic-causal-strength/badge/?version=latest)](https://dynamic-causal-strength.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

Dynamic Causal Strength (DCS) is a Python package for quantifying causal relationships in multivariate time series data. It provides methods for analyzing directional influences using model-based statistical tools, inspired by information-theoretic and autoregressive frameworks.

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
pip install dynamic-causal-strength
```

### From Source

```bash
git clone https://github.com/CMC-lab/dcs.git
cd dcs
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/CMC-lab/dcs.git
cd dcs
pip install -e ".[dev]"
```

## 🎯 Quick Start

### Basic Causality Analysis

```python
import numpy as np
from dcs import DCSCalculator, generate_signals

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
from dcs import PipelineOrchestrator, generate_signals
from dcs.config import PipelineConfig, PipelineOptions, DetectionParams, CausalParams

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
from dcs import VAREstimator, BICSelector, ModelValidator

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

👉 **[ReadTheDocs Documentation](https://dynamic-causal-strength.readthedocs.io)**

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
git clone https://github.com/CMC-lab/dcs.git
cd dcs
pip install -e ".[dev]"
pre-commit install
```

## 📖 Citing This Work

If you use **Dynamic Causal Strength (DCS)** in your research, please cite:

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
@software{nouri2025dynamic,
  title={Dynamic Causal Strength (DCS): A Python package for quantifying causal relationships in multivariate time series data},
  author={Nouri, Salar and Shao, Kaidi and Logothetis, Nikos K. and Besserve, Michel},
  year={2025},
  url={https://github.com/CMC-lab/dcs},
  doi={10.5281/zenodo.1234567}
}
```

## 📄 License

This project is licensed under the BSD 2-Clause License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on research from the CMC-Lab
- Inspired by information-theoretic causality methods
- Built with support from the scientific Python community

## 📞 Contact

- **Maintainer**: Salar Nouri (salr.nouri@gmail.com)
- **Issues**: [GitHub Issues](https://github.com/CMC-lab/dcs/issues)
- **Documentation**: [ReadTheDocs](https://dynamic-causal-strength.readthedocs.io)
