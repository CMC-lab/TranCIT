
# Dynamic Causal Strength (DCS)

[![PyPI version](https://img.shields.io/pypi/v/dynamic-causal-strength.svg)](https://pypi.org/project/dynamic-causal-strength/)
[![License](https://img.shields.io/github/license/CMC-lab/dcs)](https://github.com/CMC-lab/dcs/blob/main/LICENSE)
[![CI](https://github.com/CMC-lab/dcs/actions/workflows/ci.yml/badge.svg)](https://github.com/CMC-lab/dcs/actions/workflows/ci.yml)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Dynamic Causal Strength (DCS) is a Python package for quantifying causal relationships in multivariate time series data. It provides methods for analyzing directional influences using model-based statistical tools, inspired by information-theoretic and autoregressive frameworks.

## Features

- Estimate dynamic causal strength (DCS)
- VAR-based modeling of time series data
- Event-based snapshot extraction
- BIC-based model order selection
- Statistical analysis of event structures
- Modular and extensible pipeline

## Installation

```bash
  pip install dynamic-causal-strength
```

## Quickstart

```python
from dcs import compute_causal_strength

# Example usage
results = compute_causal_strength(data, config)
```

## Documentation

For full documentation, examples, and tutorials, see:  
👉 [ReadTheDocs Documentation](https://dynamic-causal-strength.readthedocs.io)

## Contributing

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Citing This Work

If you use **Dynamic Causal Strength (DCS)** in your research, please consider citing the following papers:

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

## License

This project is licensed under the BSD 2-Clause License. See the [LICENSE](LICENSE) file for details.
