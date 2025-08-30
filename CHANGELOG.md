# Changelog

All notable changes to the TranCIT: Transient Causal Interaction Toolbox project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-08-29

### Added

- **Initial Release** of TranCIT: Transient Causal Interaction Toolbox
- **Core Functionality**:
  - Dynamic Causal Strength (DCS) analysis for time-varying causal relationships
  - Transfer Entropy (TE) computation for information-theoretic causality measures
  - Granger Causality (GC) for linear causality detection
  - Relative Dynamic Causal Strength (rDCS) for event-based causality analysis

- **Pipeline Architecture**:
  - Modular pipeline orchestrator for complete analysis workflows
  - Event detection and snapshot extraction stages
  - Artifact removal and statistical analysis stages
  - Bootstrap analysis and DeSnap (debiased statistical) analysis

- **Model Estimation**:
  - Vector Autoregressive (VAR) model estimation
  - Bayesian Information Criterion (BIC) model selection
  - Model validation with stability checks and diagnostics
  - Multiple estimation modes (OLS, RLS)

- **Data Simulation**:
  - Coupled oscillator signal generation
  - AR process simulation with event structures
  - VAR simulation utilities
  - Morlet wavelet implementation

- **Utilities & Preprocessing**:
  - Signal preprocessing and normalization
  - Matrix regularization for numerical stability
  - Residual analysis and covariance computation
  - Statistical helper functions

- **Documentation**:
  - Comprehensive API documentation with Sphinx
  - Installation and usage guides
  - Example notebooks and scripts
  - Scientific background and references

- **Testing & Quality Assurance**:
  - Complete test suite with pytest
  - Type annotations throughout codebase
  - Code formatting with Black
  - Linting with Flake8
  - Continuous Integration with GitHub Actions

- **Package Management**:
  - Modern `pyproject.toml` configuration
  - Python 3.9+ compatibility
  - Proper dependency management
  - BSD-2-Clause license

### Technical Specifications

- **Python Compatibility**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Core Dependencies**: NumPy (≥1.19.5), SciPy (≥1.7.0), Matplotlib (≥3.5.0), Scikit-learn (≥1.0.0)
- **Optional Dependencies**: Sphinx, pytest, black, flake8, mypy
- **License**: BSD-2-Clause
- **Documentation**: Available at `https://trancit.readthedocs.io`

### Scientific Background

This implementation is based on research from:

- Shao, K., Logothetis, N. K., & Besserve, M. (2023). Information theoretic measures of causal influences during transient neural events. *Frontiers in Network Physiology*, 3, 1085347.

### Examples Included

- Basic TranCIT analysis workflow
- LFP (Local Field Potential) analysis pipeline
- Interactive Jupyter notebook demonstrations
- Figure replication from scientific literature

## [Unreleased]

### Planned Features

- Additional causality measures
- Performance optimizations
- Extended documentation
- More simulation utilities
- Advanced visualization tools

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](/CONTRIBUTING.md) for details.

## Support

- **Documentation**: `https://trancit.readthedocs.io`
- **Issues**: `https://github.com/CMC-lab/TranCIT/issues`
- **Discussions**: `https://github.com/CMC-lab/TranCIT/discussions`
