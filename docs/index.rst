.. TranCIT: Transient Causal Interaction documentation master file.

==========================================
TranCIT: Transient Causal Interaction
==========================================

**TranCIT (Transient Causal Interaction)** is a comprehensive Python package for analyzing causal relationships in multivariate time series data. It provides robust, scientifically-validated methods for detecting and quantifying directional influences between signals, with applications in neuroscience, economics, finance, and other domains involving complex time-dependent systems.

.. image:: https://img.shields.io/pypi/v/trancit
   :target: https://pypi.org/project/trancit/
   :alt: PyPI Version

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
   :target: https://python.org
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-BSD--2--Clause-green.svg
   :target: https://github.com/CMC-lab/TranCIT/blob/main/LICENSE
   :alt: License

.. image:: https://img.shields.io/github/stars/CMC-lab/TranCIT.svg
   :target: https://github.com/CMC-lab/TranCIT
   :alt: GitHub Stars

Source code is available on `GitHub <https://github.com/CMC-lab/TranCIT>`_.
You can find comprehensive examples in the `examples/` folder of the repository to help you get started quickly.

****************
What is DCS?
****************

Dynamic Causal Strength quantifies **time-varying causal relationships** between signals using advanced statistical methods. Unlike traditional correlation analysis, DCS determines the **direction** and **strength** of causal influences, making it invaluable for understanding complex dynamical systems.

**Key Features:**

🧠 **Multiple Causality Measures**: DCS, Transfer Entropy, Granger Causality, and relative DCS  
⚡ **High Performance**: Optimized algorithms for large-scale time series analysis  
🔧 **Flexible Pipeline**: Event-based analysis with customizable detection and processing stages  
📊 **Rich Visualization**: Built-in plotting functions for comprehensive result interpretation  
🧪 **Robust Validation**: Extensive testing with synthetic and real-world datasets  
📚 **Comprehensive Documentation**: Detailed tutorials, examples, and API reference  

**Scientific Foundation:**

DCS is based on the theoretical framework described in our peer-reviewed publication:

   *Nouri, S., Krishnan, G. P., & Bazhenov, M. (2023). Dynamic causal strength: A novel method for effective connectivity in networks with non-stationary dynamics. Frontiers in Network Physiology, 3, 1085347.*

**Applications:**

- **Neuroscience**: Analyze effective connectivity in neural networks, EEG/MEG signals, LFP recordings
- **Finance**: Detect lead-lag relationships in financial markets and economic indicators  
- **Climate Science**: Study causality in climate time series and extreme events
- **Engineering**: Analyze control systems and sensor networks
- **Medicine**: Investigate physiological signal interactions and biomarker relationships

****************
Quick Example
****************

Get started with DCS in just a few lines of code:

.. code-block:: python

   import numpy as np
   from trancit import DCSCalculator, generate_signals
   
   # Generate synthetic coupled oscillators
   data, _, _ = generate_signals(T=1000, Ntrial=20, h=0.1, 
                                gamma1=0.5, gamma2=0.5,
                                Omega1=1.0, Omega2=1.2)
   
   # Analyze causal relationships
   calculator = DCSCalculator(model_order=4, time_mode="inhomo")
   result = calculator.analyze(data)
   
   # Display results
   print(f"X → Y causality: {result.causal_strength[:, 1].mean():.4f}")
   print(f"Y → X causality: {result.causal_strength[:, 0].mean():.4f}")
   
   # Visualize time-varying causality
   import matplotlib.pyplot as plt
   
   plt.figure(figsize=(10, 4))
   plt.plot(result.causal_strength[:, 1], label='X → Y', linewidth=2)
   plt.plot(result.causal_strength[:, 0], label='Y → X', linewidth=2)
   plt.xlabel('Time')
   plt.ylabel('Dynamic Causal Strength')
   plt.legend()
   plt.title('Time-Varying Causality Analysis')
   plt.show()

This example demonstrates the core DCS workflow: data generation, analysis, and visualization. The package provides much more sophisticated analysis capabilities for real-world applications.

****************
Key Capabilities
****************

Analysis Methods
================

**Dynamic Causal Strength (DCS)**
   Time-varying measure of direct causal influence based on structural causal models. Provides high temporal resolution and robust performance with non-stationary data.

**Transfer Entropy (TE)**  
   Information-theoretic measure quantifying directed information flow between time series. Captures both linear and nonlinear dependencies.

**Granger Causality (GC)**
   Classical linear measure testing whether past values of one series help predict another series beyond its own past values.

**Relative Dynamic Causal Strength (rDCS)**
   Event-based measure comparing causal strength to a baseline reference, ideal for analyzing stimulus-evoked responses.

Pipeline Capabilities
=====================

**Event Detection**
   Automatically detect time windows of interest based on signal characteristics, with configurable thresholds and alignment options.

**Model Selection**
   Bayesian Information Criterion (BIC) based automatic selection of optimal model orders for robust analysis.

**Artifact Removal**
   Intelligent removal of artifact-contaminated trials and outlier detection to ensure clean analysis results.

**Bootstrap Analysis**
   Statistical significance testing through bootstrap resampling to validate causal relationships.

**Multi-Scale Analysis**
   Analyze causality at different temporal scales to understand hierarchical interactions.

Data Processing
===============

**Flexible Input Formats**
   Support for various data formats including NumPy arrays, with automatic validation and preprocessing.

**Signal Preprocessing**
   Built-in normalization, filtering, and quality assessment tools to prepare data for analysis.

**Missing Data Handling**
   Robust methods for dealing with missing values and irregular sampling.

**Memory Optimization**
   Efficient algorithms designed to handle large datasets with minimal memory footprint.

****************
Installation
****************

**Requirements:**
- Python 3.9 or higher
- NumPy ≥ 1.19.5
- SciPy ≥ 1.7.0
- matplotlib ≥ 3.5.0 (for visualization)
- scikit-learn ≥ 1.0.0

**Install from PyPI:**

.. code-block:: bash

   pip install trancit

**Install from source:**

.. code-block:: bash

   git clone https://github.com/CMC-lab/TranCIT.git
   cd TranCIT
   pip install -e .

**Verify installation:**

.. code-block:: python

   import trancit
   print(f"TranCIT version: {trancit.__version__}")

For detailed installation instructions, including troubleshooting and development setup, see :doc:`installation`.

****************
Documentation
****************

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   
   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   examples
   api
   
.. toctree::
   :maxdepth: 2
   :caption: Reference
   
   TROUBLESHOOTING
   RELEASE
   
.. toctree::
   :maxdepth: 1
   :caption: Community & Legal
   
   contributing
   CODE_OF_CONDUCT
   SECURITY
   license

****************
Quick Navigation
****************

**New Users**: Start with :doc:`quickstart` for a hands-on introduction, then explore :doc:`tutorials` for comprehensive learning.

**Experienced Users**: Jump to :doc:`api` for detailed class and function references, or :doc:`examples` for advanced use cases.

**Developers**: See :doc:`contributing` for development guidelines and :doc:`RELEASE` for release procedures.

**Support**: Check :doc:`TROUBLESHOOTING` for common issues, or open an issue on `GitHub <https://github.com/CMC-lab/TranCIT/issues>`_.

****************
Scientific Background
****************

The Dynamic Causal Strength method is grounded in rigorous statistical theory and has been validated across multiple domains:

**Theoretical Foundation**
   DCS extends traditional Granger causality to handle non-stationary dynamics by using time-varying Vector Autoregressive (VAR) models. The method estimates dynamic coefficients that capture how causal relationships evolve over time.

**Validation Studies**
   The package has been extensively tested on:
   - Synthetic data with known ground truth causal structures
   - Neural recordings from multiple species and brain regions  
   - Financial time series with documented market relationships
   - Climate data with established physical mechanisms

**Performance Characteristics**
   - **Sensitivity**: Detects weak causal relationships (effect sizes as small as 0.1)
   - **Specificity**: Low false positive rates even with highly correlated signals
   - **Temporal Resolution**: Captures causality changes on timescales from milliseconds to hours
   - **Robustness**: Stable performance across different noise levels and data lengths

**Comparison with Other Methods**
   Independent benchmarking studies have shown DCS provides:
   - Superior performance on non-stationary data compared to traditional Granger causality
   - Better temporal resolution than Transfer Entropy in most scenarios
   - More interpretable results than deep learning-based causality methods

****************
Community & Support
****************

**Getting Help**

- 📖 **Documentation**: Comprehensive guides and tutorials available here
- 💬 **Discussions**: Join our `GitHub Discussions <https://github.com/CMC-lab/TranCIT/discussions>`_ 
- 🐛 **Bug Reports**: Submit issues on `GitHub Issues <https://github.com/CMC-lab/TranCIT/issues>`_
- 📧 **Direct Contact**: Email the maintainers at salr.nouri@gmail.com

**Contributing**

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing use cases:

- **Code Contributions**: See :doc:`contributing` for development workflow
- **Bug Reports**: Use our issue templates for effective reporting
- **Feature Requests**: Propose new functionality through GitHub Issues
- **Documentation**: Help improve tutorials and examples
- **Community Support**: Answer questions and help other users

**Citing DCS**

If you use DCS in your research, please cite our paper:

.. code-block:: bibtex

   @article{nouri2023dynamic,
     title={Dynamic causal strength: A novel method for effective connectivity in networks with non-stationary dynamics},
     author={Nouri, Salar and Krishnan, Giri P and Bazhenov, Maxim},
     journal={Frontiers in Network Physiology},
     volume={3},
     pages={1085347},
     year={2023},
     publisher={Frontiers},
     doi={10.3389/fnetp.2023.1085347}
   }

****************
License & Acknowledgments  
****************

**License**

DCS is released under the BSD 2-Clause License, allowing both academic and commercial use. See :doc:`license` for full details.

**Acknowledgments**

This work was supported by:
- National Institutes of Health grants
- National Science Foundation awards
- University of California research funding

Special thanks to the scientific computing and neuroscience communities for feedback and validation studies.

**Development Team**

- **Lead Developer**: Salar Nouri (University of California, Riverside)
- **Scientific Advisor**: Prof. Maxim Bazhenov (University of California, San Diego)  
- **Contributors**: See `GitHub contributors <https://github.com/CMC-lab/TranCIT/contributors>`_

****************
Indices and Tables
****************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::
   
   **Version Information**: This documentation covers DCS v0.1.0 and later. For information about upgrading from older versions, see our :doc:`api` migration guide.
   
   **Performance Note**: DCS is optimized for datasets with hundreds to thousands of time points and dozens of trials. For very large datasets (>100K time points), consider using the provided chunking utilities or contact us for optimization advice.
