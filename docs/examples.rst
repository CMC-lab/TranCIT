
.. _examples:

########
Examples
########

This section demonstrates practical use cases of the ``dcs`` package, ranging from basic synthetic simulations to real-world analysis pipelines.

All examples are available in the ``examples/`` directory in the source code.
You can also explore them directly on GitHub:

- `View examples folder on GitHub <https://github.com/CMC-lab/dcs/tree/main/examples>`_

Basic Usage Script
==================

The `basic_usage.py` script demonstrates an end-to-end workflow using synthetic data. It includes:

- Signal simulation
- Minimal configuration of the pipeline
- Running detection and analysis
- Inspecting outputs (e.g., DCS values)

.. literalinclude:: ../examples/basic_usage.py
   :language: python
   :linenos:

Reproducing a Scientific Figure (Jupyter Notebook)
==================================================

The notebook `dcs_introduction.ipynb` shows how to replicate **Figure 4** from the associated scientific paper [https://www.frontiersin.org/journals/network-physiology/articles/10.3389/fnetp.2023.1085347/full]. It highlights:

- A more complex setup with real signal dynamics
- Visual interpretation of detection windows and causal strength
- Parameter tuning for reproducibility

You can open the notebook locally or online:

- `View notebook on GitHub <https://github.com/CMC-lab/dcs/blob/main/examples/dcs_introduction.ipynb>`_

Advanced Pipeline Example
=========================

For a more advanced and modular demonstration, check out `lfp_pipeline.py`. This script includes:

- Example of real neural signal preprocessing
- Flexible parameter configuration using dataclasses
- Logging and structure suitable for batch jobs or publication workflows

- `View lfp_pipeline.py on GitHub <https://github.com/CMC-lab/dcs/blob/main/examples/lfp_pipeline.py>`_

More Coming Soon!
=================

More examples will be added over time. Contributions are welcome — see the `contributing` guide for details!
