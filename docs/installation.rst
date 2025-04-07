.. _installation:

############
Installation
############

Prerequisites
=============

Make sure you have the following installed:

* Python 3.8 or higher (Your `setup.py` specifies >=3.12 [cite: uploaded:dynamic-causal-strength/setup.py])
* NumPy 1.21 or higher [cite: uploaded:dynamic-causal-strength/README.md] (Your `docs/requirements.txt` specifies 1.26.4 [cite: uploaded:dynamic-causal-strength/docs/requirements.txt])
* SciPy 1.7 or higher [cite: uploaded:dynamic-causal-strength/README.md] (Your `docs/requirements.txt` specifies 1.14.1 [cite: uploaded:dynamic-causal-strength/docs/requirements.txt])

.. note::
   It's recommended to use a virtual environment (`venv` or `conda`) to manage dependencies.

Install from PyPI
=================

The easiest way to install the latest stable release is using pip:

.. code-block:: bash

   pip install dynamic-causal-strength

*(Note: Replace 'dynamic-causal-strength' with your actual PyPI package name if different)*

Install from Source
===================

To install the latest development version directly from the source code:

1. Clone the repository:

   .. code-block:: bash

      # Replace with your actual repository URL
      git clone https://github.com/sa-nouri/dcs.git
      cd dcs

2. Install the package (in editable mode `-e` if you plan to develop):

   .. code-block:: bash

      pip install .
      # Or for editable install:
      # pip install -e .
