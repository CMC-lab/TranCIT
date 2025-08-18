
.. _installation:

############
Installation
############

Prerequisites
=============

Before installing **Dynamic Causal Strength (DCS)**, ensure your environment includes:

* Python 3.9 or higher
* NumPy 1.26 or higher
* SciPy 1.14 or higher

.. note::
   It's highly recommended to use a virtual environment (`venv` or `conda`) to manage dependencies and avoid conflicts.

Install from PyPI
=================

To install the latest stable release from PyPI:

.. code-block:: bash

   pip install dynamic-causal-strength

.. tip::
   Replace `'dynamic-causal-strength'` with the actual PyPI package name if different.

Install from Source
===================

To install the latest development version from the source code:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/sa-nouri/dcs.git
      cd dcs

2. Install the package:

   .. code-block:: bash

      pip install .
      # Or for editable install (recommended if you are contributing or making changes):
      pip install -e .

Verify Installation
===================

To check if DCS was installed correctly:

.. code-block:: bash

   python -c "import dcs; print(dcs.__version__)"

Optional Dependencies
=====================

Some features may require additional packages:

* `matplotlib` for plotting
* `jupyter` for running notebooks in the examples directory

To install these along with DCS:

.. code-block:: bash

   pip install dynamic-causal-strength[dev]
