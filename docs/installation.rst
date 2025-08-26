
.. _installation:

############
Installation
############

Prerequisites
=============

Before installing **TranCIT: Transient Causal Interaction**, ensure your environment includes:

* Python 3.9 or higher
* NumPy 1.26 or higher
* SciPy 1.14 or higher

.. note::
   It's highly recommended to use a virtual environment (`venv` or `conda`) to manage dependencies and avoid conflicts.

Install from PyPI
=================

To install the latest stable release from PyPI:

.. code-block:: bash

   pip install trancit

.. tip::
   Replace `'trancit'` with the actual PyPI package name if different.

Install from Source
===================

To install the latest development version from the source code:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/CMC-lab/TranCIT.git
      cd TranCIT

2. Install the package:

   .. code-block:: bash

      pip install .
      # Or for editable install (recommended if you are contributing or making changes):
      pip install -e .

Verify Installation
===================

To check if TranCIT was installed correctly:

.. code-block:: bash

   python -c "import trancit; print(trancit.__version__)"

Optional Dependencies
=====================

Some features may require additional packages:

* `matplotlib` for plotting
* `jupyter` for running notebooks in the examples directory

To install these along with TranCIT:

.. code-block:: bash

   pip install trancit[dev]
