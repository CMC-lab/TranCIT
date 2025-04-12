
.. _contributing:

############
Contributing
############

Thank you for your interest in contributing to **Dynamic Causal Strength (DCS)**! 🎉  
We welcome all forms of contributions — from fixing bugs and improving docs to proposing features or submitting new code.

If you're not sure where to start, feel free to open an issue or ask a question. All contributions, big or small, are appreciated.

TL;DR - How to Contribute
=========================

1. Fork the repo and clone it locally.
2. Create a new branch for your change.
3. Make your changes (with `black`, `flake8`, and tests).
4. Push to your fork and open a pull request.

Reporting Issues
================

Found a bug or have a suggestion? Great! Before opening a new issue, please:

- Search existing issues on `GitHub <https://github.com/CMC-lab/dcs/issues>`_ to avoid duplicates.
- If it's new, open an issue and include:
  - Expected vs. actual behavior
  - Steps to reproduce (code snippets welcome!)
  - Environment details (OS, Python version, `dcs`, `numpy`, etc.)

Feature Requests
================

We're always open to thoughtful feature proposals. Please describe:

- What problem this feature solves
- A clear use case or example
- Any implementation ideas you have

Open a discussion or issue to get started!

Contributing Code
=================

To contribute code (bug fixes, features, refactoring, etc.):

1. **Fork the repo** on GitHub.
2. **Create a new branch**:
   .. code-block:: bash

      git checkout -b my-feature

3. **Make your changes**, following our coding standards:
   - Use `black` for formatting
   - Use `flake8` to catch linting issues
   - Add **docstrings** where applicable
   - Write **tests** using `pytest`
   - Run:
     .. code-block:: bash

        pytest

4. **Update docs** if your changes affect usage or API.
5. **Commit with clear messages**, then:
   .. code-block:: bash

      git push origin my-feature

6. **Open a Pull Request** to the `main` branch with a detailed summary of your changes.

Need help with any of this? Just ask!

More Guidelines
===============

We follow best practices from the Python community. For a more detailed guide, please see the  
`CONTRIBUTING.md <https://github.com/CMC-lab/dcs/main/CONTRIBUTING.md>`_.

Thank you again for helping improve the DCS package!
