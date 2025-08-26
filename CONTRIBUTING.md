# Contributing to TranCIT: Transient Causal Interaction

Thank you for considering contributing to the TranCIT project!
Your help is highly appreciated and essential for improving this package.

Whether you're reporting bugs, suggesting features, improving documentation, or submitting code — you're welcome!

---

## 📦 Project Structure

The main code is organized under:

```bash
    trancit/
```

Docs are under:

```bash
    docs/
```

Tests are typically under:

```bash
    tests/
```

---

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/CMC-lab/TranCIT
   cd TranCIT
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv trancit_env
   source trancit_env/bin/activate
   ```

3. **Install the package (editable mode) with dev dependencies**:

   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests**:

   ```bash
   pytest
   ```

---

## Making a Contribution

### Bug Reports or Feature Requests

- Use the [GitHub Issues](https://github.com/CMC-lab/TranCIT/issues) section.
- Please provide a clear description and, if possible, steps to reproduce the issue.

### Code Contributions

- Fork the repo
- Create a feature or bugfix branch: `git checkout -b fix/some-issue`
- Write tests if necessary
- Run tests and linters locally before pushing
- Open a pull request (PR) against `main` with a meaningful description

---

## Style Guide

- **Follow flake8** for Python code.
- **Docstrings** use Google-style or NumPy-style (keep it consistent).
- Run `black` or `ruff` to auto-format.
- Type annotations are strongly encouraged.

---

## Testing

We use `pytest` for testing. Make sure your contributions include tests and that existing tests pass:

```bash
pytest
```

---

## 📚 Documentation

The documentation lives in the `docs/` folder and uses **Sphinx**.

To build locally:

```bash
cd docs
make html
```

---

## License

This project uses the BSD-2-Clause License. See [LICENSE](./LICENSE) for more details.

---

Thanks again for helping improve TranCIT! 🙏
