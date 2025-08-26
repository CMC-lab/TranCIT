# Development Tools

This directory contains temporary scripts and tools used during development and maintenance of the TranCIT package.

## Scripts

- `add_mypy_ignores.py` - Automated script to add mypy ignore comments for specific error patterns
- `fix_linting.py` - Automated script to apply code formatting (autoflake, isort, black) and fix linting issues  
- `fix_mypy_issues.py` - Automated script to add type ignore comments for mypy errors
- `run_mypy_with_threshold.py` - Script to run mypy with error threshold checking for CI

## Usage

These scripts were used during the repository refactoring and test alignment process. They are kept for reference but should not be needed for normal development.

For regular development, use the standard tools:
- `make lint-fix` - Run linting and formatting
- `make type-check` - Run type checking
- `pytest` - Run tests
