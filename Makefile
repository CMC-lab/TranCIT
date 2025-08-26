# Makefile for TranCIT: Transient Causal Interaction

PYTHON := python3

.PHONY: help lint format test docs clean lint-check lint-fix format-and-lint

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  lint        Run all linting checks"
	@echo "  lint-check  Check code style without fixing"
	@echo "  lint-fix    Auto-fix linting issues"
	@echo "  format      Format code with black, isort, autoflake"
	@echo "  format-and-lint  Format code then run lint checks"
	@echo "  test        Run tests with pytest"
	@echo "  docs        Build Sphinx HTML documentation"
	@echo "  clean       Remove build, dist, and cache files"

lint-check:
	@echo "🔍 Checking code style..."
	black --check --diff trancit/ tests/ examples/
	isort --check-only --diff --skip trancit/_version.py trancit/ tests/ examples/
	flake8 trancit/ tests/ examples/ --max-line-length=100 --extend-ignore=E203,W503,E712 --exclude=trancit/_version.py
	@echo "✅ Lint check complete!"

type-check:
	@echo "🔍 Checking types with acceptable threshold..."
	python run_mypy_with_threshold.py
	@echo "✅ Type check complete!"

type-check-simple:
	@echo "🔍 Running simple type check (research code friendly)..."
	mypy trancit/ --config-file=mypy_simple.ini
	@echo "✅ Simple type check complete!"

lint-fix:
	@echo "🔧 Auto-fixing linting issues..."
	python fix_linting.py
	@echo "✅ Auto-fixes complete!"

lint: lint-check

format:
	@echo "🎨 Formatting code..."
	autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r trancit/ tests/ examples/
	isort --skip trancit/_version.py trancit/ tests/ examples/
	black trancit/ tests/ examples/
	@echo "✨ Formatting complete!"

format-and-lint: format lint-check

test:
	pytest tests

docs:
	cd docs && make html

clean:
	rm -rf build dist .pytest_cache __pycache__ .mypy_cache .coverage
	find . -name "*.pyc" -delete
