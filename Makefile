# Makefile for dynamic-causal-strength

PYTHON := python3

.PHONY: help lint format test docs clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@echo "  lint        Run flake8 to lint code"
	@echo "  format      Run black to format code"
	@echo "  test        Run tests with pytest"
	@echo "  docs        Build Sphinx HTML documentation"
	@echo "  clean       Remove build, dist, and cache files"

lint:
	flake8 dcs tests

format:
	black dcs tests

test:
	pytest tests

docs:
	cd docs && make html

clean:
	rm -rf build dist .pytest_cache __pycache__ .mypy_cache .coverage
	find . -name "*.pyc" -delete
