# Makefile for SuperQuantX development

.PHONY: help install install-dev test lint format type-check docs clean build release

help: ## Show this help message
	@echo "SuperQuantX Development Commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation
install: ## Install package for production use
	pip install .

install-dev: ## Install package in development mode with all dependencies
	pip install -e ".[full-dev]"

install-uv: ## Install with uv (recommended for development)
	uv sync --extra full-dev

# Testing
test: ## Run all tests
	pytest

test-fast: ## Run fast tests only (skip slow/integration tests)
	pytest -m "not slow"

test-cov: ## Run tests with coverage
	pytest --cov=superquantx --cov-report=html --cov-report=term

test-backends: ## Run backend integration tests
	pytest -m "backend"

# Code Quality
lint: ## Run linting with ruff
	ruff check src/ tests/

lint-fix: ## Fix linting issues automatically
	ruff check --fix src/ tests/

format: ## Format code with black
	black src/ tests/

format-check: ## Check if code is properly formatted
	black --check src/ tests/

type-check: ## Run type checking with mypy
	mypy src/

# Pre-commit
pre-commit-install: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

# Documentation
docs-serve: ## Serve documentation locally
	mkdocs serve

docs-build: ## Build documentation
	mkdocs build

docs-deploy: ## Deploy documentation (for maintainers)
	mkdocs gh-deploy

# Development utilities
clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Building and releasing
build: ## Build wheel and source distribution
	python -m build

check-build: ## Check package build
	python -m twine check dist/*

release-test: ## Upload to test PyPI
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: ## Upload to PyPI (for maintainers only)
	python -m twine upload dist/*

# Benchmarking
benchmark: ## Run performance benchmarks
	pytest tests/benchmarks/ -v

benchmark-compare: ## Compare benchmark results
	pytest tests/benchmarks/ --benchmark-compare

# Development environment
dev-setup: install-uv pre-commit-install ## Complete development setup
	@echo "Development environment setup complete!"
	@echo "Run 'source .venv/bin/activate' to activate the environment"

dev-check: lint format-check type-check test-fast ## Run all development checks
	@echo "All development checks passed! ✅"

# Research utilities
examples: ## Run all example notebooks/scripts
	@echo "Running research examples..."
	python examples/run_examples.py

validate-algorithms: ## Validate all quantum algorithms work
	python -m superquantx.validation.algorithm_validation

backend-status: ## Check status of all quantum backends
	python -c "import superquantx as sqx; print(sqx.list_available_backends())"

# Docker (future)
docker-build: ## Build Docker image for development
	docker build -t superquantx-dev .

docker-run: ## Run development environment in Docker
	docker run -it -v $(PWD):/workspace superquantx-dev

# Help for new contributors
contributor-setup: ## Complete setup for new contributors
	@echo "Setting up SuperQuantX for contribution..."
	@echo "1. Installing development dependencies..."
	make install-uv
	@echo "2. Setting up pre-commit hooks..."
	make pre-commit-install
	@echo "3. Running initial tests..."
	make test-fast
	@echo ""
	@echo "🎉 Setup complete! You're ready to contribute to SuperQuantX!"
	@echo ""
	@echo "Next steps:"
	@echo "  - Read CONTRIBUTING.md for contribution guidelines"
	@echo "  - Check out open issues: https://github.com/superagentic/superquantx/issues"
	@echo "  - Join discussions: https://github.com/superagentic/superquantx/discussions"