# Makefile dla projektu video2svg
# Autor: Video2SVG Team
# Data: 2023

.PHONY: clean build test lint publish-test publish install dev-install

# Zmienne
PYTHON := python3
PIP := pip
PYTEST := pytest
FLAKE8 := flake8
RUFF := ruff
MYPY := mypy
TWINE := twine
PACKAGE_DIR := video2svg
DIST_DIR := dist
BUILD_DIR := build

# Komendy główne
all: clean lint test build

# Czyszczenie plików tymczasowych i dystrybucyjnych
clean:
	@echo "Czyszczenie plików tymczasowych..."
	rm -rf $(DIST_DIR) $(BUILD_DIR) *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Instalacja zależności deweloperskich
dev-install:
	$(PIP) install -e ".[dev]"

# Standardowa instalacja pakietu
install:
	$(PIP) install -e .

# Budowanie pakietu
build: clean
	@echo "Budowanie pakietu dystrybucyjnego..."
	$(PYTHON) -m build

# Uruchamianie testów
test:
	@echo "Uruchamianie testów..."
	$(PYTEST) tests/

# Testowanie z pokryciem kodu
test-coverage:
	@echo "Uruchamianie testów z pokryciem kodu..."
	$(PYTEST) --cov=$(PACKAGE_DIR) tests/

# Sprawdzanie jakości kodu
lint: lint-flake8 lint-mypy lint-ruff

lint-flake8:
	@echo "Uruchamianie flake8..."
	$(FLAKE8) $(PACKAGE_DIR)

lint-mypy:
	@echo "Uruchamianie mypy..."
	$(MYPY) $(PACKAGE_DIR)

lint-ruff:
	@echo "Uruchamianie ruff..."
	$(RUFF) check $(PACKAGE_DIR)

# Publikowanie na TestPyPI
publish-test: build
	@echo "Publikowanie pakietu na TestPyPI..."
	$(TWINE) upload --repository-url https://test.pypi.org/legacy/ $(DIST_DIR)/*

# Publikowanie na PyPI
publish: build
	@echo "Publikowanie pakietu na PyPI..."
	$(TWINE) upload $(DIST_DIR)/*

# Uruchamianie demo
demo:
	@echo "Uruchamianie przykładowego konwersji wideo na SVG..."
	$(PYTHON) -m video2svg.cli --help

# Pomoc
help:
	@echo "Dostępne komendy:"
	@echo "  make clean         - Usuwa pliki tymczasowe"
	@echo "  make build         - Buduje pakiet dystrybucyjny"
	@echo "  make test          - Uruchamia testy"
	@echo "  make test-coverage - Uruchamia testy z pokryciem kodu"
	@echo "  make lint          - Sprawdza jakość kodu"
	@echo "  make install       - Instaluje pakiet"
	@echo "  make dev-install   - Instaluje pakiet w trybie deweloperskim"
	@echo "  make publish-test  - Publikuje pakiet na TestPyPI"
	@echo "  make publish       - Publikuje pakiet na PyPI"
	@echo "  make demo          - Uruchamia demonstrację pakietu"
	@echo "  make all           - Czyści, sprawdza kod, testuje i buduje pakiet"

# Domyślna komenda
default: help
