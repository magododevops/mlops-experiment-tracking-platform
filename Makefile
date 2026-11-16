.DEFAULT_GOAL := help

.PHONY: help install lint format format-check test check verify-data

help:
	@echo "Available commands:"
	@echo "  make install       Install locked dependencies"
	@echo "  make lint          Run Ruff lint checks"
	@echo "  make format        Format Python code"
	@echo "  make format-check  Check formatting without changes"
	@echo "  make test          Run unit tests"
	@echo "  make check         Run all quality checks"

install:
	uv sync --locked

lint:
	uv run --locked ruff check .

format:
	uv run --locked ruff format .

format-check:
	uv run --locked ruff format --check .

test:
	uv run --locked pytest

check: lint format-check test

verify-data:
	uv run --locked python scripts/verify_dataset.py
