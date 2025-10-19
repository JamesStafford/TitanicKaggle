# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle Titanic competition analysis project using Python and pandas. The project analyzes the Titanic dataset to explore survival patterns, with a focus on age group survival analysis.

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src

# Run a single test file
pytest tests/test_age_group_utils.py

# Run a specific test class
pytest tests/test_age_group_utils.py::TestCreateAgeToSurvival

# Run a specific test method
pytest tests/test_age_group_utils.py::TestCreateAgeToSurvival::test_create_age_to_survival_df_does_not_throw
```

### Linting
```bash
# Run ruff linting
ruff check .

# Run ruff with auto-fix
ruff check --fix .

# Format code with ruff
ruff format .
```

### Jupyter Notebooks
```bash
# Start Jupyter notebook server
jupyter notebook

# The main analysis notebook is at: notebooks/kaggle_titanic_notebook.ipynb
```

## Code Architecture

### Data Location
- Training data: `data/train.csv`
- Test data: `data/test.csv`
- Submission template: `data/gender_submission.csv`

### Source Structure

The codebase is organized into modular utilities:

- **`src/constants.py`**: Central location for all column header constants used throughout the project (e.g., `SURVIVED_HEADER`, `AGE_HEADER`, `AGE_GROUP_HEADER`). Always use these constants instead of hardcoding column names.

- **`src/age_group_survival/`**: Feature-specific module for age group analysis
  - `age_group_utils.py`: Contains utilities for creating age bins (10-year intervals from 0-100) and processing age survival data
    - `create_age_to_survival_df()`: Main function that takes the full training DataFrame and returns a processed DataFrame with age groups
    - `create_age_bins_and_labels()`: Creates age bin ranges and labels (e.g., "20-29", "30-39")
    - Uses `AgeBinsAndLabels` NamedTuple for structured return values

### Key Design Patterns

1. **Constants-based column references**: All DataFrame column names are accessed via constants in `src/constants.py` to ensure consistency and type safety

2. **Feature modules**: Analysis features are organized into subdirectories under `src/` (e.g., `age_group_survival/`). Each feature module contains its utilities and can be imported by notebooks.

3. **Notebook-driven exploration**: Primary analysis is conducted in Jupyter notebooks (in `notebooks/`), which import utilities from `src/` modules

4. **Type hints**: Code uses type hints extensively (e.g., `DataFrame`, `NamedTuple`) for better IDE support and maintainability, with pandas-stubs installed for type checking

### Testing Approach

- Tests use pytest fixtures to load test data (currently uses `.head()` of full training data)
- Test classes organize related tests (e.g., `TestCreateAgeToSurvival`, `TestCreateAgeBinsAndLabels`)
- Tests verify both functionality and data integrity (e.g., checking age values fall within expected bin ranges)
- Tests read from actual data files (`data/train.csv`) to ensure integration with real data

## Type Checking & Linting Configuration

- **Ruff** is configured with line length of 100 characters
- Selected rules: E (pycodestyle errors), W (warnings), F (pyflakes), I (isort), B (flake8-bugbear), C4 (flake8-comprehensions)
- E501 (line too long) is ignored in favor of the line-length setting
- Type stubs for pandas are included via `pandas-stubs` package
