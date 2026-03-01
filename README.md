# Article_OCT_Diabetic_neuropathy

Research project on OCT for early diagnostics of diabetic neuropathy.

## Repository Structure

```
├── src/                 # Python source code
├── data/
│   ├── raw/             # Raw input data (not tracked by git)
│   └── processed/       # Processed/cleaned data (not tracked by git)
├── results/
│   ├── figures/         # Generated PNG figures (not tracked by git)
│   └── tables/          # Generated CSV result tables (not tracked by git)
├── docs/
│   └── TODO.md          # Shared todo list
├── pyproject.toml       # Python project config (uv)
└── .gitignore
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/) as the Python package manager.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Add a new dependency
uv add <package>

# Add a dev dependency (e.g. jupyter)
uv add --dev <package>
```
