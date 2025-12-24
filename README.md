# ChoCH-BOS Strategy

A minimal repository scaffold for developing a Change of Character (ChoCH) and Break of Structure (BOS) trading strategy. Use this project to research entry/exit rules, design experiments, and track backtesting results.

## Repository layout
- `src/` – strategy code, utilities, and orchestrators.
- `tests/` – automated checks for strategy logic and helpers.
- `notes/` – research notes, hypotheses, and observations.
- `data/` – local datasets such as price history or indicator exports (ignored from version control, except for the placeholder file).

## Getting started
1. Create a Python virtual environment (e.g., `python -m venv .venv`) and activate it.
2. Install dependencies once they are defined (for example, via `pip install -r requirements.txt`).
3. Add your strategy implementation under `src/` and corresponding tests in `tests/`.
4. Run your test suite (e.g., `pytest`) to validate any changes.

## Next steps
- Define the initial ChoCH/BOS rule set and success metrics.
- Add data loaders for your preferred broker or CSV exports.
- Implement backtests and performance reporting utilities.
- Automate notebook- or script-based experiments and capture results in `notes/`.
