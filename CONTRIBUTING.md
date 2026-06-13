# Contributing

## Workflow

1. Fork the repository and clone your fork, or ask to be added as a collaborator.
2. Create a branch from `main` (or the default branch): `git checkout -b topic/short-description`.
3. Install dependencies and the editable package: `pip install -e ".[train]"` (see `README.md`).
4. Run or edit notebooks under `notebooks/`; keep outputs and checkpoints under `outputs/` or local paths not committed to git.
5. Open a pull request with a short description of what changed and how you verified it.

## Conventions

- Prefer environment variable `RAICOM_DATA_ROOT` for dataset path when sharing machine-specific setups.
- Do not commit large raw datasets or secrets (`.env` is ignored). Checkpoints and figures under `outputs/` are meant to be versioned for the competition unless the repo owner decides otherwise.
