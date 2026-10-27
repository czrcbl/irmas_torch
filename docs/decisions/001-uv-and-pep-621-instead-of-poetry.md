# 001 — uv and PEP 621 `[project]` instead of Poetry

**Date:** 2026-10-27 · **Status:** Accepted

## Context

The project used Poetry (`[tool.poetry]`, `poetry.lock`) pinned to Python
3.10. When migrating to Python 3.14 the lock file had to be regenerated from
scratch anyway (torch 1.12 → 2.x etc.), Poetry was not installed on the dev
machine, and `uv` was.

## Decision

Declare metadata in the standard `[project]` table, dev tools in
`[dependency-groups]`, build with `uv_build`, and lock with `uv.lock`.
`.python-version` contains `3.14` (it used to name a pyenv virtualenv,
`irmas`).

```sh
uv sync                    # create .venv and install everything
uv run python train.py     # run inside the env
uv add <pkg>               # add a dependency and relock
```

## Alternatives considered

- **Keep Poetry**: would need installing Poetry just to regenerate a lock that
  was being thrown away; `[tool.poetry]` is non-standard metadata.
- **Plain pip + requirements.txt**: no lock with hashes, no Python version
  management.

## Consequences

Any PEP 621 tool (pip, hatch, Poetry 2) can read the metadata; only the lock
file is uv-specific. Old Poetry files are in git history (commit before this
change). Related topic doc: `../environment.md`.
