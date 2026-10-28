# AGENTS.md

Python 3.14 project managed with uv: `uv sync`, then `uv run python train.py`.
Smoke test after dependency changes:
`uv run python train.py +trainer.fast_dev_run=5`.

Python, uv are pinned with mise (`mise.toml`). Develop in the container. Dev container (podman + NVIDIA
CDI, mise, host `~/.claude` mounted): `.devcontainer/`, see `docs/devcontainer.md`.
`.venv` belongs to the container; run uv commands there, not on the host.

Keep `docs/` current with the `project-docs` skill: explain decisions and
the why, not every change.
