# 003 — Hydra 1.4 pre-release for Python 3.14

**Date:** 2026-10-27 · **Status:** Accepted

## Context

Python 3.14's `argparse` validates `help=` strings when an argument is
added. Hydra 1.3.x (latest stable, 1.3.8) passes a lazy non-string object
for `--shell-completion`, so *every* `@hydra.main` script crashes at startup:

```
TypeError: argument of type 'LazyCompletionHelp' is not a container or iterable
ValueError: badly formed help string
```

On 2026-10-27, only the 1.4.0 pre-releases (tested: `1.4.0rc3`) work.

## Decision

Depend on `hydra-core>=1.4.0rc3` (uv allows the pre-release because the
specifier names one). Hydra 1.4 no longer accepts `version_base="1.1"`, so
the configs were moved to current Hydra semantics:

- `train.py`: `@hydra.main(...)` without `version_base`.
- Removed `# @package _group_` headers from
  `config/datamodule/{dataset/irmas,transform/default}.yaml`. The group
  package is now the default; 1.4 would read `_group_` as a literal key.
- `config/config.yaml`: `hydra.job.chdir: true`, because `train.py` writes
  TensorBoard logs and checkpoints to `"."` and relies on running inside
  `outputs/<date>/<time>/<experiment>/`. Without it, outputs go to the repo
  root.

Check composition with `uv run python train.py --cfg job` (and `--cfg hydra`).

## Alternatives considered

- **Stable 1.3.8 + a shim** that relaxes `argparse.ArgumentParser._check_help`
  before `hydra.main`: no config changes, but patches a private stdlib method
  and keeps the project on legacy 1.1 semantics.

## Consequences

Pre-release dependency: when Hydra 1.4.0 final ships, change the specifier to
`hydra-core>=1.4` and relock. Related topic doc: `../environment.md`.
