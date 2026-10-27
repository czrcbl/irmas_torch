# Docs

Explanations of how this project works and why. Not a change log: for
*what* changed, see `git log`.

Start with [architecture.md](architecture.md).

## Topics

| Doc | Covers |
|-----|--------|
| [architecture.md](architecture.md) | How the parts fit together |
| [environment.md](environment.md) | Python 3.14 / uv setup, dependency roles, API changes from the migration, legacy scripts |

## Decisions

| # | Decision | Status |
|---|----------|--------|
| 001 | [uv and PEP 621 instead of Poetry](decisions/001-uv-and-pep-621-instead-of-poetry.md) | Accepted |
| 002 | [Load audio with soundfile, not torchaudio.load](decisions/002-load-audio-with-soundfile.md) | Accepted |
| 003 | [Hydra 1.4 pre-release for Python 3.14](decisions/003-hydra-1-4-pre-release-for-python-3-14.md) | Accepted |
