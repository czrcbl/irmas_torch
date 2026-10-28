# Dev container (podman + NVIDIA GPU)

`.devcontainer/` is the development environment for this project; nothing
is run on the host. It is Debian trixie with the host's NVIDIA GPU,
[mise](https://mise.jdx.dev) for tools (Python, uv, Claude Code), and your
host's Claude Code config. It is meant for
**rootless podman**. The repo, including `data/IRMAS`, is bind-mounted, so the dataset is
not copied into the image.

## One-time host setup

The GPU is passed in through **CDI** (Container Device Interface). CDI is a
YAML spec, generated from the installed driver, that lists the device nodes
and driver libraries a container needs. Podman reads it from `/etc/cdi/`.

```sh
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list                     # should list nvidia.com/gpu=all
podman run --rm --device nvidia.com/gpu=all ubuntu nvidia-smi   # sanity check
```

Regenerate the spec after every NVIDIA driver update; it names exact library
versions. If you don't, containers fail to start with a CDI injection error.

`~/.claude` must exist on the host (run `claude` once). The mount fails if
the source directory is missing.

## Using it

```sh
# CLI
devcontainer up   --workspace-folder . --docker-path podman
devcontainer exec --workspace-folder . --docker-path podman uv run python train.py

# VS Code: set "dev.containers.dockerPath": "podman", then
# "Dev Containers: Reopen in Container".
```

`postCreateCommand` runs `mise trust && mise install && uv sync`. Every
start prints whether CUDA is visible (`postStartCommand`). Run `claude` in the
container terminal; it uses your host login and settings.

## Tools via mise

- **Python** and **uv** are pinned in the project's `mise.toml`. mise
  installs Python as a prebuilt
  [python-build-standalone](https://github.com/astral-sh/python-build-standalone)
  build, so there is no compile step, and the exact patch version is in the
  repo instead of depending on the base image. To bump it, change
  `mise.toml`, run `mise install`, then `uv sync`. uv sees that `.venv`
  points to the old interpreter and recreates it. Keep `requires-python` and
  `.python-version` (read by uv) compatible with it.
- Keep uv in step with `uv_build>=…,<0.14` in `pyproject.toml`.
- **Claude Code** (`claude = "latest"`) is in the container user's global
  mise config (`~/.config/mise/config.toml`, written by the Containerfile). It
  is not in the project `mise.toml`, so it doesn't override the host's own
  Claude pin. "latest" is resolved on the first `mise install`. Get a newer
  version with `mise up claude`, or pin a version in the Containerfile.
- The mise version itself is pinned by `ARG MISE_VERSION` in the
  Containerfile. Match it to the host (`mise --version`).

## Claude Code config

The host's `~/.claude` (settings, login in `.credentials.json`, skills,
session history, memory) is bind-mounted at `/home/vscode/.claude`.
Consequences:

- Anything Claude in the container does to that directory changes it on the
  host too, credentials included. That is the point of sharing it, but it
  also means the container is not isolated from your Claude account.
- `~/.claude.json` (onboarding and per-project state) is a single file next
  to the directory, not inside it. It is **not** mounted, because tools that
  rewrite a file by rename fail on a single-file bind mount. The container
  keeps its own copy, which is reset when the container is recreated.
- Session history and memory are keyed by the project's absolute path. The
  container sees the repo at `/workspaces/irmas_torch`, the host at
  `/home/cezar/Projects/irmas_torch`, so the two don't share session history
  or project memory.

## Why it is set up this way

| Setting | Why |
|---------|-----|
| `--device=nvidia.com/gpu=all` | CDI is podman's native GPU path. `hostRequirements.gpu` / `--gpus` is Docker's mechanism and relies on the nvidia runtime hook. |
| `devcontainers/base` image, no Python or CUDA image | Python comes from mise. The PyPI torch wheels bundle CUDA and cuDNN, so the container only needs the driver libraries, which CDI injects from the host. |
| `--userns=keep-id:uid=1000,gid=1000` + `updateRemoteUserUID: false` | Maps your host user to the image's `vscode` user (uid 1000). Files created in the bind mount stay owned by you on the host. The devcontainer CLI's own UID rewrite is not needed. |
| `--security-opt=label=disable` | Keeps SELinux from blocking the bind mount and GPU devices on Fedora-like hosts. No effect on Arch. |
| `--shm-size=8g` | DataLoader workers pass batches through `/dev/shm`. Podman's 64 MB default crashes with `num_workers: 8`. |
| venv in the project's `.venv` | The container owns it. Do not run `uv sync`/`uv run` on the host: a venv stores absolute paths to its interpreter (`.venv/bin/python` → `~/.local/share/mise/installs/python/…`, `pyvenv.cfg`) and to itself (script shebangs), so uv on the host would see a broken venv and recreate it for the host's Python, and the container would then recreate it again. A venv left over from the host is replaced on the first `uv sync` in the container. |
| uv cache in the `irmas-torch-uv-cache` volume, `UV_LINK_MODE=copy` | torch + CUDA wheels are several GB. The cache survives container rebuilds. Copy mode because the cache (volume) and `.venv` (bind mount) are on different filesystems, so hardlinks are impossible. |
| `UV_PYTHON_DOWNLOADS=never` | uv must use mise's Python, never download its own, so `mise.toml` stays the single source of the version. |
| mise shims on `PATH` (Containerfile `ENV`) | `mise activate` only affects interactive shells. Shims also work for lifecycle commands and `podman exec`. |
| mise installs in the `irmas-torch-mise` volume | Python, uv and Claude Code survive rebuilds. `.venv` links into this volume, so deleting it means re-running `mise install && uv sync`. |
| Containerfile runs as root throughout | It creates the volume mount points and `chown`s them to `vscode`. Who the container runs as is set by `containerUser`/`remoteUser` in `devcontainer.json`, not by the image's `USER`. |

## Related project change

`config/datamodule/dataset/irmas.yaml` uses `root: data/IRMAS`, and
`IRMASDataset` resolves relative roots against the project root. It used to
be an absolute host path, which doesn't exist in the container.

## Resetting

```sh
rm -rf .venv                                        # rebuilt by the next uv sync
podman volume rm irmas-torch-uv-cache irmas-torch-mise   # after removing the container
```
