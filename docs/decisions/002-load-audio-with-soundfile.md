# 002 — Load audio with soundfile, not `torchaudio.load`

**Date:** 2026-10-27 · **Status:** Accepted

## Context

Since torchaudio 2.9, `torchaudio.load` is a thin wrapper around
TorchCodec (`load_with_torchcodec`). Without the `torchcodec` package it
fails with:

```
ImportError: TorchCodec is required for load_with_torchcodec.
```

TorchCodec in turn needs system FFmpeg shared libraries of a compatible
version. IRMAS is plain 16-bit WAV, which needs none of that.

## Decision

`IRMASDataset.load_audio` (`irmas_torch/datasets.py`) reads files with
`soundfile.read(fn, dtype="float32", always_2d=True)` and transposes to the
`(channels, samples)` layout `torchaudio.load` used to return. The rest of
torchaudio (`Resample`, `MelSpectrogram`) is still used for the transforms;
those are pure tensor ops and unaffected.

## Alternatives considered

- **Add `torchcodec`**: extra dependency plus an FFmpeg ABI coupling to the
  OS, for a format libsndfile handles natively.
- **`scipy.io.wavfile`**: already a dependency, but returns int16 that must
  be rescaled by hand and is stricter about WAV variants.

## Consequences

`soundfile` ships its own libsndfile in the wheel, so no system packages are
needed. If non-WAV audio (mp3, etc.) is ever added, revisit TorchCodec.
Related topic doc: `../environment.md`.
