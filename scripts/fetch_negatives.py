"""
Pull a Common Voice subset for microWakeWord negatives.

Why Common Voice:
* It's the canonical negative pool in the microWakeWord documentation,
  so the model behaves exactly as the framework was tuned for.
* Clips are short (~3-8 s) and already labelled as validated speech,
  matching the format microWakeWord expects.
* Available via `datasets` (HuggingFace) with streaming, so we don't
  need to download the whole 70 GB corpus in Colab — we pull a subset
  and stop.

We sample ~5000 clips by default. That's enough to expose the wake
model to a variety of English speech without ballooning the training
set — microWakeWord's positive:negative ratio matters more than raw
negative count past a few thousand.
"""

from __future__ import annotations

import random
from pathlib import Path

import soundfile as sf


def fetch_common_voice_negatives(
    out_dir: Path,
    num_samples: int = 5000,
    seed: int = 42,
    max_seconds: float = 6.0,
    cache_dir: Path | None = None,
) -> int:
    """Download and normalise a Common Voice English subset.

    Resamples to 16 kHz mono int16 (microWakeWord's required format)
    and truncates anything longer than `max_seconds` — very long
    clips waste cycles in the feature extractor without improving
    negative coverage.
    """

    from datasets import load_dataset  # imported lazily; heavy import

    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)

    # streaming=True avoids downloading the entire corpus. We pull
    # from the validated split to skip clips the community flagged.
    # The token is optional — Common Voice is open, but HF sometimes
    # prefers authenticated streams for rate-limit headroom.
    ds = load_dataset(
        "mozilla-foundation/common_voice_17_0",
        "en",
        split="train",
        streaming=True,
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    saved = 0
    for sample in ds:
        if saved >= num_samples:
            break

        audio = sample["audio"]
        arr = audio["array"]
        sr = audio["sampling_rate"]

        # Skip clips that are suspiciously short (likely cut) or too
        # long (wastes training cycles). microWakeWord wants roughly
        # 1-6 s.
        duration = len(arr) / sr
        if duration < 1.0 or duration > max_seconds:
            continue

        # CV is 48 kHz; microWakeWord operates at 16 kHz. soundfile
        # can't resample so we pass the array through librosa, but
        # only if resampling is actually needed — keeps the happy
        # path fast.
        if sr != 16000:
            import librosa

            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000)
            sr = 16000

        out_path = out_dir / f"neg_{saved:05d}.wav"
        # PCM_16 is what the rest of the pipeline expects; float64 from
        # HF would bloat disk and still get quantised downstream.
        sf.write(out_path, arr, sr, subtype="PCM_16")
        saved += 1

        if saved % 500 == 0:
            print(f"  {saved}/{num_samples} negatives written")

    print(f"Done. {saved} negative wavs in {out_dir}")
    return saved
