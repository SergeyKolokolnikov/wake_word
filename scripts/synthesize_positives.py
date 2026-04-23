"""
Generate synthetic "Beni" / "Hey Beni" WAVs for wake-word training.

Design decisions worth pointing out:

* We train on BOTH "Beni" and "Hey Beni". The short form matters most —
  elderly users don't naturally prepend "hey" to every sentence. The
  two-word form is cheap to add and broadens the trigger window.
* Variants include a trailing comma (",") and an em-dash because Piper
  reads punctuation prosodically — a trailing comma produces the rising
  intonation of "Beni, recommend a recipe" which is our most common
  real-world trigger.
* We sweep `length_scale` (speaking rate) and `noise_scale` (prosody
  jitter) so the resulting set isn't a single Piper fingerprint that
  the classifier just memorises.

Voices are picked on breadth, not quality: multiple English accents
(US, GB, AU, IN, ZA if available), both sexes, a few child-ish and
older-sounding voices where Piper has them. Anything marked "high"
quality gets priority because low-quality voices produce audible
robotic artifacts that don't transfer to real speech.

All synthesis is single-threaded on purpose — Piper is cheap per
utterance (~100-300 ms on Colab CPU), and a worker pool mainly creates
log noise without meaningful speedup below ~10 k samples.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# Phrasings we train on. The model fires if ANY of these land in the
# input window — the user doesn't have to pick a specific one.
PHRASINGS: tuple[str, ...] = (
    "Beni",
    "Beni,",
    "Beni.",
    "Hey Beni",
    "Hey Beni,",
    "Hey Beni.",
    "Hey, Beni",
    "Hey, Beni,",
)

# Piper voices to sweep. Format is <lang>-<name>-<quality>.
# We deliberately stay inside English — multi-language wake words
# generalise poorly on a model this small. Order matters only for
# reproducibility; the synthesis loop shuffles.
VOICES: tuple[str, ...] = (
    # US English
    "en_US-amy-medium",
    "en_US-danny-low",
    "en_US-kathleen-low",
    "en_US-lessac-medium",
    "en_US-lessac-high",
    "en_US-libritts-high",
    "en_US-libritts_r-medium",
    "en_US-ryan-medium",
    "en_US-ryan-high",
    "en_US-kusal-medium",
    "en_US-joe-medium",
    # GB English
    "en_GB-alan-low",
    "en_GB-alan-medium",
    "en_GB-cori-medium",
    "en_GB-cori-high",
    "en_GB-jenny_dioco-medium",
    "en_GB-northern_english_male-medium",
    "en_GB-semaine-medium",
    "en_GB-vctk-medium",
)

# Prosody sweep. length_scale > 1.0 is slower speech; < 1.0 is faster.
# noise_scale controls how much expressive jitter Piper injects.
LENGTH_SCALES: tuple[float, ...] = (0.85, 1.0, 1.15, 1.3)
NOISE_SCALES: tuple[float, ...] = (0.4, 0.667, 0.85)


@dataclass
class SynthesisPlan:
    """Everything the notebook needs to reproduce this synthesis run.

    Written out as manifest.json so we can tell, post-hoc, exactly which
    voices and prosody combos produced the shipped model.
    """

    seed: int
    samples_per_combo: int
    voices: list[str]
    phrasings: list[str]
    length_scales: list[float]
    noise_scales: list[float]
    produced_files: list[str] = field(default_factory=list)
    voice_download_urls: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "samples_per_combo": self.samples_per_combo,
            "voices": self.voices,
            "phrasings": self.phrasings,
            "length_scales": self.length_scales,
            "noise_scales": self.noise_scales,
            "num_files": len(self.produced_files),
        }


def _voice_download_url(voice: str) -> tuple[str, str]:
    """Map a Piper voice id to its onnx + json URLs on Hugging Face.

    Piper voices live at rhasspy/piper-voices on HF and follow a strict
    directory naming. We construct URLs instead of shelling out to
    `piper --download` because HF sometimes rate-limits the CLI and
    we'd rather fail loudly with a URL than hang.
    """

    lang_region, name_quality = voice.split("-", 1)
    lang = lang_region.split("_")[0]
    base = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
        f"{lang}/{lang_region}/{name_quality.rsplit('-', 1)[0]}/"
        f"{name_quality.rsplit('-', 1)[1]}"
    )
    onnx = f"{base}/{voice}.onnx"
    meta = f"{base}/{voice}.onnx.json"
    return onnx, meta


def download_voices(voices: Iterable[str], voices_dir: Path) -> dict[str, str]:
    """Fetch Piper voice onnx+json pairs into voices_dir.

    Returns a map of voice_id -> onnx URL for the manifest. Skips
    voices already present on disk so the notebook is re-runnable.
    """

    voices_dir.mkdir(parents=True, exist_ok=True)
    urls: dict[str, str] = {}
    for voice in voices:
        onnx_path = voices_dir / f"{voice}.onnx"
        json_path = voices_dir / f"{voice}.onnx.json"
        onnx_url, json_url = _voice_download_url(voice)
        urls[voice] = onnx_url

        for url, path in ((onnx_url, onnx_path), (json_url, json_path)):
            if path.exists() and path.stat().st_size > 0:
                continue
            # -L follows the HF redirect; --fail makes 4xx/5xx surface
            # as a non-zero exit so we don't silently cache an HTML
            # error page as if it were a voice file.
            subprocess.run(
                ["curl", "-fL", "-o", str(path), url],
                check=True,
            )
    return urls


def _synthesize_one(
    text: str,
    voice_onnx: Path,
    out_wav: Path,
    length_scale: float,
    noise_scale: float,
) -> None:
    """Run piper once for a single (text, voice, prosody) tuple.

    We pipe text via stdin so punctuation-heavy phrasings don't get
    mangled by shell quoting. Piper writes a 16 kHz mono int16 WAV,
    which is exactly microWakeWord's expected input format — no
    resampling step needed.
    """

    subprocess.run(
        [
            "piper",
            "--model", str(voice_onnx),
            "--output_file", str(out_wav),
            "--length_scale", str(length_scale),
            "--noise_scale", str(noise_scale),
        ],
        input=text.encode("utf-8"),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def synthesize_positives(
    out_dir: Path,
    voices_dir: Path,
    seed: int = 42,
    samples_per_combo: int = 3,
) -> SynthesisPlan:
    """Produce the full positive training set.

    samples_per_combo controls how many duplicates we render for each
    (voice, phrasing, length_scale, noise_scale) tuple. Duplicates are
    useful because Piper's internal noise makes each render slightly
    different — more samples = more natural-sounding variety, at the
    cost of longer synthesis. 3 is a reasonable default that yields
    ~4000 WAVs with the current voice/phrasing lists.
    """

    random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = SynthesisPlan(
        seed=seed,
        samples_per_combo=samples_per_combo,
        voices=list(VOICES),
        phrasings=list(PHRASINGS),
        length_scales=list(LENGTH_SCALES),
        noise_scales=list(NOISE_SCALES),
    )
    plan.voice_download_urls = download_voices(VOICES, voices_dir)

    combos: list[tuple[str, str, float, float]] = [
        (voice, phrase, length, noise)
        for voice in VOICES
        for phrase in PHRASINGS
        for length in LENGTH_SCALES
        for noise in NOISE_SCALES
    ]
    random.shuffle(combos)

    total = len(combos) * samples_per_combo
    print(f"Synthesising {total} positive clips across {len(VOICES)} voices...")

    for idx, (voice, phrase, length_scale, noise_scale) in enumerate(combos):
        voice_onnx = voices_dir / f"{voice}.onnx"
        for k in range(samples_per_combo):
            # Filename encodes the full config — easier to debug "why
            # did this one misfire" later without needing a separate
            # index file.
            digest = hashlib.md5(
                f"{voice}|{phrase}|{length_scale}|{noise_scale}|{k}".encode()
            ).hexdigest()[:10]
            out_wav = out_dir / f"pos_{digest}.wav"
            if out_wav.exists():
                plan.produced_files.append(str(out_wav.name))
                continue

            try:
                _synthesize_one(
                    phrase, voice_onnx, out_wav, length_scale, noise_scale
                )
                plan.produced_files.append(str(out_wav.name))
            except subprocess.CalledProcessError as exc:
                # One bad voice shouldn't kill the run — log and carry
                # on. The final count in the manifest tells us how many
                # actually made it.
                print(
                    f"[warn] {voice} failed on '{phrase}' "
                    f"len={length_scale} noise={noise_scale}: {exc}",
                    file=sys.stderr,
                )

        if (idx + 1) % 25 == 0:
            print(
                f"  progress: {idx + 1}/{len(combos)} combos, "
                f"{len(plan.produced_files)} wavs so far"
            )

    print(
        f"Done. {len(plan.produced_files)} positive wavs in {out_dir} "
        f"(target was {total})"
    )
    return plan


def write_manifest(plan: SynthesisPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.to_dict(), indent=2))


if __name__ == "__main__":  # pragma: no cover - manual invocation only
    root = Path(os.environ.get("WW_ROOT", "/content/wake_word"))
    out = root / "data" / "positive"
    voices = root / "voices"
    manifest = root / "artifacts" / "positives_manifest.json"

    plan = synthesize_positives(out, voices)
    write_manifest(plan, manifest)
