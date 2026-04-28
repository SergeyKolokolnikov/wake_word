# Beni wake-word model

On-device wake-word detector that fires on the phrase **"Hey Beni"**,
targeting the Waveshare ESP32-S3-AUDIO-Board. The output artifact is a
~30 kB int8 streaming TFLite graph consumed by the firmware's
`main/wake_word/` module via `esp-tflite-micro`.

## Design choices, briefly

- **Framework:** [microWakeWord](https://github.com/kahrendt/microWakeWord)
  (MixedNet on 40-band log-Mel features, 10 ms hop). Chosen over
  Espressif's proprietary WakeNet because we want in-house control of the
  word list — Espressif's custom-model service is paid, gated, and a
  1–2 week round trip.
- **Positives:** 10 000 synthetic clips via
  [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator).
  The shipped `en_US-libritts_r-medium` voice is multi-speaker, so one
  generation pass already covers hundreds of distinct vocal timbres.
  On top of that we sample `--noise-scales` / `--length-scales` /
  `--noise-scale-ws` from explicit ranges (instead of the package
  defaults of one fixed value each) — adds expressiveness and
  speaking-rate variation without needing a second voice model. English
  phonetic spelling ("hey beni") works fine — the detector is acoustic,
  not linguistic.
- **Negatives:** pre-generated negative spectrogram sets hosted by the
  microWakeWord author on HuggingFace (`kahrendt/microwakeword`). Using
  Kevin's sets guarantees the spectrogram front-end matches what the
  firmware runs — the #1 way wake-word models regress is a
  training-vs-inference front-end mismatch.
- **Augmentation:** microWakeWord's built-in `Augmentation` with
  MIT-impulse-response reverb, AudioSet background noise, and FMA music
  backdrop. Standard recipe, no custom pipeline.
- **Training backend:** the upstream CLI `python -m
  microwakeword.model_train_eval` reading a YAML config. We deliberately
  do **not** invoke the MixedNet class directly — the package is written
  as a CLI trainer and the class/function API is not part of its
  contract (it moves between versions).
- **Why not always-on server-side STT?** Reviewed and rejected — costs
  $450–1800/device/month depending on STT vendor, streams full-day audio
  to the cloud (privacy + regulatory headache with vulnerable-adult
  users), and kills device battery. See conversation log 2026-04-23.

## Expected quality

Pure-synthetic training gives:
- False reject ~5–15 % on known-voice profiles, higher on elderly /
  strong-accent speakers (the actual target audience — this matters).
- False accept a few times a day in quiet rooms, more with TV/radio.
- Tunable by threshold; the firmware exposes one config knob.

Plan to iterate once we have real recordings from the device. **Do not
ship V1 without a real-hardware listening test.**

## Where training runs: Google Colab

The full pipeline lives inside **`train_beni.ipynb`** and is meant to
run on a Colab T4 GPU. Training locally is possible but painful — a
Colab run finishes end-to-end in ~1–2 h (most of it is the AudioSet
download, not the training itself).

**How to use it:**

1. Push this folder to GitHub (`SergeyKolokolnikov/wake_word`).
2. In Colab: `File → Open notebook → GitHub`, pick
   `train_beni.ipynb`.
3. Runtime → Change runtime type → **T4 GPU**.
4. `Run all`. After the install cell Colab will prompt to restart the
   session — do it, then continue.
5. The last cells trigger downloads of
   `stream_state_internal_quant.tflite` and `hey_beni_model.h`. Drop
   both into `firmware/main/wake_word/model/` and rebuild firmware.

## Directory layout

```
ml/wake_word/
├── README.md             ← this file
├── train_beni.ipynb      ← Colab end-to-end training notebook
├── scripts/
│   └── dump_c_header.py  ← wraps the trained .tflite into a C header
└── artifacts/            ← committed .tflite / .h outputs
```

All intermediate data (positives, negatives, mmap features, trained
checkpoints) lives only inside the Colab runtime; we commit only the
final exported model, never raw data.

## Pipeline (what the notebook does)

1. **Install microWakeWord** from git (it's not on PyPI) +
   piper-sample-generator.
2. **Generate 2000 "hey beni" WAVs** via piper-sample-generator.
3. **Download augmentation sources** — MIT RIRs, one AudioSet shard,
   a small FMA slice.
4. **Configure `Clips` + `Augmentation`** to mix them.
5. **Materialise augmented spectrograms** as RaggedMmap features
   (train / val / test splits).
6. **Download pre-generated negative spectrograms** from
   `kahrendt/microwakeword` on HF (speech, dinner-party, no-speech,
   plus a held-out dinner-party eval set).
7. **Write `training_parameters.yaml`** describing the sampling mix.
8. **Train MixedNet** via `python -m microwakeword.model_train_eval`.
9. **Dump a C header** via `scripts/dump_c_header.py` so the firmware
   can embed the model in flash.

## Reproducibility

All versions pinned in the install cell. The notebook is fully
sequential and idempotent — re-running from scratch produces the same
tflite modulo Piper's sampling randomness (which we do **not** seed;
for bit-identical reruns set `--random-seed` on piper-sample-generator).
