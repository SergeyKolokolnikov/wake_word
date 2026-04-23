# Beni wake-word model

On-device wake-word detector trained to fire on the word **"Beni"** (and the
phrase "Hey Beni"), targeting the Waveshare ESP32-S3-AUDIO-Board. The
output artifact is a ~30 kB int8 TFLite graph consumed by the firmware's
`main/wake_word/` module via `esp-tflite-micro`.

We deliberately train on the bare word "Beni" as well as "Hey Beni", so
the end user does not have to prefix every sentence with a wake phrase —
"Beni, recommend a recipe" must fire just like "Hey Beni, …". Short
single-word triggers are on the edge of what wake-word models handle
reliably, so the training set leans heavy on this pattern.

## Design choices, briefly

- **Framework:** [microWakeWord](https://github.com/kahrendt/microWakeWord)
  (MixedNet on 40-band log-Mel features, 10 ms hop). Chosen over
  Espressif's proprietary WakeNet because we want in-house control of the
  word list — Espressif's custom-model service is paid, gated, and a
  1–2 week round trip.
- **Positives:** pure-synthetic via [Piper TTS](https://github.com/rhasspy/piper).
  Many voices, speeds, pitches, pauses, punctuation variants. Eventually
  supplemented by real recordings once we have a device in hand.
- **Negatives:** Mozilla Common Voice English subset (short clips, varied
  speakers) — the standard microWakeWord negative pool.
- **Augmentation:** room-impulse-response convolution + MUSAN/ESC-50 noise
  mixing at varied SNR, plus light speed/pitch perturbation. Standard
  microWakeWord recipe.
- **Why not always-on server-side STT?** Reviewed and rejected — costs
  $450–1800/device/month depending on STT vendor, streams full-day audio
  to the cloud (privacy + regulatory headache with vulnerable-adult
  users), and kills device battery. See conversation log 2026-04-23.

## Expected quality (be honest with yourself)

Pure-synthetic training gives:
- False reject ~5–15 % on known-voice profiles, higher on elderly /
  strong-accent speakers (the actual target audience — this matters).
- False accept a few times a day in quiet rooms, more with TV/radio.
- Tunable by threshold; the firmware will expose one config knob.

Plan to iterate once we have real recordings from the device. Do not
ship V1 without a real-hardware listening test.

## Where training runs: Google Colab

The full pipeline lives inside **`train_beni.ipynb`** and is meant to run
on a free Colab T4. Training the model locally is possible (TensorFlow
supports Apple Metal on M-series chips) but slow — a Colab run finishes
the whole pipeline end to end in ~30–45 minutes, and keeps ~3 GB of
TF / Piper voices off the dev machine.

**How to use it:**

1. Commit this folder, push to GitHub.
2. In Colab: `File → Open notebook → GitHub`, paste the repo URL, pick
   `ml/wake_word/train_beni.ipynb`.
3. Runtime → Change runtime type → **T4 GPU**.
4. `Run all`. The last cell auto-downloads `hey_beni.tflite` to your
   computer.
5. Drop that file into `firmware/main/wake_word/model/` and rebuild
   firmware.

## Directory layout

```
ml/wake_word/
├── README.md                ← this file
├── train_beni.ipynb         ← Colab-first end-to-end training notebook
├── scripts/                 ← Python helpers the notebook imports
│   ├── synthesize_positives.py
│   ├── fetch_negatives.py
│   └── export_model.py
└── artifacts/               ← committed .tflite models (tracked in git)
```

Dataset directories (`data/positive`, `data/negative`, `data/augmented`)
exist only inside the Colab runtime and are never committed — the
notebook rebuilds them deterministically from the committed seed.

## Pipeline (what the notebook does)

1. **Install deps** — `piper-tts`, `microwakeword`, `librosa`,
   `datasets` (for Common Voice streaming).
2. **Synthesize positives** — `scripts/synthesize_positives.py` loops
   over ~15 Piper voices × {"beni", "hey beni", "beni,", "hey beni,"} ×
   prosody variants (speaker scale, speed, noise) → ~3–5 k WAVs at
   16 kHz mono.
3. **Fetch negatives** — stream a ~20 h English subset of Mozilla
   Common Voice via `datasets`.
4. **Augment** — delegated to microWakeWord's built-in augmenter
   (RIR + MUSAN noise + SNR sweep).
5. **Train** — microWakeWord `MixedNet` config, ~30 epochs, GPU.
6. **Export** — int8 TFLite → `hey_beni.tflite` + a short metrics
   report (FRR at fixed FAR).
7. **Download** — `files.download(...)` sends the .tflite to the host
   browser; also saves to the notebook's `/content/artifacts/`.

## Reproducibility

The notebook pins all pip versions in cell 1 and uses a fixed
`RANDOM_SEED=42`. The resolved Piper voice list + Common Voice subset
checksums are written to `artifacts/manifest.json`, which we commit
alongside the .tflite.
