# CallClear Benchmark Results

## Benchmark status

The current measured benchmark is a **2-sample pilot evaluation** using
Hindi FLEURS (`hi_in`) speech.

It covers:

- 2 clean reference utterances
- SNR: 20, 10, 5, and 0 dB
- Enhancement methods:
  - none
  - highpass
  - spectral subtraction
  - Wiener
  - VAD
- 40 non-clean evaluations
- Faster-Whisper `base` for the initial pilot run
- Raw WER
- Normalized WER
- Latency
- Real-time factor (RTF)

The pilot was used to validate the complete benchmark pipeline and
compare the behavior of the enhancement methods under controlled
telephony degradation.

This pilot is **not large enough to support statistical claims about
overall enhancement performance**. A larger benchmark is required for
those conclusions.

---

## Important ASR finding

The initial pilot used Faster-Whisper `base`.

The clean Hindi samples exposed a model-level limitation: although the
model correctly detected the language as Hindi, its transcription was
frequently emitted using Urdu/Arabic script rather than Devanagari.

For example, the reference for sample 0 was Hindi Devanagari:

> कुछ अणुओं में अस्थिर केंद्रक होता है जिसका मतलब यह है कि उनमें थोड़े या बिना किसी झटके से टूटने की प्रवृत्ति होती है

while the `base` model produced Urdu-script output.

A controlled test using Faster-Whisper `small` on the same audio produced
Devanagari Hindi output instead.

Therefore, the `base` pilot results should not be interpreted as a
reliable measurement of Hindi ASR robustness.

For a future larger benchmark, the `small` model is the preferred
starting point.

---

## Pilot clean baseline

The two clean utterances produced:

| Sample | Raw WER | Normalized WER | Latency |
|---|---:|---:|---:|
| 0 | 1.125 | 1.125 | 0.84 s |
| 1 | 1.025 | 1.025 | 2.11 s |

Aggregate clean baseline:

| Metric | Value |
|---|---:|
| Raw WER | 1.075 |
| Normalized WER | 1.075 |
| p50 latency | 1.47 s |
| p95 latency | 2.05 s |
| Mean RTF | 0.122 |

The high clean WER is primarily an indication that the initial
`base`-model configuration is unsuitable for reliable Hindi
Devanagari evaluation.

---

## Pilot observations

### 20 dB

For sample 0:

- none: normalized WER 1.000
- highpass: normalized WER 1.000
- spectral subtraction: normalized WER 0.833
- Wiener: normalized WER 1.000
- VAD: normalized WER 0.833

For sample 1:

- none: normalized WER 1.050
- highpass: normalized WER 1.050
- spectral subtraction: normalized WER 1.000
- Wiener: normalized WER 1.050
- VAD: normalized WER 1.125

The results vary substantially between utterances, so no enhancement
method can be declared superior from this pilot.

### 10 dB

Both samples generally produced normalized WER values close to 1.0,
with noticeable latency variation between enhancement methods.

### 5 dB

Recognition remained poor for the `base` model across most conditions.

### 0 dB

Recognition was again generally poor, with normalized WER near 1.0 for
most conditions.

---

## What the pilot demonstrates

The pilot successfully demonstrates that CallClear can:

1. Load cached Hindi speech.
2. Apply controlled telephony degradation.
3. Sweep multiple SNR conditions.
4. Apply multiple DSP enhancement methods.
5. Run Faster-Whisper ASR.
6. Calculate raw and normalized WER.
7. Measure latency and RTF.
8. Generate CSV, JSON, and visualization artifacts.
9. Fall back to local storage when MongoDB is unavailable.

Generated artifacts:

- `results/benchmark_results.csv`
- `results/benchmark_summary.json`
- `results/wer_vs_snr.png`

---

## Limitations

The current results have several important limitations:

- Only two utterances were evaluated.
- The initial pilot used Faster-Whisper `base`.
- The `base` model produced Urdu-script output for Hindi speech.
- The pilot therefore cannot establish the true effectiveness of the
  enhancement algorithms for Hindi ASR.
- MongoDB was unavailable during the pilot, so results were saved
  locally.
- No statistical significance testing was performed.
- The results should not be generalized to all Indian telephony audio.

---

## Future benchmark

A larger evaluation should use the `small` Whisper model or another
validated Hindi ASR configuration and evaluate at least 100 utterances.

The planned full experiment is:

- 100–200 Hindi utterances
- 20, 10, 5, and 0 dB SNR
- none, highpass, spectral subtraction, Wiener, and VAD
- raw and normalized WER
- latency p50/p95
- real-time factor
- reproducible random seed

The purpose of the larger experiment would be to determine whether
classical DSP enhancement provides a consistent improvement over the
degraded-audio baseline.

Until that experiment is completed, **no enhancement method is claimed
to be the overall winner**.