# CallClear

[![CI](https://github.com/Siddharth-21s/callclear/actions/workflows/ci.yml/badge.svg)](https://github.com/Siddharth-21s/callclear/actions/workflows/ci.yml)

CallClear is a Python-based speech engineering project for evaluating automatic speech recognition (ASR) robustness on Indian telephony-style audio.

The system takes clean Hindi speech, simulates a degraded telephone channel, optionally applies DSP-based enhancement, runs Faster-Whisper ASR, and measures recognition accuracy and inference performance.

The benchmark is designed to answer a practical question:

> How much does telephone-channel degradation affect ASR, and do classical DSP enhancement methods actually improve recognition?

The project intentionally reports measured results rather than assuming that enhancement helps.

---

## Project status

The benchmark infrastructure is implemented and reproducible.

A **2-sample pilot benchmark has been completed** across:

- SNR: `20, 10, 5, 0 dB`
- Enhancement: `none`, `highpass`, `spectral_subtraction`, `wiener`, `vad`
- Faster-Whisper `base`
- Raw WER
- Normalized WER
- Latency
- p50/p95 latency
- Real-time factor (RTF)

The pilot exposed an important ASR configuration issue: the tested Faster-Whisper `base` model produced Hindi speech in Urdu/Arabic script for the tested samples. A controlled test with the `small` model produced Devanagari output on the same type of Hindi audio.

Because the pilot contains only two utterances, it is **not treated as a final benchmark** and no enhancement method is declared a winner.

The larger 100–200 utterance benchmark remains pending.

Detailed pilot findings are documented in [`docs/results.md`](docs/results.md).

---

## Problem

Clean speech is not the same as telephone speech.

Real telephony audio can introduce:

- reduced bandwidth
- sampling-rate changes
- quantization and companding effects
- background noise
- packet-loss gaps
- reduced speech intelligibility

CallClear creates a controlled approximation of these effects so that ASR robustness can be measured reproducibly.

---

## Pipeline

```mermaid
flowchart LR

    A[Clean Hindi speech<br/>FLEURS hi_in] --> B[Telephony simulator]

    B --> C[8 kHz resampling]
    C --> D[300–3400 Hz band limit]
    D --> E[μ-law companding]
    E --> F[Controlled noise]
    F --> G[Packet-loss simulation]
    G --> H[Degraded WAV]

    H --> I[DSP enhancement]
    I --> J[Enhanced WAV]

    A --> K[Faster-Whisper]
    H --> K
    J --> K

    K --> L[WER / latency / RTF]
    L --> M[CSV + JSON + chart]

    N[FastAPI] --> B
    N --> I
    N --> K

    M --> O[Optional MongoDB storage]