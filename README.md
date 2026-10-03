# CallClear

[![CI](https://github.com/Siddharth-21s/callclear/actions/workflows/ci.yml/badge.svg)](https://github.com/Siddharth-21s/callclear/actions/workflows/ci.yml)

CallClear is a Python-based speech engineering project for evaluating
automatic speech recognition (ASR) robustness on Indian telephony-style
audio.

The system takes clean Hindi speech, simulates a degraded telephone
channel, optionally applies DSP-based enhancement, runs Faster-Whisper
ASR, and measures recognition accuracy and inference performance.

The benchmark is designed to answer a practical question:

> How much does telephone-channel degradation affect ASR, and do
> classical DSP enhancement methods actually improve recognition?

The project intentionally reports measured results rather than assuming
that enhancement helps.

---

## Project status

The benchmark infrastructure is implemented and reproducible.

The final multi-condition benchmark sweep is still pending, so final
performance numbers are intentionally **not fabricated** in this README.

The planned benchmark evaluates:

- SNR: `20, 10, 5, 0 dB`
- Enhancement: `none`, `highpass`, `spectral_subtraction`, `wiener`, `vad`
- Hindi speech from FLEURS `hi_in`
- Raw WER
- Normalized WER
- Latency
- p50/p95 latency
- Real-time factor (RTF)

The final benchmark results will be added after the complete sweep is
run on the local machine.

---

## Problem

Clean speech is not the same as telephone speech.

Real telephony audio can introduce:

- reduced bandwidth
- sampling-rate changes
- quantization/companding effects
- background noise
- packet-loss gaps
- reduced speech intelligibility

CallClear creates a controlled approximation of these effects so that
ASR robustness can be measured reproducibly.

---

## Pipeline

```mermaid
flowchart LR
    A[Clean Hindi speech<br/>FLEURS hi_in] --> B[Telephony simulator]

    B --> C[8 kHz resampling]
    C --> D[300–3400 Hz band limit]
    D --> E[G.711 μ-law]
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