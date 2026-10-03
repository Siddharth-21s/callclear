# CallClear

CallClear benchmarks automatic speech recognition (ASR) robustness on Indian telephony-style audio and evaluates whether DSP-based audio enhancement improves recognition.

The project takes clean Hindi speech, simulates a degraded telephone channel, runs ASR on clean/degraded/enhanced audio, and measures word error rate (WER) and inference performance.

## Problem

Clean speech is not the same as telephone speech.

CallClear simulates a conservative telephony channel by:

- Resampling audio to 8 kHz
- Band-limiting speech to approximately 300–3400 Hz
- Applying G.711 μ-law companding
- Adding controlled noise
- Simulating short packet-loss gaps

The resulting degraded audio is passed through a DSP enhancement stage and then through Faster-Whisper.

The benchmark compares:

1. Clean speech
2. Telephony-degraded speech
3. DSP-enhanced speech

The project intentionally reports measured results rather than assuming that enhancement improves ASR.

## Architecture

```mermaid
flowchart LR
    A[Clean Hindi speech<br/>FLEURS hi_in] --> B[Telephony simulator]

    B --> C[8 kHz resampling]
    C --> D[300–3400 Hz band limit]
    D --> E[G.711 μ-law]
    E --> F[Noise + packet loss]
    F --> G[Degraded WAV]

    G --> H[DSP enhancement]
    H --> I[Enhanced WAV]

    A --> J[Faster-Whisper]
    G --> J
    I --> J

    J --> K[WER / latency / RTF]
    K --> L[CSV + JSON + chart]

    M[FastAPI] --> B
    M --> H
    M --> J