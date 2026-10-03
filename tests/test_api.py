"""Offline FastAPI endpoint tests."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

import app.main as main_module
from app.asr.engine import TranscriptionResult


def _wav_bytes(
    frequency_hz: float = 440.0,
    duration_seconds: float = 1.0,
    sample_rate: int = 8000,
) -> bytes:
    """Create a small synthetic mono WAV in memory."""
    samples = np.arange(
        int(sample_rate * duration_seconds),
        dtype=np.float32,
    )

    audio = (
        0.25
        * np.sin(
            2 * np.pi * frequency_hz * samples / sample_rate
        )
    ).astype(np.float32)

    buffer = io.BytesIO()

    sf.write(
        buffer,
        audio,
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )

    return buffer.getvalue()


def test_root() -> None:
    """Test the root endpoint."""
    with TestClient(main_module.app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"


def test_health() -> None:
    """Test the health endpoint."""
    with TestClient(main_module.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mongodb"] is False


def test_degrade_endpoint_returns_wav() -> None:
    """Test synthetic telephony degradation through the API."""
    wav = _wav_bytes()

    with TestClient(main_module.app) as client:
        response = client.post(
            "/degrade",
            files={
                "file": (
                    "input.wav",
                    wav,
                    "audio/wav",
                )
            },
            data={
                "snr_db": "10",
                "packet_loss_probability": "0",
                "packet_loss_ms": "60",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "audio/wav"
    )

    audio, sample_rate = sf.read(
        io.BytesIO(response.content),
        always_2d=False,
    )

    assert audio.ndim == 1
    assert sample_rate == 8000
    assert audio.size == 8000


def test_enhance_endpoint_returns_wav() -> None:
    """Test DSP enhancement through the API."""
    wav = _wav_bytes()

    with TestClient(main_module.app) as client:
        response = client.post(
            "/enhance",
            files={
                "file": (
                    "input.wav",
                    wav,
                    "audio/wav",
                )
            },
            data={
                "method": "highpass",
            },
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith(
        "audio/wav"
    )

    audio, sample_rate = sf.read(
        io.BytesIO(response.content),
        always_2d=False,
    )

    assert audio.ndim == 1
    assert sample_rate == 8000
    assert audio.size == 8000


@dataclass
class _FakeWhisperEngine:
    """Minimal offline replacement for WhisperEngine."""

    model_name: str
    device: str
    compute_type: str

    def transcribe(
        self,
        audio_path,
        language: str,
        beam_size: int,
    ) -> TranscriptionResult:
        """Return deterministic fake ASR output."""
        del audio_path
        del beam_size
        del language

        return TranscriptionResult(
            text="यह एक परीक्षण है",
            model=self.model_name,
            latency_seconds=0.01,
            audio_duration_seconds=1.0,
            real_time_factor=0.01,
        )


def test_transcribe_endpoint_uses_mocked_asr(
    monkeypatch,
) -> None:
    """Test /transcribe without downloading or loading Whisper."""
    monkeypatch.setattr(
        main_module,
        "WhisperEngine",
        _FakeWhisperEngine,
    )

    wav = _wav_bytes()

    with TestClient(main_module.app) as client:
        response = client.post(
            "/transcribe",
            files={
                "file": (
                    "input.wav",
                    wav,
                    "audio/wav",
                )
            },
            data={
                "model": "base",
                "language": "hi",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["text"] == "यह एक परीक्षण है"
    assert body["model"] == "base"
    assert body["latency_seconds"] == 0.01
    assert body["audio_duration_seconds"] == 1.0
    assert body["real_time_factor"] == 0.01


async def _fake_benchmark_task(
    limit: int,
    model: str,
) -> None:
    """Offline replacement for the background benchmark."""
    assert limit == 2
    assert model == "base"


def test_benchmark_run_starts_background_task(
    monkeypatch,
) -> None:
    """Test benchmark job submission without running Whisper."""
    monkeypatch.setattr(
        main_module,
        "_benchmark_task",
        _fake_benchmark_task,
    )

    with TestClient(main_module.app) as client:
        response = client.post(
            "/benchmarks/run",
            params={
                "limit": 2,
                "model": "base",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "started"
    assert body["limit"] == 2
    assert body["model"] == "base"
    assert body["benchmark"] == "local"


def test_benchmark_run_rejects_limit_above_api_limit() -> None:
    """The API currently restricts background jobs to at most 8 samples."""
    with TestClient(main_module.app) as client:
        response = client.post(
            "/benchmarks/run",
            params={
                "limit": 9,
                "model": "base",
            },
        )

    assert response.status_code == 400
    assert "between 1 and 8" in response.json()["detail"]