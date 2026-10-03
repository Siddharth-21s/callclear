"""Pydantic request and response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class DegradeOptions(BaseModel):
    """Options for the telephony simulator."""

    snr_db: float = Field(10.0, ge=-20.0, le=60.0)
    packet_loss_probability: float = Field(0.02, ge=0.0, le=1.0)
    packet_loss_ms: float = Field(60.0, gt=0.0, le=1000.0)


class EnhanceOptions(BaseModel):
    """Options for DSP enhancement."""

    method: Literal["wiener", "spectral_gate", "vad", "highpass"] = "wiener"


class TranscriptionResponse(BaseModel):
    """ASR response."""

    text: str
    model: str
    latency_seconds: float
    audio_duration_seconds: float
    real_time_factor: float
