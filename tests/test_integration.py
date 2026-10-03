"""Opt-in integration tests requiring real services and model downloads."""

import asyncio
import os

import pytest

from app.core.config import get_settings


@pytest.mark.integration
def test_real_mongodb_and_whisper() -> None:
    """Ping configured MongoDB and instantiate configured Faster-Whisper."""
    if os.environ.get("CALLCLEAR_RUN_INTEGRATION") != "1":
        pytest.skip(
            "Set CALLCLEAR_RUN_INTEGRATION=1 to run real integration checks."
        )

    motor = pytest.importorskip("motor")
    del motor

    faster_whisper = pytest.importorskip("faster_whisper")
    from motor.motor_asyncio import AsyncIOMotorClient

    settings = get_settings()

    async def ping() -> None:
        client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=1500,
        )

        try:
            await client.admin.command("ping")
        finally:
            client.close()

    asyncio.run(ping())

    faster_whisper.WhisperModel(
        settings.asr_model,
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
    )