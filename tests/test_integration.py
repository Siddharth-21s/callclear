"""Opt-in integration tests requiring real services and model downloads."""

import os

import pytest


@pytest.mark.integration
def test_real_mongodb_and_whisper() -> None:
    """Ping MongoDB and instantiate Faster-Whisper when explicitly enabled."""
    if os.environ.get("CALLCLEAR_RUN_INTEGRATION") != "1":
        pytest.skip("Set CALLCLEAR_RUN_INTEGRATION=1 to run real integration checks.")
    motor = pytest.importorskip("motor")
    del motor
    faster_whisper = pytest.importorskip("faster_whisper")
    from motor.motor_asyncio import AsyncIOMotorClient

    import asyncio

    async def ping() -> None:
        client = AsyncIOMotorClient("mongodb://localhost:27017", serverSelectionTimeoutMS=1500)
        try:
            await client.admin.command("ping")
        finally:
            client.close()

    asyncio.run(ping())
    faster_whisper.WhisperModel("base", device="cpu", compute_type="int8")
