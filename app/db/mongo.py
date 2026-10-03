"""Async MongoDB repository using Motor."""

from collections.abc import Mapping
from typing import Any


class MongoRepository:
    """Small repository wrapper for benchmark collections."""

    def __init__(self, uri: str, database: str) -> None:
        self.uri = uri
        self.database_name = database
        self.client = None
        self.db = None

    async def connect(self) -> None:
        """Connect and create useful indexes."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError as exc:
            raise RuntimeError("Motor is not installed.") from exc
        self.client = AsyncIOMotorClient(self.uri, serverSelectionTimeoutMS=1500)
        await self.client.admin.command("ping")
        self.db = self.client[self.database_name]
        await self.db.runs.create_index("created_at")
        await self.db.audio_samples.create_index("sample_id", unique=True)
        await self.db.benchmark_results.create_index("run_id")

    async def close(self) -> None:
        """Close the MongoDB client."""
        if self.client is not None:
            self.client.close()

    async def insert_run(self, document: Mapping[str, Any]) -> Any:
        """Insert a benchmark run document."""
        if self.db is None:
            raise RuntimeError("MongoDB is not connected.")
        return await self.db.runs.insert_one(dict(document))

    async def insert_audio_sample(self, document: Mapping[str, Any]) -> Any:
        """Insert an audio sample document."""
        if self.db is None:
            raise RuntimeError("MongoDB is not connected.")
        return await self.db.audio_samples.insert_one(dict(document))

    async def insert_benchmark_result(self, document: Mapping[str, Any]) -> Any:
        """Insert a benchmark result document."""
        if self.db is None:
            raise RuntimeError("MongoDB is not connected.")
        return await self.db.benchmark_results.insert_one(dict(document))

    async def list_benchmarks(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent benchmark results."""
        if self.db is None:
            raise RuntimeError("MongoDB is not connected.")
        cursor = self.db.benchmark_results.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
