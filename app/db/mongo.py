"""Async MongoDB repository using Motor."""

from collections.abc import Mapping
from typing import Any


class MongoRepository:
    """Small repository wrapper for CallClear benchmark collections."""

    def __init__(
        self,
        uri: str,
        database: str,
    ) -> None:
        self.uri = uri
        self.database_name = database
        self.client = None
        self.db = None

    async def connect(self) -> None:
        """Connect to MongoDB and create useful indexes."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
        except ImportError as exc:
            raise RuntimeError(
                "Motor is not installed. "
                "Run: uv sync --extra database"
            ) from exc

        self.client = AsyncIOMotorClient(
            self.uri,
            serverSelectionTimeoutMS=1500,
        )

        await self.client.admin.command("ping")

        self.db = self.client[self.database_name]

        await self.db.runs.create_index(
            "created_at"
        )

        await self.db.runs.create_index(
            "run_id",
            unique=True,
        )

        await self.db.audio_samples.create_index(
            "sample_id",
            unique=True,
        )

        await self.db.benchmark_results.create_index(
            "run_id"
        )

        await self.db.benchmark_results.create_index(
            "created_at"
        )

    async def close(self) -> None:
        """Close the MongoDB client."""
        if self.client is not None:
            self.client.close()
            self.client = None
            self.db = None

    def _require_db(self) -> Any:
        """Return the connected database or raise a clear error."""
        if self.db is None:
            raise RuntimeError(
                "MongoDB is not connected."
            )

        return self.db

    async def insert_run(
        self,
        document: Mapping[str, Any],
    ) -> Any:
        """Insert a benchmark run document."""
        db = self._require_db()

        return await db.runs.insert_one(
            dict(document)
        )

    async def insert_audio_sample(
        self,
        document: Mapping[str, Any],
    ) -> Any:
        """Insert an audio sample document."""
        db = self._require_db()

        return await db.audio_samples.insert_one(
            dict(document)
        )

    async def insert_benchmark_result(
        self,
        document: Mapping[str, Any],
    ) -> Any:
        """Insert one benchmark result document."""
        db = self._require_db()

        return await db.benchmark_results.insert_one(
            dict(document)
        )

    async def list_benchmarks(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return recent benchmark result rows."""
        if limit < 1:
            return []

        db = self._require_db()

        cursor = (
            db.benchmark_results
            .find(
                {},
                {"_id": 0},
            )
            .sort(
                "created_at",
                -1,
            )
            .limit(limit)
        )

        return await cursor.to_list(
            length=limit
        )

    async def list_runs(
        self,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Return recent benchmark run records."""
        if limit < 1:
            return []

        db = self._require_db()

        cursor = (
            db.runs
            .find(
                {},
                {"_id": 0},
            )
            .sort(
                "created_at",
                -1,
            )
            .limit(limit)
        )

        return await cursor.to_list(
            length=limit
        )