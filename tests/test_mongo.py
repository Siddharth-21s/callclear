"""Offline MongoDB repository tests using mocked async collections."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from app.db.mongo import MongoRepository


class _FakeInsertResult:
    """Minimal replacement for InsertOneResult."""

    inserted_id = "fake-id"


class _FakeCursor:
    """Minimal async Mongo cursor supporting sort/limit/to_list."""

    def __init__(self, documents):
        self.documents = documents

    def sort(self, field: str, direction: int):
        reverse = direction < 0
        self.documents = sorted(
            self.documents,
            key=lambda document: document.get(field),
            reverse=reverse,
        )
        return self

    def limit(self, amount: int):
        self.documents = self.documents[:amount]
        return self

    async def to_list(self, length: int):
        return self.documents[:length]


class _FakeCollection:
    """Minimal async collection used by repository tests."""

    def __init__(self):
        self.documents = []
        self.indexes = []

    async def create_index(
        self,
        field,
        unique=False,
    ):
        self.indexes.append(
            {
                "field": field,
                "unique": unique,
            }
        )
        return f"{field}_index"

    async def insert_one(self, document):
        self.documents.append(dict(document))
        return _FakeInsertResult()

    def find(self, filter_document, projection):
        del filter_document
        del projection
        return _FakeCursor(list(self.documents))


class _FakeAdmin:
    async def command(self, command):
        assert command == "ping"
        return {"ok": 1}


class _FakeClient:
    """Minimal AsyncIOMotorClient replacement."""

    def __init__(
        self,
        uri,
        serverSelectionTimeoutMS,
    ):
        self.uri = uri
        self.timeout = serverSelectionTimeoutMS
        self.admin = _FakeAdmin()
        self.database = SimpleNamespace(
            runs=_FakeCollection(),
            audio_samples=_FakeCollection(),
            benchmark_results=_FakeCollection(),
        )
        self.closed = False

    def __getitem__(self, name):
        assert name == "callclear"
        return self.database

    def close(self):
        self.closed = True


def test_repository_connects_and_creates_indexes(
    monkeypatch,
) -> None:
    """Repository should connect and prepare required indexes."""

    async def run() -> None:
        import motor.motor_asyncio

        fake_client = _FakeClient(
            "mongodb://example",
            1500,
        )

        monkeypatch.setattr(
            motor.motor_asyncio,
            "AsyncIOMotorClient",
            lambda uri, serverSelectionTimeoutMS: fake_client,
        )

        repository = MongoRepository(
            "mongodb://example",
            "callclear",
        )

        await repository.connect()

        assert repository.db is fake_client.database

        assert {
            "field": "run_id",
            "unique": True,
        } in fake_client.database.runs.indexes

        assert {
            "field": "sample_id",
            "unique": True,
        } in fake_client.database.audio_samples.indexes

        await repository.close()

        assert fake_client.closed is True
        assert repository.client is None
        assert repository.db is None

    asyncio.run(run())


def test_repository_inserts_and_lists_benchmark_data() -> None:
    """Repository should insert and retrieve run/result documents."""

    async def run() -> None:
        repository = MongoRepository(
            "mongodb://example",
            "callclear",
        )

        fake_client = _FakeClient(
            "mongodb://example",
            1500,
        )

        repository.client = fake_client
        repository.db = fake_client.database

        await repository.insert_run(
            {
                "run_id": "run-1",
                "created_at": "2026-10-03T00:00:00Z",
            }
        )

        await repository.insert_benchmark_result(
            {
                "run_id": "run-1",
                "sample_id": 1,
                "wer": 0.25,
                "created_at": "2026-10-03T00:00:01Z",
            }
        )

        runs = await repository.list_runs(limit=20)
        results = await repository.list_benchmarks(limit=20)

        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-1"

        assert len(results) == 1
        assert results[0]["sample_id"] == 1
        assert results[0]["wer"] == 0.25

    asyncio.run(run())


def test_repository_requires_connection() -> None:
    """Repository operations should fail clearly when not connected."""

    async def run() -> None:
        repository = MongoRepository(
            "mongodb://example",
            "callclear",
        )

        with pytest.raises(
            RuntimeError,
            match="MongoDB is not connected",
        ):
            await repository.insert_run(
                {
                    "run_id": "run-1",
                }
            )

    asyncio.run(run())