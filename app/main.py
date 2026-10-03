"""CallClear FastAPI application."""

import asyncio
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.asr.engine import WhisperEngine
from app.benchmarks.runner import run_local_benchmark
from app.core.config import get_settings
from app.db.mongo import MongoRepository
from app.dsp.enhancement import enhance_audio
from app.dsp.telephony import (
    DegradationConfig,
    degrade_audio,
)


settings = get_settings()

app = FastAPI(
    title="CallClear",
    description=(
        "Benchmarking and improving ASR robustness "
        "on Indian telephony audio."
    ),
    version="0.1.0",
)


def _ensure_dirs() -> None:
    """Create application data directories."""
    for path in (
        settings.raw_dir,
        settings.degraded_dir,
        settings.enhanced_dir,
        Path("results"),
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


@app.on_event("startup")
async def startup() -> None:
    """Prepare local directories."""
    _ensure_dirs()


def _save_upload(
    upload: UploadFile,
    directory: Path,
) -> Path:
    """Save an uploaded file and return its path."""
    suffix = (
        Path(
            upload.filename or "audio.wav"
        ).suffix.lower()
        or ".wav"
    )

    path = (
        directory
        / f"{uuid.uuid4().hex}{suffix}"
    )

    with path.open("wb") as handle:
        while chunk := upload.file.read(
            1024 * 1024
        ):
            handle.write(chunk)

    return path


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "CallClear API is running",
        "version": "0.1.0",
    }


@app.get("/health")
async def health() -> dict[str, object]:
    """Return service and MongoDB health information."""
    mongo_ok = False

    repository = MongoRepository(
        settings.mongodb_uri,
        settings.mongodb_database,
    )

    try:
        await repository.connect()
        mongo_ok = True
    except Exception:
        mongo_ok = False
    finally:
        await repository.close()

    return {
        "status": "ok",
        "service": "callclear",
        "mongodb": mongo_ok,
    }


@app.post("/degrade")
def degrade(
    file: UploadFile = File(...),
    snr_db: float = Form(10.0),
    packet_loss_probability: float = Form(0.02),
    packet_loss_ms: float = Form(60.0),
) -> FileResponse:
    """Apply telephony degradation to an uploaded WAV."""
    _ensure_dirs()

    source = _save_upload(
        file,
        settings.raw_dir,
    )

    try:
        audio, sample_rate = sf.read(
            source,
            always_2d=False,
        )

        if getattr(audio, "ndim", 1) != 1:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Only mono WAV files are supported."
                ),
            )

        degraded, output_rate = degrade_audio(
            audio,
            sample_rate,
            DegradationConfig(
                target_sample_rate=(
                    settings.telephony_sample_rate
                ),
                snr_db=snr_db,
                packet_loss_probability=(
                    packet_loss_probability
                ),
                packet_loss_ms=packet_loss_ms,
            ),
        )

        output = (
            settings.degraded_dir
            / f"{source.stem}_degraded.wav"
        )

        sf.write(
            output,
            degraded,
            output_rate,
            subtype="PCM_16",
        )

        return FileResponse(
            output,
            media_type="audio/wav",
            filename=output.name,
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/enhance")
def enhance(
    file: UploadFile = File(...),
    method: str = Form("wiener"),
) -> FileResponse:
    """Enhance an uploaded WAV."""
    _ensure_dirs()

    source = _save_upload(
        file,
        settings.degraded_dir,
    )

    try:
        audio, sample_rate = sf.read(
            source,
            always_2d=False,
        )

        enhanced = enhance_audio(
            audio,
            sample_rate,
            method=method,
        )

        output = (
            settings.enhanced_dir
            / f"{source.stem}_enhanced.wav"
        )

        sf.write(
            output,
            enhanced,
            sample_rate,
            subtype="PCM_16",
        )

        return FileResponse(
            output,
            media_type="audio/wav",
            filename=output.name,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@app.post("/transcribe")
def transcribe(
    file: UploadFile = File(...),
    model: str | None = Form(None),
    language: str | None = Form(None),
) -> dict[str, object]:
    """Transcribe an uploaded WAV."""
    _ensure_dirs()

    source = _save_upload(
        file,
        settings.raw_dir,
    )

    engine = WhisperEngine(
        model_name=(
            model or settings.asr_model
        ),
        device=settings.asr_device,
        compute_type=settings.asr_compute_type,
    )

    try:
        result = engine.transcribe(
            source,
            language=(
                language
                or settings.asr_language
            ),
            beam_size=settings.asr_beam_size,
        )

        return result.__dict__

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


async def _benchmark_task(
    limit: int,
    model: str,
) -> None:
    """Run the benchmark in a worker thread."""
    await asyncio.to_thread(
        run_local_benchmark,
        Path("results"),
        limit,
        model,
    )


@app.post("/benchmarks/run")
async def benchmark_run(
    background_tasks: BackgroundTasks,
    limit: int = 8,
    model: str = "base",
) -> dict[str, object]:
    """Start a local benchmark in the background."""
    if not 1 <= limit <= 8:
        raise HTTPException(
            status_code=400,
            detail=(
                "limit must be between 1 and 8"
            ),
        )

    background_tasks.add_task(
        _benchmark_task,
        limit,
        model,
    )

    return {
        "status": "started",
        "limit": limit,
        "model": model,
        "benchmark": "local",
    }


@app.get("/benchmarks")
async def benchmarks() -> dict[str, object]:
    """Return recent benchmark runs and result rows."""
    repository = MongoRepository(
        settings.mongodb_uri,
        settings.mongodb_database,
    )

    try:
        await repository.connect()

        runs = await repository.list_runs(
            limit=20
        )

        rows = await repository.list_benchmarks(
            limit=100
        )

        return {
            "mongodb": "available",
            "runs_count": len(runs),
            "runs": runs,
            "results_count": len(rows),
            "results": rows,
        }

    except Exception as exc:
        return {
            "mongodb": "unavailable",
            "runs_count": 0,
            "runs": [],
            "results_count": 0,
            "results": [],
            "detail": str(exc),
        }

    finally:
        await repository.close()