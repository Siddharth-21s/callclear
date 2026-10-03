"""Verify the local CallClear development environment."""

from __future__ import annotations

import ctypes.util
import os
import platform
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check_python() -> bool:
    """Check that Python 3.12 is being used."""
    version = sys.version_info
    passed = version.major == 3 and version.minor == 12
    status = "PASS" if passed else "FAIL"
    print(f"{status}  Python: {platform.python_version()}")
    return passed


def check_architecture() -> bool:
    """Check that the native Apple Silicon architecture is active."""
    architecture = platform.machine()
    passed = architecture == "arm64"
    status = "PASS" if passed else "FAIL"
    print(f"{status}  Architecture: {architecture}")
    return passed


def check_ffmpeg() -> bool:
    """Check that FFmpeg is available on PATH."""
    path = shutil.which("ffmpeg")
    passed = path is not None
    status = "PASS" if passed else "FAIL"
    print(f"{status}  ffmpeg: {path or 'not found'}")
    return passed


def check_libsndfile() -> bool:
    """Check that libsndfile is available."""
    library = ctypes.util.find_library("sndfile")

    homebrew_candidates = [
        Path("/opt/homebrew/opt/libsndfile/lib/libsndfile.dylib"),
        Path("/opt/homebrew/opt/libsndfile/lib/libsndfile.1.dylib"),
    ]

    if library:
        print(f"PASS  libsndfile: {library}")
        return True

    for candidate in homebrew_candidates:
        if candidate.exists():
            print(f"PASS  libsndfile: {candidate}")
            return True

    print("FAIL  libsndfile: library not found")
    return False


def check_python_package(package_name: str, display_name: str) -> bool:
    """Check whether a Python package can be imported."""
    try:
        __import__(package_name)
    except ImportError:
        print(f"FAIL  {display_name}: not installed")
        return False

    print(f"PASS  {display_name}: installed")
    return True


def check_mongodb(strict: bool) -> bool:
    """Check MongoDB connectivity when requested."""
    try:
        from pymongo import MongoClient
        from pymongo.errors import PyMongoError

        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        client = MongoClient(uri, serverSelectionTimeoutMS=1000)
        client.admin.command("ping")
        print("PASS  MongoDB: reachable")
        return True
    except Exception:
        if strict:
            print("FAIL  MongoDB: not reachable")
            return False

        print("INFO  MongoDB: not reachable (optional unless --strict)")
        return True


def check_whisper_model() -> bool:
    """Optionally load the configured Faster-Whisper model."""
    if os.getenv("CALLCLEAR_CHECK_MODEL") != "1":
        print(
            "INFO  Whisper model: skipped; "
            "set CALLCLEAR_CHECK_MODEL=1 to load/download it"
        )
        return True

    try:
        from faster_whisper import WhisperModel

        model_name = os.getenv("ASR_MODEL", "base")
        device = os.getenv("ASR_DEVICE", "cpu")
        compute_type = os.getenv("ASR_COMPUTE_TYPE", "int8")

        print(
            f"INFO  Whisper model: loading {model_name} "
            f"({device}, {compute_type})"
        )

        WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        print("PASS  Whisper model: loaded")
        return True
    except Exception as exc:
        print(f"FAIL  Whisper model: {exc}")
        return False


def main() -> int:
    """Run all environment checks."""
    strict = "--strict" in sys.argv

    checks = [
        check_python(),
        check_architecture(),
        check_ffmpeg(),
        check_libsndfile(),
        check_python_package("faster_whisper", "Faster-Whisper package"),
        check_python_package("silero_vad", "Silero VAD package"),
        check_mongodb(strict),
        check_whisper_model(),
    ]

    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())