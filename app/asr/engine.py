"""Faster-Whisper wrapper with lazy model loading."""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


@dataclass(frozen=True)
class TranscriptionResult:
    """ASR result and performance metrics."""

    text: str
    model: str
    latency_seconds: float
    audio_duration_seconds: float
    real_time_factor: float


class WhisperEngine:
    """Lazy CPU Faster-Whisper engine."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "Faster-Whisper is not installed. Run "
                    "uv sync --extra asr."
                ) from exc
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        language: str = "hi",
        beam_size: int = 5,
    ) -> TranscriptionResult:
        """Transcribe a local audio file and return timing metrics."""
        import soundfile as sf

        model = self._get_model()
        info = sf.info(audio_path)
        duration = float(info.frames / info.samplerate)
        start = perf_counter()
        segments, _ = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        latency = perf_counter() - start
        rtf = latency / duration if duration > 0 else 0.0
        return TranscriptionResult(text, self.model_name, latency, duration, rtf)
