"""Pydantic request/response models for the classifier API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None


class SubgenresResponse(BaseModel):
    subgenres: list[str]


class ClassifyRequest(BaseModel):
    """Classify a single audio file already present on the server's filesystem."""

    path: str = Field(..., description="Absolute path to an MP3/AIFF/WAV file.")


class Top2Item(BaseModel):
    subgenre: str
    probability: float


class PredictionResponse(BaseModel):
    path: str
    subgenre: str
    confidence: float
    top2: list[Top2Item]
    probabilities: dict[str, float]


class JobMode(StrEnum):
    """What to do with each classified file."""

    classify = "classify"  # predict only, don't touch files
    move = "move"  # organize by moving files into per-subgenre subfolders
    copy = "copy"  # organize by copying files into per-subgenre subfolders


class JobRequest(BaseModel):
    """Start a batch classification job over a directory of audio files."""

    directory: str = Field(..., description="Directory containing audio files to classify.")
    mode: JobMode = JobMode.move
    recursive: bool = Field(False, description="Recurse into subdirectories.")
    confidence_threshold: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Tracks below this confidence go to a 'Revisar' folder (0 = disabled).",
    )


class JobStatus(StrEnum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class FileResult(BaseModel):
    path: str
    subgenre: str
    confidence: float
    organized_path: str | None = None  # where the file was moved/copied, if any
    review: bool = False  # True when sent to the 'Revisar' folder (low confidence)
    second_choice: Top2Item | None = None  # surfaced for review tracks


class JobResponse(BaseModel):
    """Full state of a batch job (progress + results)."""

    job_id: str
    status: JobStatus
    mode: JobMode
    directory: str
    total: int
    processed: int
    current_file: str | None = None
    subgenre_counts: dict[str, int] = {}
    review_count: int = 0
    average_confidence: float | None = None
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    results: list[FileResult] = []
    error: str | None = None


class JobCreatedResponse(BaseModel):
    job_id: str
    status: JobStatus
    total: int
