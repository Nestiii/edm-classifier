"""Background batch-classification jobs with progress tracking.

A job scans a directory for audio files, classifies each track, optionally
organizes it into a per-subgenre folder, and exposes live progress (processed
count, current file, per-subgenre counts, average confidence, elapsed and ETA).
Jobs run on a worker thread so the API stays responsive (Req 2.2/2.5/2.6).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from edm_classifier.api.organizer import organize_file, review_file
from edm_classifier.api.schemas import (
    FileResult,
    JobMode,
    JobResponse,
    JobStatus,
    Top2Item,
)
from edm_classifier.config import SUBGENRES, SUPPORTED_EXTENSIONS


class PredictorLike(Protocol):
    """Minimal predictor surface the job manager depends on."""

    def predict_track(self, path: str | Path): ...  # returns an object with .subgenre/.confidence


def list_audio_files(directory: Path, recursive: bool) -> list[Path]:
    """Return supported audio files in ``directory`` (sorted, deterministic)."""
    globber = directory.rglob("*") if recursive else directory.glob("*")
    files = [
        p for p in globber if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


@dataclass
class Job:
    """Mutable state of a single batch job, guarded by ``lock``."""

    job_id: str
    directory: str
    mode: JobMode
    recursive: bool
    total: int
    confidence_threshold: float = 0.0
    status: JobStatus = JobStatus.pending
    processed: int = 0
    current_file: str | None = None
    subgenre_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(SUBGENRES, 0))
    review_count: int = 0
    confidence_sum: float = 0.0
    results: list[FileResult] = field(default_factory=list)
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_event: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> JobResponse:
        with self.lock:
            elapsed = None
            if self.started_at is not None:
                end = self.finished_at if self.finished_at is not None else time.monotonic()
                elapsed = end - self.started_at
            eta = None
            if elapsed and self.processed and self.status == JobStatus.running:
                eta = elapsed / self.processed * (self.total - self.processed)
            avg_conf = self.confidence_sum / self.processed if self.processed else None
            return JobResponse(
                job_id=self.job_id,
                status=self.status,
                mode=self.mode,
                directory=self.directory,
                total=self.total,
                processed=self.processed,
                current_file=self.current_file,
                subgenre_counts=dict(self.subgenre_counts),
                review_count=self.review_count,
                average_confidence=avg_conf,
                elapsed_seconds=elapsed,
                eta_seconds=eta,
                results=list(self.results),
                error=self.error,
            )


class JobManager:
    """Creates and runs batch jobs, keyed by job id."""

    def __init__(self, predictor: PredictorLike) -> None:
        self.predictor = predictor
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(
        self,
        directory: str,
        mode: JobMode,
        recursive: bool,
        confidence_threshold: float = 0.0,
    ) -> Job:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
        files = list_audio_files(root, recursive)
        job = Job(
            job_id=uuid.uuid4().hex,
            directory=str(root),
            mode=mode,
            recursive=recursive,
            total=len(files),
            confidence_threshold=confidence_threshold,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run, args=(job, files), daemon=True)
        thread.start()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None or job.status in (JobStatus.completed, JobStatus.failed):
            return False
        job.cancel_event.set()
        return True

    def _run(self, job: Job, files: list[Path]) -> None:
        with job.lock:
            job.status = JobStatus.running
            job.started_at = time.monotonic()
        try:
            for path in files:
                if job.cancel_event.is_set():
                    with job.lock:
                        job.status = JobStatus.cancelled
                        job.finished_at = time.monotonic()
                        job.current_file = None
                    return
                with job.lock:
                    job.current_file = str(path)

                prediction = self.predictor.predict_track(path)
                is_review = float(prediction.confidence) < job.confidence_threshold

                organized_path: str | None = None
                if job.mode in (JobMode.move, JobMode.copy):
                    move = job.mode == JobMode.move
                    if is_review:
                        dest = review_file(path, job.directory, move=move)
                    else:
                        dest = organize_file(path, job.directory, prediction.subgenre, move=move)
                    organized_path = str(dest)

                # Second choice from top-2, surfaced for review tracks.
                second = None
                if len(prediction.top2) > 1:
                    s_name, s_prob = prediction.top2[1]
                    second = Top2Item(subgenre=s_name, probability=float(s_prob))

                with job.lock:
                    job.processed += 1
                    job.confidence_sum += float(prediction.confidence)
                    if is_review:
                        job.review_count += 1
                    else:
                        job.subgenre_counts[prediction.subgenre] = (
                            job.subgenre_counts.get(prediction.subgenre, 0) + 1
                        )
                    job.results.append(
                        FileResult(
                            path=str(path),
                            subgenre=prediction.subgenre,
                            confidence=float(prediction.confidence),
                            organized_path=organized_path,
                            review=is_review,
                            second_choice=second if is_review else None,
                        )
                    )
            with job.lock:
                job.status = JobStatus.completed
                job.finished_at = time.monotonic()
                job.current_file = None
        except Exception as exc:  # noqa: BLE001 - surface any failure to the client
            with job.lock:
                job.status = JobStatus.failed
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = time.monotonic()
                job.current_file = None
