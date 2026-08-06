"""FastAPI application exposing the classifier to the desktop UI.

Endpoints (all over local HTTP):
    GET  /health              service + model status
    GET  /subgenres           the eight target subgenres
    POST /classify            classify a single audio file (path)
    POST /jobs                start a batch job over a directory
    GET  /jobs/{job_id}       poll a batch job's progress + results
    DELETE /jobs/{job_id}     request cancellation of a running job
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from edm_classifier import SUBGENRES
from edm_classifier.api.jobs import JobManager
from edm_classifier.api.model_service import ModelService
from edm_classifier.api.schemas import (
    ClassifyRequest,
    HealthResponse,
    JobCreatedResponse,
    JobRequest,
    JobResponse,
    PredictionResponse,
    SubgenresResponse,
    Top2Item,
)
from edm_classifier.config import SUPPORTED_EXTENSIONS


class _PredictorAdapter:
    """Fetches the (lazily loaded) predictor for each call, for the JobManager."""

    def __init__(self, model_service: ModelService) -> None:
        self._model_service = model_service

    def predict_track(self, path):
        return self._model_service.predictor().predict_track(path)


def create_app(model_service: ModelService | None = None) -> FastAPI:
    """Build the FastAPI app. A ModelService can be injected for testing."""
    app = FastAPI(title="EDM Classifier API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local desktop tool: renderer talks to localhost only
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    service = model_service or ModelService()
    app.state.model_service = service
    app.state.job_manager = JobManager(_PredictorAdapter(service))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_loaded=service.is_loaded,
            device=service.device,
        )

    @app.get("/subgenres", response_model=SubgenresResponse)
    def subgenres() -> SubgenresResponse:
        return SubgenresResponse(subgenres=list(SUBGENRES))

    @app.post("/classify", response_model=PredictionResponse)
    def classify(req: ClassifyRequest) -> PredictionResponse:
        path = Path(req.path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(422, f"Unsupported format: {path.suffix!r}")
        if not path.is_file():
            raise HTTPException(404, f"File not found: {req.path}")
        try:
            pred = service.predictor().predict_track(path)
        except RuntimeError as exc:  # model not configured
            raise HTTPException(503, str(exc)) from exc
        return PredictionResponse(
            path=str(path),
            subgenre=pred.subgenre,
            confidence=pred.confidence,
            top2=[Top2Item(subgenre=s, probability=p) for s, p in pred.top2],
            probabilities=pred.probabilities,
        )

    @app.post("/jobs", response_model=JobCreatedResponse)
    def create_job(req: JobRequest) -> JobCreatedResponse:
        try:
            job = app.state.job_manager.create(
                req.directory, req.mode, req.recursive, req.confidence_threshold
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return JobCreatedResponse(job_id=job.job_id, status=job.status, total=job.total)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, f"Job not found: {job_id}")
        return job.snapshot()

    @app.delete("/jobs/{job_id}", response_model=JobResponse)
    def cancel_job(job_id: str) -> JobResponse:
        job = app.state.job_manager.get(job_id)
        if job is None:
            raise HTTPException(404, f"Job not found: {job_id}")
        app.state.job_manager.cancel(job_id)
        return job.snapshot()

    return app
