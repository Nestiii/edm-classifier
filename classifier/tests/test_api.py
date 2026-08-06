"""Tests for the FastAPI inference API (with a fake predictor)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edm_classifier.api.app import create_app
from edm_classifier.api.model_service import ModelService
from edm_classifier.config import SUBGENRES
from edm_classifier.inference.predictor import Prediction


class FakePredictor:
    """Deterministic predictor: subgenre chosen from the filename hash."""

    def predict_track(self, path) -> Prediction:
        idx = sum(ord(c) for c in Path(path).stem) % len(SUBGENRES)
        subgenre = SUBGENRES[idx]
        probs = {name: 0.02 for name in SUBGENRES}
        probs[subgenre] = 0.86
        second = SUBGENRES[(idx + 1) % len(SUBGENRES)]
        probs[second] = 0.12
        return Prediction(
            subgenre=subgenre,
            confidence=probs[subgenre],
            top2=[(subgenre, probs[subgenre]), (second, probs[second])],
            probabilities=probs,
        )


@pytest.fixture
def client() -> TestClient:
    service = ModelService()
    service.set_predictor(FakePredictor(), device="fake")
    return TestClient(create_app(service))


@pytest.fixture
def unloaded_client() -> TestClient:
    return TestClient(create_app(ModelService()))


def _wait_for_job(client: TestClient, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = client.get(f"/jobs/{job_id}").json()
        if data["status"] in ("completed", "failed", "cancelled"):
            return data
        time.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not finish in time")


def test_health_loaded(client: TestClient):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["device"] == "fake"


def test_health_unloaded(unloaded_client: TestClient):
    data = unloaded_client.get("/health").json()
    assert data["model_loaded"] is False


def test_subgenres(client: TestClient):
    data = client.get("/subgenres").json()
    assert data["subgenres"] == list(SUBGENRES)


def test_classify_single_file(client: TestClient, wav_file: Path):
    resp = client.post("/classify", json={"path": str(wav_file)})
    assert resp.status_code == 200
    body = resp.json()
    assert body["subgenre"] in SUBGENRES
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top2"]) == 2
    assert len(body["probabilities"]) == len(SUBGENRES)


def test_classify_missing_file(client: TestClient, tmp_path: Path):
    resp = client.post("/classify", json={"path": str(tmp_path / "nope.wav")})
    assert resp.status_code == 404


def test_classify_unsupported_format(client: TestClient, tmp_path: Path):
    bad = tmp_path / "song.flac"
    bad.write_text("x")
    resp = client.post("/classify", json={"path": str(bad)})
    assert resp.status_code == 422


def test_classify_without_model_returns_503(unloaded_client: TestClient, wav_file: Path):
    resp = unloaded_client.post("/classify", json={"path": str(wav_file)})
    assert resp.status_code == 503


def test_batch_job_move_organizes_files(client: TestClient, dataset_dir: Path):
    # Flatten a few tracks into one folder to classify + organize.
    import soundfile as sf

    from edm_classifier.config import settings

    work = dataset_dir.parent / "inbox"
    work.mkdir()
    sr = settings.audio.sample_rate
    for i in range(3):
        sf.write(work / f"track_{i}.wav", [0.0] * sr, sr, subtype="PCM_16")

    resp = client.post("/jobs", json={"directory": str(work), "mode": "move"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert resp.json()["total"] == 3

    data = _wait_for_job(client, job_id)
    assert data["status"] == "completed"
    assert data["processed"] == 3
    assert sum(data["subgenre_counts"].values()) == 3
    assert data["average_confidence"] is not None

    # Every result was moved into a subgenre subfolder; originals no longer at root.
    for result in data["results"]:
        assert result["organized_path"] is not None
        assert Path(result["organized_path"]).exists()
    assert not list(work.glob("*.wav"))  # all moved out of the root


def test_batch_job_low_confidence_goes_to_review(client: TestClient, tmp_path: Path):
    import soundfile as sf

    from edm_classifier.config import REVIEW_DIRNAME, settings

    work = tmp_path / "inbox"
    work.mkdir()
    sr = settings.audio.sample_rate
    for i in range(2):
        sf.write(work / f"t{i}.wav", [0.0] * sr, sr, subtype="PCM_16")

    # FakePredictor confidence is 0.86; a 0.95 threshold sends everything to review.
    resp = client.post(
        "/jobs", json={"directory": str(work), "mode": "move", "confidence_threshold": 0.95}
    )
    data = _wait_for_job(client, resp.json()["job_id"])
    assert data["status"] == "completed"
    assert data["review_count"] == 2
    assert sum(data["subgenre_counts"].values()) == 0  # none organized by subgenre
    review_dir = work / REVIEW_DIRNAME
    assert review_dir.is_dir() and len(list(review_dir.glob("*.wav"))) == 2
    for result in data["results"]:
        assert result["review"] is True
        assert result["second_choice"] is not None
        assert REVIEW_DIRNAME in result["organized_path"]


def test_batch_job_classify_only_does_not_move(client: TestClient, tmp_path: Path):
    import soundfile as sf

    from edm_classifier.config import settings

    work = tmp_path / "inbox"
    work.mkdir()
    sr = settings.audio.sample_rate
    sf.write(work / "a.wav", [0.0] * sr, sr, subtype="PCM_16")

    resp = client.post("/jobs", json={"directory": str(work), "mode": "classify"})
    data = _wait_for_job(client, resp.json()["job_id"])
    assert data["status"] == "completed"
    assert data["results"][0]["organized_path"] is None
    assert (work / "a.wav").exists()  # not moved


def test_job_not_found(client: TestClient):
    assert client.get("/jobs/deadbeef").status_code == 404


def test_create_job_missing_directory(client: TestClient, tmp_path: Path):
    resp = client.post("/jobs", json={"directory": str(tmp_path / "nope")})
    assert resp.status_code == 404
