from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def _raw_post(content: bytes, extra_headers: dict[str, str] | None = None):
    headers = {"content-type": "application/octet-stream"}
    if extra_headers:
        headers.update(extra_headers)
    return client.post("/analyze", content=content, headers=headers)


def test_invalid_audio_returns_error_envelope():
    response = _raw_post(b"not-audio")
    assert response.status_code == 400
    body = response.json()["error"]
    assert body["code"] == "invalid_audio"
    assert body["message"]
    assert body["request_id"]


def test_invalid_contact_id_returns_error_envelope():
    response = client.post(
        "/analyze",
        files={"file": ("clip.wav", b"xxxx", "audio/wav")},
        data={"contact_id": "not-a-uuid"},
    )
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "invalid_contact_id"
    assert "UUID" in body["message"]


def test_missing_multipart_file_returns_error_envelope():
    response = client.post(
        "/analyze",
        files={"other": ("note.txt", b"hi", "text/plain")},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_file"


def test_payload_too_large_is_rejected():
    with patch("app.main.MAX_UPLOAD_BYTES", 16):
        response = _raw_post(b"x" * 64)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


def test_echoes_request_id_header():
    response = _raw_post(b"not-audio", extra_headers={"X-Request-ID": "req-123"})
    assert response.headers["X-Request-ID"] == "req-123"
    assert response.json()["error"]["request_id"] == "req-123"


def test_generates_request_id_when_missing():
    response = _raw_post(b"not-audio")
    assert response.headers["X-Request-ID"]
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


def test_unhandled_error_hides_internal_details():
    with patch("app.main.analyze", side_effect=RuntimeError("secret-stack")):
        response = _raw_post(b"not-empty")
    assert response.status_code == 500
    body = response.json()["error"]
    assert body["code"] == "internal_error"
    assert body["message"] == "internal error"
    assert "secret-stack" not in response.text
