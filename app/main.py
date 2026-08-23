import time
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.pipeline import analyze
from app.schemas import AnalyzeResponse

app = FastAPI(title="Contact Attribute Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    file: UploadFile = File(...),
    contact_id: str | None = Form(default=None),
) -> AnalyzeResponse:
    resolved_id = _resolve_contact_id(contact_id)
    audio_bytes = await file.read()
    started = time.perf_counter()
    result = analyze(audio_bytes, str(resolved_id))
    result.processing_ms = int((time.perf_counter() - started) * 1000)
    return result


def _resolve_contact_id(contact_id: str | None) -> UUID:
    if contact_id is None or contact_id == "":
        return uuid4()
    try:
        return UUID(contact_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="contact_id must be a UUID"
        ) from exc
