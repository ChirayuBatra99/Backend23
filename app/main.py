import time
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from starlette.datastructures import UploadFile as StarletteUploadFile

from app.audio import AudioDecodeError
from app.pipeline import analyze
from app.schemas import AnalyzeResponse

app = FastAPI(title="Contact Attribute Service")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    request: Request,
    contact_id: str | None = Query(default=None),
) -> AnalyzeResponse:
    audio_bytes, form_contact_id = await _read_audio(request)
    resolved_id = _resolve_contact_id(form_contact_id or contact_id)
    started = time.perf_counter()
    try:
        result = analyze(audio_bytes, str(resolved_id))
    except AudioDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result.processing_ms = int((time.perf_counter() - started) * 1000)
    return result


async def _read_audio(request: Request) -> tuple[bytes, str | None]:
    content_type = request.headers.get("content-type", "")
    if content_type.lower().startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, (UploadFile, StarletteUploadFile)):
            raise HTTPException(
                status_code=400, detail="multipart field 'file' is required"
            )
        data = await upload.read()
        raw_id = form.get("contact_id")
        form_contact_id = raw_id if isinstance(raw_id, str) else None
        return data, form_contact_id

    data = await request.body()
    return data, None


def _resolve_contact_id(contact_id: str | None) -> UUID:
    if contact_id is None or contact_id == "":
        return uuid4()
    try:
        return UUID(contact_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail="contact_id must be a UUID"
        ) from exc
