import logging
import time
from uuid import UUID, uuid4

from fastapi import FastAPI, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from app.audio import MAX_UPLOAD_BYTES, AudioDecodeError
from app.errors import AppError, error_body, request_id_var
from app.logging import configure_logging
from app.pipeline import analyze
from app.schemas import AnalyzeResponse

logger = logging.getLogger(__name__)

configure_logging()

app = FastAPI(title="Contact Attribute Service")


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        incoming = request.headers.get("x-request-id", "").strip()
        request_id = incoming or str(uuid4())
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        return response


app.add_middleware(RequestIdMiddleware)


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    logger.warning("client_error code=%s message=%s", exc.code, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code=exc.code, message=exc.message),
    )


@app.exception_handler(AudioDecodeError)
async def decode_error_handler(_request: Request, exc: AudioDecodeError) -> JSONResponse:
    logger.warning("invalid_audio message=%s", exc)
    return JSONResponse(
        status_code=400,
        content=error_body(code="invalid_audio", message=str(exc)),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("validation_error")
    return JSONResponse(
        status_code=422,
        content=error_body(code="validation_error", message="request validation failed"),
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    _request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "request failed"
    code = "not_found" if exc.status_code == 404 else "http_error"
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(code=code, message=message),
    )


@app.exception_handler(Exception)
async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error type=%s", type(exc).__name__)
    return JSONResponse(
        status_code=500,
        content=error_body(code="internal_error", message="internal error"),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_audio(
    request: Request,
    contact_id: str | None = Query(default=None),
) -> AnalyzeResponse:
    started = time.perf_counter()
    audio_bytes, form_contact_id = await _read_audio(request)
    try:
        resolved_id = _resolve_contact_id(form_contact_id or contact_id)
        result = analyze(audio_bytes, str(resolved_id))
        result.processing_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "analyze_ok contact_id=%s audio_quality=%s bytes=%s processing_ms=%s",
            result.contact_id,
            result.audio_quality.value,
            len(audio_bytes),
            result.processing_ms,
        )
        return result
    finally:
        # Drop the only copy of caller audio for this request.
        del audio_bytes


async def _read_audio(request: Request) -> tuple[bytes, str | None]:
    content_type = request.headers.get("content-type", "")
    if content_type.lower().startswith("multipart/form-data"):
        return await _read_multipart(request)
    body = await _read_limited_body(request)
    return body, None


async def _read_multipart(request: Request) -> tuple[bytes, str | None]:
    _reject_oversized_content_length(request)
    form = await request.form()
    try:
        upload = form.get("file")
        if not isinstance(upload, (UploadFile, StarletteUploadFile)):
            raise AppError(
                400, "missing_file", "multipart field 'file' is required"
            )
        data = await upload.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise AppError(413, "payload_too_large", "audio payload exceeds size limit")
        raw_id = form.get("contact_id")
        form_contact_id = raw_id if isinstance(raw_id, str) else None
        return data, form_contact_id
    finally:
        await form.close()


async def _read_limited_body(request: Request) -> bytes:
    _reject_oversized_content_length(request)
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise AppError(413, "payload_too_large", "audio payload exceeds size limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_oversized_content_length(request: Request) -> None:
    header = request.headers.get("content-length")
    if header is None:
        return
    try:
        length = int(header)
    except ValueError as exc:
        raise AppError(400, "invalid_content_length", "invalid Content-Length") from exc
    if length > MAX_UPLOAD_BYTES:
        raise AppError(413, "payload_too_large", "audio payload exceeds size limit")


def _resolve_contact_id(contact_id: str | None) -> UUID:
    if contact_id is None or contact_id == "":
        return uuid4()
    try:
        return UUID(contact_id)
    except ValueError as exc:
        raise AppError(422, "invalid_contact_id", "contact_id must be a UUID") from exc
