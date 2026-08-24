"""WebSocket progressive analysis.

Clients send binary audio (PCM s16le 16 kHz mono by default, or encoded
bytes if encoding=encoded). The server keeps an in-memory buffer, re-runs
the pipeline as speech accumulates, and drops the buffer when the socket
closes. Text message {"event":"end"} flushes a final result.
"""

from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

from app.audio import (
    MAX_DECODE_SECONDS,
    MAX_UPLOAD_BYTES,
    TARGET_SAMPLE_RATE,
    AudioDecodeError,
    decode_to_pcm16k_mono,
    waveform_from_pcm16le,
)
from app.errors import error_body
from app.pipeline import analyze_waveform
from app.schemas import StreamUpdate

logger = logging.getLogger(__name__)

MIN_INFER_SECONDS = 0.5
INFER_STRIDE_SECONDS = 0.5


async def handle_analyze_socket(
    websocket: WebSocket,
    contact_id: str | None = None,
    encoding: str = "pcm",
) -> None:
    await websocket.accept()
    resolved_id = contact_id or str(uuid4())
    encoding = encoding.lower().strip() or "pcm"
    buffer = bytearray()
    last_infer_bytes = 0
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("text") is not None:
                if _is_end_event(message["text"]):
                    await _emit(websocket, buffer, encoding, resolved_id, partial=False)
                    break
                continue
            chunk = message.get("bytes") or b""
            if not chunk:
                continue
            if len(buffer) + len(chunk) > MAX_UPLOAD_BYTES:
                await websocket.send_json(
                    error_body(
                        code="payload_too_large",
                        message="audio payload exceeds size limit",
                    )
                )
                break
            buffer.extend(chunk)
            if _should_infer(len(buffer), last_infer_bytes, encoding):
                await _emit(websocket, buffer, encoding, resolved_id, partial=True)
                last_infer_bytes = len(buffer)
    except WebSocketDisconnect:
        logger.info("ws_disconnect contact_id=%s", resolved_id)
    except Exception:
        logger.exception("ws_unhandled contact_id=%s", resolved_id)
        try:
            await websocket.send_json(
                error_body(code="internal_error", message="internal error")
            )
        except Exception:
            pass
    finally:
        buffer.clear()
        try:
            await websocket.close()
        except Exception:
            pass


def _is_end_event(text: str) -> bool:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text.strip().lower() == "end"
    return payload.get("event") == "end"


def _should_infer(total_bytes: int, last_infer_bytes: int, encoding: str) -> bool:
    if encoding != "pcm":
        return total_bytes > last_infer_bytes and total_bytes >= 2048
    min_bytes = int(MIN_INFER_SECONDS * TARGET_SAMPLE_RATE) * 2
    stride = int(INFER_STRIDE_SECONDS * TARGET_SAMPLE_RATE) * 2
    if total_bytes < min_bytes:
        return False
    return last_infer_bytes == 0 or total_bytes - last_infer_bytes >= stride


async def _emit(
    websocket: WebSocket,
    buffer: bytearray,
    encoding: str,
    contact_id: str,
    *,
    partial: bool,
) -> None:
    if not buffer:
        return
    started = time.perf_counter()
    try:
        if encoding == "pcm":
            max_pcm = int(MAX_DECODE_SECONDS * TARGET_SAMPLE_RATE) * 2
            waveform = waveform_from_pcm16le(bytes(buffer[-max_pcm:]))
            buffered_ms = int(1000 * (len(buffer) / 2) / TARGET_SAMPLE_RATE)
        else:
            waveform = decode_to_pcm16k_mono(bytes(buffer))
            buffered_ms = int(waveform.duration_s * 1000)
        result = analyze_waveform(waveform, contact_id)
    except AudioDecodeError as exc:
        await websocket.send_json(
            error_body(code="invalid_audio", message=str(exc))
        )
        return
    result.processing_ms = int((time.perf_counter() - started) * 1000)
    update = StreamUpdate(
        **result.model_dump(),
        partial=partial,
        buffered_ms=buffered_ms,
    )
    await websocket.send_json(json.loads(update.model_dump_json()))
