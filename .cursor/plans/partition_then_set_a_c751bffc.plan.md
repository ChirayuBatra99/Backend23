---
name: Partition then Set A
overview: Split the assignment into sequenced work sets. Set A is a FastAPI skeleton that locks the required JSON contract and request handling, with stub predictions and no Docker or ML yet.
todos:
  - id: set-a-skeleton
    content: "Set A: FastAPI app, Pydantic contract, stub POST /analyze, health, pytest contract test"
    status: completed
  - id: set-b-ingest
    content: "Set B (later): in-memory ffmpeg decode + audio_quality heuristics"
    status: pending
  - id: set-c-infer
    content: "Set C (later): gender/age models mapped to contract + quality gating"
    status: pending
  - id: set-d-ops-tests
    content: "Set D/E (later): errors, logging, no persistence, sample audio + tests"
    status: pending
  - id: set-f-g-bonus-submit
    content: "Set F/G (later): bonuses, Docker, README, design write-up"
    status: pending
isProject: false
---

# Partition the assignment, then Set A

The product is a **stateless, in-memory** service: audio in, `{gender, age_bracket, audio_quality, processing_ms}` out, under ~500ms on a 5s chunk. No caller history. Docker, compose, and private GitHub come later.

Suggested stack (locked later in Set C, not Set A): FastAPI + ffmpeg for decode; SpeechBrain or a small Hugging Face age/gender model for inference; librosa/openSMILE only if we need extra acoustic features or quality metrics. Whisper is a poor fit for gender/age (it is ASR). pyannote is for “who spoke when,” not demographics. VoxCeleb has no age labels; Common Voice can support an eval harness later.

```mermaid
flowchart LR
  client[Client] --> api[POST_analyze]
  api --> ingest[Decode_and_quality]
  ingest --> infer[Gender_and_age]
  infer --> json[JSON_response]
```

## Work sets (do in order)

**Set A — Contract and skeleton (this slice)**  
Empty FastAPI app that accepts audio and returns the **exact** response schema with stubs (`unknown` / low confidence / `insufficient` until later sets fill them in). Proves the API shape, timing field, and “no disk persistence” rule. No models, no Docker.

**Set B — Audio ingestion**  
Decode messy logistics audio (mp3, m4a, wav, webm, telephony codecs) via ffmpeg to 16 kHz mono in memory. Chunked HTTP body and/or multipart. Compute `audio_quality` (`good` / `degraded` / `insufficient`) from duration, RMS, clipping, SNR-ish heuristics. Drop or refuse audio that is too short/silent.

**Set C — Attribute inference**  
Gender (`male` | `female` | `unknown`) and age bracket (`18-30` | `31-45` | `46-60` | `60+` | `unknown`) with confidences. Map model outputs to the contract; if quality is `insufficient`, force `unknown` instead of guessing.

**Set D — Reliability**  
Validation errors, 4xx/5xx JSON, request-id logging, `processing_ms`, never write audio to disk (temp buffers only, deleted when the request ends).

**Set E — Tests and sample audio**  
At least one unit/integration test plus a sample clip (or fetch instructions) for a smoke `POST /analyze`.

**Set F — Bonuses (optional)**  
WebSocket progressive predictions; language/accent field; Common Voice eval script.

**Set G — Ops and submission (later)**  
Dockerfile, `docker compose up`, README, ~200-word design write-up, private repo.

## Set A in detail (implement after you confirm)

Create a small Python package, e.g.:

- `app/main.py` — FastAPI app, `POST /analyze` (multipart file) and `GET /health`
- `app/schemas.py` — Pydantic models matching the expected JSON (enums for gender, age bracket, audio quality)
- `app/pipeline.py` — `analyze(audio_bytes) -> result` stub: no decode/ML yet; returns `unknown` + `audio_quality: insufficient`
- `tests/test_analyze_contract.py` — post a tiny fake wav/bytes, assert status 200 and schema keys/enums
- `requirements.txt` — `fastapi`, `uvicorn`, `python-multipart`, `pytest`, `httpx`

Rules for Set A:

- `contact_id`: accept optional client UUID or generate one per request
- Measure `processing_ms` even for the stub
- Do not save uploads
- Ignore Docker, models, WebSocket, eval, and README until later sets

Out of Set A you can hit `POST /analyze` and get a valid contract payload. Set B then replaces the stub ingest; Set C replaces the stub predictions.
