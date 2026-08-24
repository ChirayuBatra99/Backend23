# Contact attribute service

Stateless HTTP (and WebSocket) service that estimates **gender**, **age bracket**, **audio quality**, and best-effort **language** from a short audio clip. Built for logistics voice agents that must personalize a call without a prior profile. Audio is processed in memory and discarded when the request ends.

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

ffmpeg is required for encoded formats (mp3, m4a, webm, …). If it is not on `PATH`, `imageio-ffmpeg` provides a bundled binary.

Smoke test:

```bash
curl -s -F "file=@samples/sample.wav" http://127.0.0.1:8000/analyze
```

The first inference downloads public Hugging Face weights (audEERING wav2vec2, and Whisper tiny if language ID runs). Later requests use the cache.

```bash
pytest -q
```

## Docker

```bash
docker compose up --build
```

The API listens on `http://localhost:8000`. Weights land in a named volume (`hf-cache`) so they survive restarts. No other external services are required.

## API

`GET /health` → `{"status":"ok"}`

`POST /analyze`

- Multipart field `file`, optional form/query `contact_id` (UUID)
- Raw body (`audio/wav`, `application/octet-stream`, chunked) also accepted
- Header `X-Request-ID` is echoed (generated if omitted)

Success:

```json
{
  "contact_id": "uuid",
  "gender": {"prediction": "male|female|unknown", "confidence": 0.0},
  "age_bracket": {"prediction": "18-30|31-45|46-60|60+|unknown", "confidence": 0.0},
  "processing_ms": 0,
  "audio_quality": "good|degraded|insufficient",
  "language": {"prediction": "en", "confidence": 0.6}
}
```

Errors:

```json
{"error": {"code": "invalid_audio", "message": "...", "request_id": "..."}}
```

| Code | Status |
| --- | --- |
| `invalid_audio`, `missing_file` | 400 |
| `payload_too_large` | 413 |
| `invalid_contact_id` | 422 |
| `internal_error` | 500 |

`audio_quality=insufficient` (too short or silent) returns `unknown` for gender/age and **does not** run the neural net.

### WebSocket (bonus)

`WS /ws/analyze?contact_id=<uuid>&encoding=pcm`

- Binary frames: PCM s16le, 16 kHz, mono (`encoding=encoded` to send mp3/wav/webm fragments concatenated in memory)
- Text `{"event":"end"}` for a final result
- Server emits the same payload plus `partial` and `buffered_ms`
- Buffer is dropped on disconnect (never written to disk)

## Model choice

Gender and age use [audEERING wav2vec2-large-robust-6-ft-age-gender](https://huggingface.co/audeering/wav2vec2-large-robust-6-ft-age-gender): one public checkpoint for both heads, 6 transformer layers (closer to a 500 ms / 5 s budget than the 24-layer variant). Age is a 0–1 score mapped to years then to brackets; gender is child/female/male. Child-dominant or low-confidence outputs become `unknown`. Degraded audio still infers, with confidence scaled down.

Language (bonus) uses Whisper tiny token prefixes (`<|en|>`, …). Accent is **not** predicted; that needs dialect-labelled data we do not ship. Whisper and audEERING weights are research licences (Whisper MIT; audEERING CC-BY-NC-SA). Swap the age/gender net before a commercial deploy.

## Privacy

Caller audio is PII. Decode uses ffmpeg stdin/stdout only. Multipart spools are closed at the end of the request. Bytes are deleted when the handler returns. Logs record `contact_id`, size, quality, and latency — never samples. Nothing is written under `/tmp` by our code for the audio path.

## Eval (bonus)

Common Voice must be downloaded separately (Mozilla terms). Then:

```bash
python scripts/eval_common_voice.py --tsv /path/validated.tsv --clips /path/clips --max-samples 200
```

Prints gender/age accuracy and expected calibration error (ECE). Mapping lives in `app/eval_cv.py` (CV decade labels → our brackets).

## Scaling to ~1000 concurrent calls

See [DESIGN.md](DESIGN.md). Short version: one worker is not enough; run several CPU (or GPU) replicas behind a load balancer, cap in-flight inferences, and keep sessions stateless so WebSocket buffers stay on the replica that owns the socket.

## Limitations

- Age from 5 s of telephony audio is noisy; treat brackets as priors, not identity.
- Warehouse noise is flagged `degraded`, not “fixed.”
- Overlapping speakers are not separated (no diarization on the hot path).
- First request is slow (weight download + model load).
