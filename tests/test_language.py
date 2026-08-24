from app.language import language_from_whisper_tokens


def test_language_from_whisper_tokens_reads_iso_code():
    tokens = ["<|startoftranscript|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>"]
    assert language_from_whisper_tokens(tokens) == "en"


def test_language_from_whisper_tokens_unknown_without_lang():
    tokens = ["<|startoftranscript|>", "<|transcribe|>"]
    assert language_from_whisper_tokens(tokens) == "unknown"
