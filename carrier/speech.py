"""Speech building blocks: Whisper STT (faster-whisper), Piper TTS, energy VAD, speaker features, prosody.

Models load lazily and are cached. Local copies under ~/models are used when present; otherwise
they download from Hugging Face on first use (Streamlit Community Cloud).
"""
from __future__ import annotations

import io
import os
import threading
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

SR = 16000
BATCH_MODEL = os.environ.get("BATCH_STT_MODEL", "small.en")
LIVE_MODEL = os.environ.get("LIVE_STT_MODEL", "base.en")
AGENT_VOICE = "en_US-amy-medium"
_lock = threading.Lock()
_whisper: dict = {}
_voices: dict = {}
PIPER_DIR = Path(os.environ.get("PIPER_DIR", Path.home() / "models" / "piper"))
SPK_MODEL = Path(os.environ.get("SPK_MODEL", Path.home() / "models" / "spk" / "resnet34.onnx"))


# ---- STT ----------------------------------------------------------------------
def whisper(name: str):
    from faster_whisper import WhisperModel
    with _lock:
        if name not in _whisper:
            _whisper[name] = WhisperModel(name, device="cpu", compute_type="int8",
                                          cpu_threads=int(os.environ.get("STT_THREADS", min(4, os.cpu_count() or 2))))
        return _whisper[name]


def transcribe(audio: np.ndarray, model: str | None = None, prompt: str | None = None, beam_size: int = 1) -> str:
    segs, _ = whisper(model or LIVE_MODEL).transcribe(audio, beam_size=beam_size, initial_prompt=prompt,
                                                      condition_on_previous_text=False, language="en",
                                                      vad_filter=False)
    return " ".join(s.text.strip() for s in segs).strip()


def to16k(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr == SR:
        return audio.astype(np.float32)
    g = gcd(SR, sr)
    return resample_poly(audio, SR // g, sr // g).astype(np.float32)


def read_audio(data: bytes) -> np.ndarray:
    import soundfile as sf
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    return to16k(audio, sr)


def wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---- TTS ----------------------------------------------------------------------
def _voice_path(name: str) -> Path:
    local = PIPER_DIR / f"{name}.onnx"
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download
    lang, speaker, quality = name.split("-")
    sub = f"{lang.split('_')[0]}/{lang}/{speaker}/{quality}/{name}"
    hf_hub_download("rhasspy/piper-voices", f"{sub}.onnx.json")
    return Path(hf_hub_download("rhasspy/piper-voices", f"{sub}.onnx"))


def voice(name: str = AGENT_VOICE):
    from piper import PiperVoice
    with _lock:
        if name not in _voices:
            p = _voice_path(name)
            cfg = p.with_suffix(".onnx.json")
            _voices[name] = PiperVoice.load(str(p), config_path=str(cfg) if cfg.exists() else None)
        return _voices[name]


def synth(text: str, voice_name: str = AGENT_VOICE, length_scale: float = 1.0, noise_scale: float = 0.667) -> tuple[np.ndarray, int]:
    from piper import SynthesisConfig
    from .textnorm import spoken
    v = voice(voice_name)
    cfg = SynthesisConfig(length_scale=length_scale, noise_scale=noise_scale)
    chunks = [c.audio_float_array for c in v.synthesize(spoken(text), syn_config=cfg)]
    audio = np.concatenate(chunks) if chunks else np.zeros(1, np.float32)
    return audio.astype(np.float32), v.config.sample_rate


def synth16k(text: str, voice_name: str = AGENT_VOICE, **kw) -> np.ndarray:
    a, sr = synth(text, voice_name, **kw)
    return to16k(a, sr)


def local_voices() -> list[str]:
    return sorted(p.stem for p in PIPER_DIR.glob("*.onnx"))


# ---- VAD and speaker features ---------------------------------------------------------
def frame_energy(audio: np.ndarray, frame: int = 320) -> np.ndarray:
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0)
    f = audio[: n * frame].reshape(n, frame)
    return 10 * np.log10(np.mean(f ** 2, axis=1) + 1e-10)


def vad_regions(audio: np.ndarray, min_gap: float = 0.3, min_len: float = 0.25) -> list[tuple[float, float]]:
    """Energy VAD -> speech regions (seconds). 20 ms frames, adaptive threshold."""
    e = frame_energy(audio)
    if len(e) == 0:
        return []
    noise = np.percentile(e, 10)
    thr = max(noise + 12, np.percentile(e, 50) - 15)
    regions, start = [], None
    for i, s in enumerate(e > thr):
        t = i * 0.02
        if s and start is None:
            start = t
        elif not s and start is not None:
            regions.append([start, t])
            start = None
    if start is not None:
        regions.append([start, len(e) * 0.02])
    merged: list[list[float]] = []
    for r in regions:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(round(a, 2), round(b, 2)) for a, b in merged if b - a >= min_len]


def _mel_fb(n_fft=512, n_mels=32, sr=SR):
    def hz2mel(h): return 2595 * np.log10(1 + h / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(60), hz2mel(7600), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        lo, c, r = bins[i - 1], bins[i], bins[i + 1]
        fb[i - 1, lo:c] = (np.arange(lo, c) - lo) / max(c - lo, 1)
        fb[i - 1, c:r] = (r - np.arange(c, r)) / max(r - c, 1)
    return fb


def pitch_track(audio: np.ndarray, frame: int = 640, hop: int = 320) -> np.ndarray:
    out = []
    lo, hi = SR // 400, SR // 70
    for s in range(0, len(audio) - frame, hop):
        x = audio[s:s + frame]
        if np.mean(x ** 2) < 1e-5:
            out.append(0.0)
            continue
        x = x - x.mean()
        ac = np.correlate(x, x, "full")[frame - 1:]
        if ac[0] <= 0:
            out.append(0.0)
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        out.append(SR / lag if ac[lag] / ac[0] > 0.45 else 0.0)
    return np.array(out)


_FB32 = _mel_fb()


def cheap_embedding(audio: np.ndarray) -> np.ndarray:
    """Lightweight speaker embedding: log-mel mean/std + pitch statistics (no model download)."""
    n_fft, hop = 512, 160
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))
    frames = np.lib.stride_tricks.sliding_window_view(audio, n_fft)[::hop] * np.hanning(n_fft)
    mel = np.log((np.abs(np.fft.rfft(frames, axis=1)) ** 2) @ _FB32.T + 1e-8)
    energy = mel.mean(axis=1)
    voiced = mel[energy > np.percentile(energy, 40)]
    f0 = pitch_track(audio)
    f0v = f0[f0 > 0]
    pitch = [np.median(f0v), np.std(f0v)] if len(f0v) else [0.0, 0.0]
    v = np.concatenate([voiced.mean(0) - voiced.mean(), voiced.std(0), np.log(np.array(pitch) + 1) * 4])
    return v / (np.linalg.norm(v) + 1e-9)


_spk = None


def neural_available() -> bool:
    return SPK_MODEL.exists()


def neural_embedding(audio: np.ndarray) -> np.ndarray:
    """WeSpeaker ResNet34 (ONNX) speaker embedding."""
    global _spk
    import onnxruntime as ort
    if _spk is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _spk = ort.InferenceSession(str(SPK_MODEL), so, providers=["CPUExecutionProvider"])
    x = audio.astype(np.float64) * 32768
    win, hop, n_fft = 400, 160, 512
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    frames = np.lib.stride_tricks.sliding_window_view(x, win)[::hop].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1]
    frames[:, 0] *= 0.03
    frames *= np.hanning(win + 1)[:-1] ** 0.85
    spec = np.abs(np.fft.rfft(frames, n_fft)) ** 2
    feats = np.log(np.maximum(spec @ _mel_fb(n_fft, 80).T, 1.19e-7))
    feats = (feats - feats.mean(axis=0)).astype(np.float32)[None]
    emb = _spk.run(None, {_spk.get_inputs()[0].name: feats})[0][0]
    return emb / (np.linalg.norm(emb) + 1e-9)


def speaker_embedding(audio: np.ndarray, neural: bool | None = None) -> np.ndarray:
    use = neural_available() if neural is None else neural and neural_available()
    return neural_embedding(audio) if use else cheap_embedding(audio)


def prosody(audio: np.ndarray, words: int) -> dict:
    """Loudness and speaking rate of a caller turn, used with text cues to predict mood."""
    e = frame_energy(audio)
    voiced = e[e > np.percentile(e, 30)] if len(e) else e
    loud_db = float(np.mean(voiced)) if len(voiced) else -100.0
    dur = max(0.5, sum(b - a for a, b in vad_regions(audio)) or len(audio) / SR)
    f0 = pitch_track(audio)
    f0v = f0[f0 > 0]
    rate = words / dur
    return {"loudness_db": round(loud_db, 1), "words_per_s": round(rate, 2),
            "pitch_hz": round(float(np.median(f0v)), 1) if len(f0v) else None,
            "loud": loud_db > -18, "fast": rate > 3.3}
