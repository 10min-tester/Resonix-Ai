import gc
import importlib.util
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
import zipfile
from typing import Any
from urllib.parse import quote

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import uvicorn
from scipy import signal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


APP_VERSION = "1.0.0"
APP_HOME = os.path.join(os.path.expanduser("~"), ".packaged_audio_ai")
LOG_DIR = os.path.join(APP_HOME, "logs")
UPLOAD_DIR = os.path.join(APP_HOME, "temp_uploads")
OUTPUT_DIR = os.path.join(APP_HOME, "temp_outputs")
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
OUTPUT_SAMPLE_RATES = {44100, 48000, 96000}
OUTPUT_BIT_DEPTHS = {"16", "24"}
FINAL_PCM_CEILING_DB = -0.3
MANUAL_DSP_SAFE_CEILING_DB = -1.5
STEM_FINAL_CEILING_DB = -1.2
LOUDNESS_ABSOLUTE_GATE_LUFS = -70.0
LOUDNESS_RELATIVE_GATE_OFFSET_LU = -10.0
LOUDNESS_BLOCK_SECONDS = 0.400
LOUDNESS_BLOCK_OVERLAP = 0.75
STEM_INTENSITY_CAP = 0.45
STEM_VOCAL_INTENSITY_CAP = 0.36
STEM_VOCAL_BLEED_CLEANUP_STRENGTH = 0.32
STEM_GAIN_MATCH_CEILING_DB = -2.0
STEM_RESIDUAL_BLEND = 0.16
STEM_RESIDUAL_MAX_SOURCE_RATIO = 0.08
STEM_RESIDUAL_MIN_SOURCE_RATIO = 0.003
STEM_RESIDUAL_LOWCUT_HZ = 120.0
QUALITY_GUARD_MAX_BLEND = 0.36
QUALITY_GUARD_BASE_BLEND = 0.14
STEM_QUALITY_MODES = {"fast", "balanced", "precision"}
STEM_VOCAL_GAIN = 0.95
STEM_INSTRUMENTAL_GAIN = 0.95
STEM_REMIX_GAINS = {
    "vocals": 0.95,
    "instrumental": 0.95,
    "drums": 0.92,
    "bass": 0.95,
    "other": 0.95,
}
STEM_GAIN_MATCH_MAX_BOOST_DB = {
    "vocals": 2.0,
    "instrumental": 2.0,
    "drums": 1.4,
    "bass": 1.6,
    "other": 1.8,
}
ENABLE_GTCRN_MODEL = os.getenv("RESONIX_ENABLE_GTCRN", "0") == "1"
GTCRN_MAX_MODEL_SECONDS = 8.0
PROCESSING_TARGETS = [
    "restore",
    "hifi_clean",
    "hifi_bright",
    "warm_analog",
    "loud_modern",
    "bass_boost",
    "voice_focus",
]
TARGET_PROFILES = {
    "restore": {
        "lowcut_hz": 35,
        "low_boost_db": 0.0,
        "mid_cut_db": 0.0,
        "high_boost_db": 0.6,
        "compress_ratio": 1.2,
        "target_lufs": -18.0,
        "limiter_ceiling_db": -1.5,
        "exciter_amount": 0.05,
        "saturation_amount": 0.02,
    },
    "hifi_clean": {
        "lowcut_hz": 38,
        "low_boost_db": 0.4,
        "mid_cut_db": 0.0,
        "high_boost_db": 1.0,
        "compress_ratio": 1.25,
        "target_lufs": -16.0,
        "limiter_ceiling_db": -1.5,
        "exciter_amount": 0.08,
        "saturation_amount": 0.03,
    },
    "hifi_bright": {
        "lowcut_hz": 40,
        "low_boost_db": 0.0,
        "mid_cut_db": -0.6,
        "high_boost_db": 2.4,
        "compress_ratio": 1.25,
        "target_lufs": -15.0,
        "limiter_ceiling_db": -1.7,
        "exciter_amount": 0.18,
        "saturation_amount": 0.02,
    },
    "warm_analog": {
        "lowcut_hz": 35,
        "low_boost_db": 1.8,
        "mid_cut_db": 0.0,
        "high_boost_db": -0.8,
        "compress_ratio": 1.45,
        "target_lufs": -17.0,
        "limiter_ceiling_db": -1.5,
        "exciter_amount": 0.03,
        "saturation_amount": 0.16,
    },
    "loud_modern": {
        "lowcut_hz": 38,
        "low_boost_db": 1.0,
        "mid_cut_db": -0.3,
        "high_boost_db": 1.6,
        "compress_ratio": 2.8,
        "target_lufs": -12.0,
        "limiter_ceiling_db": -1.2,
        "exciter_amount": 0.12,
        "saturation_amount": 0.1,
    },
    "bass_boost": {
        "lowcut_hz": 30,
        "low_boost_db": 2.6,
        "mid_cut_db": -0.4,
        "high_boost_db": 0.3,
        "compress_ratio": 1.7,
        "target_lufs": -15.0,
        "limiter_ceiling_db": -1.7,
        "exciter_amount": 0.04,
        "saturation_amount": 0.08,
    },
    "voice_focus": {
        "lowcut_hz": 110,
        "low_boost_db": -1.5,
        "mid_cut_db": 0.8,
        "high_boost_db": 1.8,
        "compress_ratio": 1.8,
        "target_lufs": -16.0,
        "limiter_ceiling_db": -1.5,
        "exciter_amount": 0.1,
        "saturation_amount": 0.04,
    },
}

for directory in (LOG_DIR, UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(directory, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Resonix AI Engine", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_resource_path(relative_path: str) -> str:
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)


def validate_audio_file(filename: str) -> str:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use wav, mp3, m4a, or flac.",
        )
    return ext


def save_upload(file: UploadFile, ext: str) -> str:
    task_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return input_path


def safe_download_stem(filename: str, fallback: str = "audio") -> str:
    stem = os.path.splitext(os.path.basename(filename or ""))[0].strip() or fallback
    blocked = '<>:"/\\|?*'
    cleaned = "".join("_" if char in blocked or ord(char) < 32 else char for char in stem)
    return cleaned[:80].strip(" ._") or fallback


def safe_download_filename(filename: str | None, fallback: str = "download.wav") -> str:
    raw_name = os.path.basename(filename or "").strip() or fallback
    blocked = '<>:"/\\|?*'
    cleaned = "".join("_" if char in blocked or ord(char) < 32 else char for char in raw_name)
    cleaned = cleaned[:140].strip(" ._") or fallback
    return cleaned or fallback


def build_download_url(output_filename: str, download_name: str) -> str:
    safe_output = quote(os.path.basename(output_filename), safe="")
    safe_name = quote(safe_download_filename(download_name), safe="")
    return f"/api/download/{safe_output}?name={safe_name}"


def initialize_onnx_session(model_path: str):
    available_providers = ort.get_available_providers()
    logger.info("[SYSTEM] Available ONNX providers: %s", available_providers)

    providers_to_use: list[Any] = []
    if "CUDAExecutionProvider" in available_providers:
        providers_to_use.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                },
            )
        )

    if "DmlExecutionProvider" in available_providers:
        providers_to_use.append(
            (
                "DmlExecutionProvider",
                {
                    "device_id": 0,
                    "memory_limit": 2 * 1024 * 1024 * 1024,
                },
            )
        )

    providers_to_use.append("CPUExecutionProvider")

    sess_options = ort.SessionOptions()
    sess_options.enable_cpu_mem_arena = False
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    if not os.path.exists(model_path):
        logger.warning("[WARN] Model file not found: %s", model_path)
        return None

    try:
        session = ort.InferenceSession(model_path, sess_options, providers=providers_to_use)
        logger.info("[SYSTEM] Active ONNX providers: %s", session.get_providers())
        return session
    except Exception:
        logger.exception("[ERROR] ONNX session initialization failed.")
        return None


def load_audio(path: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    try:
        info = sf.info(path)
        source_info = {
            "channels": int(info.channels),
            "frames": int(info.frames),
            "samplerate": int(info.samplerate),
            "duration": float(info.duration),
            "format": info.format,
            "subtype": info.subtype,
        }
    except Exception:
        logger.exception("[WARN] soundfile metadata read failed.")
        source_info = {}

    audio, sr = librosa.load(path, sr=None, mono=False)
    if audio.size == 0:
        raise HTTPException(status_code=400, detail="Audio data is empty.")
    return audio.astype(np.float32), int(sr), source_info


def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0).astype(np.float32)


def as_channel_matrix(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio[np.newaxis, :].astype(np.float32)
    return audio.astype(np.float32)


def audio_channels(audio: np.ndarray) -> list[np.ndarray]:
    if audio.ndim == 1:
        return [audio]
    return [audio[index] for index in range(audio.shape[0])]


def stack_channels(channels: list[np.ndarray], was_mono: bool) -> np.ndarray:
    if was_mono:
        return channels[0].astype(np.float32)
    min_len = min(len(channel) for channel in channels)
    return np.stack([channel[:min_len] for channel in channels], axis=0).astype(np.float32)


def choose_output_sample_rate(sr: int) -> int:
    if sr < 32000:
        return 44100
    return sr


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    channels = as_channel_matrix(audio)
    resampled_channels = [
        librosa.resample(channel, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
        for channel in channels
    ]
    return stack_channels(resampled_channels, audio.ndim == 1)


def match_channel_balance(source_audio: np.ndarray, processed_audio: np.ndarray) -> np.ndarray:
    if source_audio.ndim == 1 or processed_audio.ndim == 1:
        return processed_audio

    channel_count = min(source_audio.shape[0], processed_audio.shape[0])
    if channel_count < 2:
        return processed_audio

    matched = processed_audio.copy()
    source_rms = np.array(
        [np.sqrt(np.mean(np.square(source_audio[index]))) for index in range(channel_count)],
        dtype=np.float64,
    )
    processed_rms = np.array(
        [np.sqrt(np.mean(np.square(processed_audio[index]))) for index in range(channel_count)],
        dtype=np.float64,
    )

    if float(np.max(source_rms)) <= 1e-9 or float(np.max(processed_rms)) <= 1e-9:
        return processed_audio

    source_ratio = source_rms / (float(np.mean(source_rms)) + 1e-9)
    processed_ratio = processed_rms / (float(np.mean(processed_rms)) + 1e-9)
    gains = np.clip(source_ratio / (processed_ratio + 1e-9), 0.5, 2.0)

    for index in range(channel_count):
        matched[index] = matched[index] * gains[index]

    return matched.astype(np.float32)


def stereo_to_mid_side(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = audio[0]
    right = audio[1]
    min_len = min(len(left), len(right))
    left = left[:min_len]
    right = right[:min_len]
    return ((left + right) * 0.5).astype(np.float32), ((left - right) * 0.5).astype(np.float32)


def mid_side_to_stereo(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    min_len = min(len(mid), len(side))
    mid = mid[:min_len]
    side = side[:min_len]
    return np.stack([mid + side, mid - side], axis=0).astype(np.float32)


def match_mid_side_balance(source_audio: np.ndarray, processed_audio: np.ndarray) -> np.ndarray:
    if source_audio.ndim == 1 or processed_audio.ndim == 1:
        return processed_audio
    if source_audio.shape[0] != 2 or processed_audio.shape[0] != 2:
        return processed_audio

    source_mid, source_side = stereo_to_mid_side(source_audio)
    processed_mid, processed_side = stereo_to_mid_side(processed_audio)
    source_mid_rms = float(np.sqrt(np.mean(np.square(source_mid)))) + 1e-9
    source_side_rms = float(np.sqrt(np.mean(np.square(source_side))))
    processed_mid_rms = float(np.sqrt(np.mean(np.square(processed_mid)))) + 1e-9
    processed_side_rms = float(np.sqrt(np.mean(np.square(processed_side))))

    if processed_side_rms <= 1e-9:
        return processed_audio

    source_width = source_side_rms / source_mid_rms
    processed_width = processed_side_rms / processed_mid_rms
    side_gain = float(np.clip(source_width / (processed_width + 1e-9), 0.25, 4.0))
    return mid_side_to_stereo(processed_mid, processed_side * side_gain)


def lowpass_component(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    if len(audio) < 16 or sr <= 0:
        return np.zeros_like(audio, dtype=np.float32)

    cutoff = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    if cutoff >= sr * 0.49:
        return audio.astype(np.float32)

    sos = signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    try:
        return signal.sosfiltfilt(sos, audio).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, audio).astype(np.float32)


def bandpass_component(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    if len(audio) < 16 or sr <= 0:
        return np.zeros_like(audio, dtype=np.float32)

    low_hz = float(np.clip(low_hz, 20.0, sr * 0.45))
    high_hz = float(np.clip(high_hz, low_hz + 50.0, sr * 0.48))
    if high_hz <= low_hz + 20.0:
        return np.zeros_like(audio, dtype=np.float32)

    sos = signal.butter(2, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    try:
        return signal.sosfiltfilt(sos, audio).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, audio).astype(np.float32)


def apply_low_bass_phase_guard(
    source_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 130.0,
) -> tuple[np.ndarray, float]:
    if source_audio.ndim == 1 or processed_audio.ndim == 1:
        return processed_audio.astype(np.float32), 0.0
    if source_audio.shape[0] != 2 or processed_audio.shape[0] != 2:
        return processed_audio.astype(np.float32), 0.0

    source_mid, source_side = stereo_to_mid_side(source_audio)
    processed_mid, processed_side = stereo_to_mid_side(processed_audio)
    source_low_mid = lowpass_component(source_mid, sr, cutoff_hz)
    source_low_side = lowpass_component(source_side, sr, cutoff_hz)
    processed_low_mid = lowpass_component(processed_mid, sr, cutoff_hz)
    processed_low_side = lowpass_component(processed_side, sr, cutoff_hz)

    source_ratio = rms(source_low_side) / (rms(source_low_mid) + 1e-9)
    processed_ratio = rms(processed_low_side) / (rms(processed_low_mid) + 1e-9)
    allowed_ratio = max(source_ratio * 1.25, 0.10)
    if processed_ratio <= allowed_ratio or processed_ratio <= 1e-9:
        return processed_audio.astype(np.float32), 0.0

    low_side_gain = float(np.clip(allowed_ratio / processed_ratio, 0.35, 1.0))
    corrected_side = processed_side - processed_low_side * (1.0 - low_side_gain)
    return mid_side_to_stereo(processed_mid, corrected_side), db(low_side_gain)


def resolve_output_bit_depth(bit_depth: str | None) -> str:
    value = str(bit_depth or "24").strip()
    return value if value in OUTPUT_BIT_DEPTHS else "24"


def resolve_output_sample_rate(option: str | None, source_sr: int, default_sr: int) -> int:
    value = str(option or "auto").strip().lower()
    if value == "source":
        return int(source_sr)
    if value == "auto":
        return int(default_sr)
    try:
        requested = int(value)
    except ValueError:
        return int(default_sr)
    return requested if requested in OUTPUT_SAMPLE_RATES else int(default_sr)


def apply_sample_peak_guard(audio: np.ndarray, ceiling_db: float = FINAL_PCM_CEILING_DB) -> tuple[np.ndarray, float]:
    ceiling = 10 ** (ceiling_db / 20.0)
    sanitized = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    peak = float(np.max(np.abs(sanitized))) if sanitized.size else 0.0
    if peak <= ceiling or peak <= 1e-9:
        return sanitized, 0.0

    gain = ceiling / peak
    return (sanitized * gain).astype(np.float32), db(gain)


def apply_tpdf_dither(audio: np.ndarray, bit_depth: str | None) -> np.ndarray:
    if resolve_output_bit_depth(bit_depth) != "16":
        return audio.astype(np.float32)

    lsb = 1.0 / 32768.0
    noise = (np.random.random(audio.shape) - np.random.random(audio.shape)) * lsb
    dithered = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0) + noise
    return np.clip(dithered, -1.0, 1.0).astype(np.float32)


def write_audio(path: str, audio: np.ndarray, sr: int, bit_depth: str | None = "24") -> None:
    audio, _ = apply_sample_peak_guard(audio)
    subtype = None
    if os.path.splitext(path)[1].lower() == ".wav":
        resolved_bit_depth = resolve_output_bit_depth(bit_depth)
        subtype = "PCM_16" if resolved_bit_depth == "16" else "PCM_24"
        audio = apply_tpdf_dither(audio, resolved_bit_depth)
    if audio.ndim == 1:
        sf.write(path, audio, sr, subtype=subtype)
        return
    sf.write(path, audio.T, sr, subtype=subtype)


def db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-9)))


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def high_shelf_sos(sr: int, frequency_hz: float = 1681.974450955533, gain_db: float = 4.0, q: float = 0.7071752369554196) -> np.ndarray:
    frequency_hz = float(np.clip(frequency_hz, 20.0, max(20.0, sr * 0.45)))
    a = 10 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * frequency_hz / max(float(sr), 1.0)
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    sqrt_a = np.sqrt(a)

    b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha)
    b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0)
    b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha)
    a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha
    a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0)
    a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def apply_loudness_k_weighting(channels: np.ndarray, sr: int) -> np.ndarray:
    if channels.size == 0 or sr <= 0:
        return channels.astype(np.float64)
    if sr <= 100:
        return channels.astype(np.float64)

    weighted = channels.astype(np.float64, copy=True)
    try:
        shelf = high_shelf_sos(sr)
        highpass_cutoff = float(np.clip(38.0, 10.0, max(10.0, sr * 0.45)))
        highpass = signal.butter(2, highpass_cutoff, btype="highpass", fs=sr, output="sos")
        for index in range(weighted.shape[0]):
            weighted[index] = signal.sosfilt(shelf, weighted[index])
            weighted[index] = signal.sosfilt(highpass, weighted[index])
    except Exception:
        logger.debug("[LOUDNESS] K-weighting fallback used.", exc_info=True)
    return weighted


def estimate_integrated_loudness(audio: np.ndarray, sr: int | None = None) -> float:
    channels = as_channel_matrix(audio)
    if channels.size == 0:
        return -180.0

    rms_value = rms(channels)
    fallback_lufs = db(rms_value) - 0.691
    if rms_value <= 1e-12 or sr is None or sr <= 0:
        return fallback_lufs

    weighted = apply_loudness_k_weighting(channels, sr)
    block_size = max(int(round(sr * LOUDNESS_BLOCK_SECONDS)), 1)
    hop_size = max(int(round(block_size * (1.0 - LOUDNESS_BLOCK_OVERLAP))), 1)
    if weighted.shape[-1] < block_size:
        energy = float(np.mean(np.square(weighted)))
        return -0.691 + 10.0 * np.log10(max(energy, 1e-18))

    block_energies: list[float] = []
    for start in range(0, weighted.shape[-1] - block_size + 1, hop_size):
        block = weighted[:, start:start + block_size]
        block_energies.append(float(np.mean(np.sum(np.square(block), axis=0))))

    energies = np.asarray(block_energies, dtype=np.float64)
    if energies.size == 0:
        return fallback_lufs

    block_loudness = -0.691 + 10.0 * np.log10(np.maximum(energies, 1e-18))
    absolute_mask = block_loudness > LOUDNESS_ABSOLUTE_GATE_LUFS
    if not np.any(absolute_mask):
        return fallback_lufs

    absolute_energies = energies[absolute_mask]
    ungated_loudness = -0.691 + 10.0 * np.log10(max(float(np.mean(absolute_energies)), 1e-18))
    relative_gate = ungated_loudness + LOUDNESS_RELATIVE_GATE_OFFSET_LU
    gated_energies = absolute_energies[block_loudness[absolute_mask] > relative_gate]
    if gated_energies.size == 0:
        gated_energies = absolute_energies

    return float(-0.691 + 10.0 * np.log10(max(float(np.mean(gated_energies)), 1e-18)))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or not np.isfinite(value):
            return default
        return float(value)
    except Exception:
        return default


def band_energy(spectrum_power: np.ndarray, freqs: np.ndarray, start_hz: float, end_hz: float) -> float:
    total = float(np.sum(spectrum_power)) or 1.0
    mask = (freqs >= start_hz) & (freqs < end_hz)
    return float(np.sum(spectrum_power[mask]) / total)


def band_separation_score(low_energy: float, mid_energy: float, high_energy: float) -> tuple[float, dict[str, float]]:
    total = max(float(low_energy + mid_energy + high_energy), 1e-9)
    low = float(low_energy / total)
    mid = float(mid_energy / total)
    high = float(high_energy / total)
    masking_penalty = (
        max(0.0, low - 0.42) * 1.4
        + max(0.0, high - 0.38) * 1.0
        + max(0.0, 0.24 - mid) * 1.6
    )
    balance_penalty = (
        abs(low - 0.30) * 0.9
        + abs(mid - 0.42) * 0.7
        + abs(high - 0.28) * 0.8
    )
    score = float(np.clip(100.0 - (masking_penalty + balance_penalty) * 100.0, 0.0, 100.0))
    return score, {"low": low, "mid": mid, "high": high}


def estimate_noise_floor_db(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> float:
    channels = as_channel_matrix(audio)
    frame_rms_values: list[np.ndarray] = []

    for channel in channels:
        if len(channel) < frame_length:
            frame_rms_values.append(np.array([np.sqrt(np.mean(np.square(channel)))]))
            continue

        frame_rms_values.append(librosa.feature.rms(y=channel, frame_length=frame_length, hop_length=hop_length)[0])

    noise_floor = float(np.percentile(np.concatenate(frame_rms_values), 10))
    return db(noise_floor)


def spectral_centroid_from_power(spectrum_power: np.ndarray, freqs: np.ndarray) -> float:
    total = float(np.sum(spectrum_power))
    if total <= 1e-12:
        return 0.0
    return float(np.sum(freqs * spectrum_power) / total)


def spectral_rolloff_from_power(spectrum_power: np.ndarray, freqs: np.ndarray, roll_percent: float = 0.85) -> float:
    total = float(np.sum(spectrum_power))
    if total <= 1e-12:
        return 0.0
    cumulative = np.cumsum(spectrum_power)
    index = int(np.searchsorted(cumulative, total * roll_percent, side="left"))
    return float(freqs[min(index, len(freqs) - 1)])


def spectral_flatness_from_power(spectrum_power: np.ndarray) -> float:
    magnitude = np.sqrt(np.maximum(spectrum_power, 1e-18))
    return float(np.exp(np.mean(np.log(magnitude))) / (np.mean(magnitude) + 1e-18))


def zero_crossing_rate_channels(channels: np.ndarray) -> float:
    values = [np.mean(librosa.feature.zero_crossing_rate(y=channel)[0]) for channel in channels]
    return safe_float(np.mean(values))


def estimate_true_peak(audio: np.ndarray, oversample_factor: int = 4) -> float:
    channels = as_channel_matrix(audio)
    if channels.size == 0:
        return 0.0

    peaks = []
    for channel in channels:
        oversampled = signal.resample_poly(channel.astype(np.float64), oversample_factor, 1)
        peaks.append(float(np.max(np.abs(oversampled))))
    return max(peaks) if peaks else 0.0


def analyze_array(audio: np.ndarray, sr: int, source_info: dict[str, Any] | None = None) -> dict[str, Any]:
    source_info = source_info or {}
    channels = as_channel_matrix(audio)
    detected_channels = int(channels.shape[0])
    stereo_metrics = analyze_stereo_image(audio, sr)
    abs_audio = np.abs(channels)
    rms = float(np.sqrt(np.mean(np.square(channels))))
    peak = float(np.max(abs_audio))
    crest_db = db(peak / max(rms, 1e-9))
    lufs = estimate_integrated_loudness(channels, sr)

    spectrum = np.abs(np.fft.rfft(channels, axis=1))
    spectrum_power = np.mean(np.square(spectrum), axis=0)
    freqs = np.fft.rfftfreq(channels.shape[1], 1.0 / sr)

    low_energy = band_energy(spectrum_power, freqs, 20, 250)
    mid_energy = band_energy(spectrum_power, freqs, 250, 4000)
    high_energy = band_energy(spectrum_power, freqs, 4000, min(sr / 2, 20000))
    separation_score, band_balance = band_separation_score(low_energy, mid_energy, high_energy)

    centroid = spectral_centroid_from_power(spectrum_power, freqs)
    rolloff = spectral_rolloff_from_power(spectrum_power, freqs)
    flatness = spectral_flatness_from_power(spectrum_power)
    zcr = zero_crossing_rate_channels(channels)

    clipping_threshold = 0.98
    silence_threshold = 10 ** (-50 / 20)
    clipping_ratio = float(np.mean(abs_audio >= clipping_threshold))
    silence_ratio = float(np.mean(abs_audio <= silence_threshold))
    channel_dc = np.mean(channels, axis=1)
    dc_offset = float(np.mean(channel_dc))
    max_channel_dc_offset = float(np.max(np.abs(channel_dc)))
    noise_floor_db = estimate_noise_floor_db(channels)
    true_peak = estimate_true_peak(channels)
    dynamic_range_db = db(peak) - noise_floor_db

    quality_flags: list[str] = []
    if clipping_ratio > 0.001:
        quality_flags.append("clipping_detected")
    if max_channel_dc_offset > 0.01:
        quality_flags.append("dc_offset")
    if noise_floor_db > -45:
        quality_flags.append("high_noise_floor")
    if high_energy < 0.08 and sr >= 32000:
        quality_flags.append("dull_high_end")
    if lufs < -28:
        quality_flags.append("very_low_loudness")
    if lufs > -10:
        quality_flags.append("hot_loudness")
    if silence_ratio > 0.35:
        quality_flags.append("large_silence_sections")
    if sr < 32000:
        quality_flags.append("low_sample_rate")

    return {
        "sr": sr,
        "duration": float(channels.shape[1] / sr),
        "channels": int(source_info.get("channels", detected_channels)),
        "source_format": source_info.get("format", "UNKNOWN"),
        "source_subtype": source_info.get("subtype", "UNKNOWN"),
        "rms": rms,
        "peak": peak,
        "peak_db": db(peak),
        "true_peak": true_peak,
        "true_peak_db": db(true_peak),
        "lufs": lufs,
        "crest_db": crest_db,
        "dynamic_range_db": dynamic_range_db,
        "noise_floor_db": noise_floor_db,
        "dc_offset": dc_offset,
        "max_channel_dc_offset": max_channel_dc_offset,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": safe_float(centroid),
        "spectral_rolloff_hz": safe_float(rolloff),
        "spectral_flatness": safe_float(flatness),
        "low_energy": low_energy,
        "mid_energy": mid_energy,
        "high_energy": high_energy,
        "band_separation_score": separation_score,
        "band_balance": band_balance,
        "stereo_width": stereo_metrics["stereo_width"],
        "phase_correlation": stereo_metrics["phase_correlation"],
        "mid_lufs": stereo_metrics["mid_lufs"],
        "side_lufs": stereo_metrics["side_lufs"],
        "quality_flags": quality_flags,
    }


def analyze_stereo_image(audio: np.ndarray, sr: int | None = None) -> dict[str, float]:
    if audio.ndim == 1 or audio.shape[0] < 2:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        return {
            "stereo_width": 0.0,
            "phase_correlation": 1.0,
            "mid_lufs": estimate_integrated_loudness(audio, sr),
            "side_lufs": -180.691,
        }

    left = audio[0].astype(np.float64)
    right = audio[1].astype(np.float64)
    min_len = min(len(left), len(right))
    left = left[:min_len]
    right = right[:min_len]

    left_rms = float(np.sqrt(np.mean(np.square(left)))) + 1e-9
    right_rms = float(np.sqrt(np.mean(np.square(right)))) + 1e-9
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid_rms = float(np.sqrt(np.mean(np.square(mid)))) + 1e-9
    side_rms = float(np.sqrt(np.mean(np.square(side))))
    correlation = float(np.mean(left * right) / (left_rms * right_rms))

    return {
        "stereo_width": float(np.clip(side_rms / mid_rms, 0.0, 4.0)),
        "phase_correlation": float(np.clip(correlation, -1.0, 1.0)),
        "mid_lufs": estimate_integrated_loudness(mid, sr),
        "side_lufs": estimate_integrated_loudness(side, sr),
    }


def clamp_dsp_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "lowcut_hz": float(np.clip(float(params.get("lowcut_hz", 80)), 20, 200)),
        "low_boost_db": float(np.clip(float(params.get("low_boost_db", 0)), -6, 6)),
        "mid_cut_db": float(np.clip(float(params.get("mid_cut_db", 0)), -6, 6)),
        "high_boost_db": float(np.clip(float(params.get("high_boost_db", 0)), -6, 6)),
        "compress_ratio": float(np.clip(float(params.get("compress_ratio", 1.0)), 1.0, 4.0)),
        "target_lufs": float(np.clip(float(params.get("target_lufs", -16.0)), -24.0, -10.0)),
        "limiter_ceiling_db": float(np.clip(float(params.get("limiter_ceiling_db", -1.5)), -3.0, -0.8)),
        "exciter_amount": float(np.clip(float(params.get("exciter_amount", 0.0)), 0.0, 0.35)),
        "saturation_amount": float(np.clip(float(params.get("saturation_amount", 0.0)), 0.0, 0.35)),
        "normalize": params.get("normalize", True) is not False,
    }


def parse_dsp_params(raw_params: str | None) -> dict[str, Any] | None:
    if not raw_params:
        return None

    raw_params = raw_params.strip().lstrip("\ufeff").lstrip("ï»¿").lstrip("癤?")

    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError:
        logger.warning("[WARN] Invalid dsp_params JSON: %s", raw_params)
        return None

    safe_params = clamp_dsp_params(params)
    safe_params["limiter_ceiling_db"] = min(safe_params["limiter_ceiling_db"], MANUAL_DSP_SAFE_CEILING_DB)
    safe_params["normalize"] = True
    return safe_params


def apply_manual_dsp_tuning(base_params: dict[str, Any], tweaks: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(base_params)
    tuned["lowcut_hz"] = tuned.get("lowcut_hz", 80.0) + float(np.clip(float(tweaks.get("lowcut_offset_hz", 0.0)), -80.0, 80.0))
    tuned["low_boost_db"] = tuned.get("low_boost_db", 0.0) + float(np.clip(float(tweaks.get("low_boost_delta_db", 0.0)), -6.0, 6.0))
    tuned["mid_cut_db"] = tuned.get("mid_cut_db", 0.0) + float(np.clip(float(tweaks.get("mid_delta_db", 0.0)), -6.0, 6.0))
    tuned["high_boost_db"] = tuned.get("high_boost_db", 0.0) + float(np.clip(float(tweaks.get("high_boost_delta_db", 0.0)), -6.0, 6.0))
    tuned["compress_ratio"] = tuned.get("compress_ratio", 1.0) + float(np.clip(float(tweaks.get("compress_delta", 0.0)), -1.5, 2.0))
    tuned["target_lufs"] = tuned.get("target_lufs", -16.0) + float(np.clip(float(tweaks.get("target_lufs_delta", 0.0)), -8.0, 8.0))
    tuned["exciter_amount"] = tuned.get("exciter_amount", 0.0) + float(np.clip(float(tweaks.get("exciter_delta", 0.0)), -0.35, 0.35))
    tuned["saturation_amount"] = tuned.get("saturation_amount", 0.0) + float(np.clip(float(tweaks.get("saturation_delta", 0.0)), -0.35, 0.35))
    tuned["limiter_ceiling_db"] = min(
        float(tweaks.get("limiter_ceiling_db", tuned.get("limiter_ceiling_db", -1.5))),
        MANUAL_DSP_SAFE_CEILING_DB,
    )
    tuned["normalize"] = True

    safe_params = clamp_dsp_params(tuned)
    safe_params["limiter_ceiling_db"] = min(safe_params["limiter_ceiling_db"], MANUAL_DSP_SAFE_CEILING_DB)
    safe_params["normalize"] = True
    return safe_params


def parse_manual_dsp_request(raw_params: str | None, base_params: dict[str, Any]) -> dict[str, Any] | None:
    if not raw_params:
        return None

    raw_params = raw_params.strip().lstrip("\ufeff").lstrip("챦쨩쩔").lstrip("??")
    try:
        params = json.loads(raw_params)
    except json.JSONDecodeError:
        logger.warning("[WARN] Invalid dsp_params JSON: %s", raw_params)
        return None

    delta_keys = {
        "lowcut_offset_hz",
        "low_boost_delta_db",
        "mid_delta_db",
        "high_boost_delta_db",
        "compress_delta",
        "target_lufs_delta",
        "exciter_delta",
        "saturation_delta",
    }
    if params.get("manual_mode") == "fine_tune" or any(key in params for key in delta_keys):
        return apply_manual_dsp_tuning(base_params, params)

    return parse_dsp_params(raw_params)


def parse_processing_targets(target: str) -> list[str]:
    legacy_targets = {
        "auto": "hifi_clean",
        "music": "hifi_clean",
        "natural": "hifi_clean",
        "archive": "restore",
        "loud": "loud_modern",
        "bass": "bass_boost",
        "low": "bass_boost",
        "voice": "voice_focus",
    }
    raw_targets = [item.strip() for item in (target or "hifi_clean").split(",")]
    resolved: list[str] = []
    for item in raw_targets:
        mapped = legacy_targets.get(item, item)
        if mapped in PROCESSING_TARGETS and mapped not in resolved:
            resolved.append(mapped)
    return resolved or ["hifi_clean"]


def blend_target_profiles(targets: list[str]) -> dict[str, Any]:
    blended: dict[str, Any] = {}
    keys = TARGET_PROFILES["hifi_clean"].keys()
    for key in keys:
        blended[key] = float(np.mean([TARGET_PROFILES[target][key] for target in targets]))
    blended["normalize"] = True
    return blended


def scale_dsp_params_for_ai_amount(
    params: dict[str, Any],
    analysis: dict[str, Any],
    amount: float,
) -> dict[str, Any]:
    amount = float(np.clip(amount, 0.0, 1.0))
    source_lufs = float(np.clip(analysis.get("lufs", params["target_lufs"]), -24.0, -10.0))
    scaled = params.copy()
    scaled["lowcut_hz"] = 20.0 + (params["lowcut_hz"] - 20.0) * amount
    scaled["low_boost_db"] = params["low_boost_db"] * amount
    scaled["mid_cut_db"] = params["mid_cut_db"] * amount
    scaled["high_boost_db"] = params["high_boost_db"] * amount
    scaled["compress_ratio"] = 1.0 + (params["compress_ratio"] - 1.0) * amount
    scaled["target_lufs"] = source_lufs + (params["target_lufs"] - source_lufs) * amount
    scaled["exciter_amount"] = params["exciter_amount"] * amount
    scaled["saturation_amount"] = params["saturation_amount"] * amount
    scaled["normalize"] = params.get("normalize", True)
    return clamp_dsp_params(scaled)


def recommend_processing(
    analysis: dict[str, Any],
    target: str = "hifi_clean",
    user_intensity: float | None = None,
) -> dict[str, Any]:
    flags = set(analysis.get("quality_flags", []))
    targets = parse_processing_targets(target)
    target = targets[0]
    has_bass_boost = "bass_boost" in targets
    ai_amount = 0.6 if user_intensity is None else float(np.clip(user_intensity, 0.0, 1.0))
    intensity = 0.45
    reasons: list[str] = []
    params: dict[str, Any] = blend_target_profiles(targets)

    mode = "denoise"
    if len(targets) > 1:
        reasons.append("Blended listening targets: " + ", ".join(targets) + ".")

    if target == "restore":
        intensity = 0.65
        reasons.append("Restore target prioritizes cleanup and source preservation.")
    elif target == "hifi_clean":
        intensity = 0.55
        reasons.append("Hi-Fi Clean target keeps a balanced, low-fatigue sound.")
    elif target == "hifi_bright":
        intensity = 0.55
        reasons.append("Hi-Fi Bright target adds presence and upper detail.")
    elif target == "warm_analog":
        intensity = 0.45
        reasons.append("Warm Analog target adds body and smooths the top end.")
    elif target == "loud_modern":
        intensity = 0.6
        reasons.append("Loud Modern target increases density and perceived level.")
    elif target == "bass_boost":
        intensity = 0.55
        reasons.append("Bass Boost target adds low-end weight while protecting headroom.")
    elif target == "voice_focus":
        intensity = 0.75
        reasons.append("Voice Focus target emphasizes speech clarity.")

    if analysis["noise_floor_db"] > -45:
        intensity = max(intensity, 0.65)
        reasons.append("Raised denoise intensity because the estimated noise floor is high.")
    elif target in {"hifi_clean", "warm_analog"}:
        intensity = min(intensity, 0.55)
        reasons.append("Kept denoise intensity controlled to preserve natural texture.")

    if analysis["low_energy"] < 0.18 and target != "voice_focus":
        params["low_boost_db"] = max(params["low_boost_db"], 2.0)
        reasons.append("Added low-band support because bass energy is weak.")
    elif analysis["low_energy"] > 0.45 and has_bass_boost:
        params["lowcut_hz"] = max(params["lowcut_hz"], 38)
        params["low_boost_db"] = float(np.clip(params["low_boost_db"], 0.8, 1.8))
        reasons.append("Controlled bass boost because the source already has strong low energy.")
    elif analysis["low_energy"] > 0.45 and target == "restore":
        params["lowcut_hz"] = max(params["lowcut_hz"], 35)
        params["low_boost_db"] = min(params["low_boost_db"], 0.0)
        reasons.append("Preserved low-end weight while avoiding rumble in restore mode.")
    elif analysis["low_energy"] > 0.45 or target == "voice_focus":
        params["lowcut_hz"] = max(params["lowcut_hz"], 100 if target == "voice_focus" else 45)
        params["low_boost_db"] = min(params["low_boost_db"], -1.0)
        reasons.append("Reduced low-band buildup to improve clarity.")

    if analysis["mid_energy"] > 0.72:
        params["mid_cut_db"] = min(params["mid_cut_db"], -1.5)
        reasons.append("Applied a small mid cut because the mix is mid-heavy.")

    if analysis["high_energy"] < 0.12 and "low_sample_rate" not in flags:
        params["high_boost_db"] = max(params["high_boost_db"], 2.0)
        reasons.append("Added high-band lift because the source sounds dull.")
    elif analysis["spectral_flatness"] > 0.08 and target != "hifi_bright":
        params["high_boost_db"] = min(params["high_boost_db"], -1.0)
        reasons.append("Reduced high band slightly because the source appears noisy.")

    if analysis["lufs"] < -28:
        params["compress_ratio"] = max(params["compress_ratio"], 2.2)
        reasons.append("Added compression because loudness is very low.")
    elif analysis["crest_db"] > 18:
        params["compress_ratio"] = max(params["compress_ratio"], 1.8)
        reasons.append("Added gentle compression because dynamic range is wide.")

    if "clipping_detected" in flags:
        params["compress_ratio"] = min(max(params["compress_ratio"], 1.4), 2.2)
        reasons.append("Detected clipping; avoided aggressive gain changes.")

    if analysis["sr"] < 32000:
        mode = "upsample"
        reasons.append("Selected model-assisted upsample mode for low sample-rate material.")

    output_sr = choose_output_sample_rate(int(analysis["sr"])) if mode == "upsample" else int(analysis["sr"])

    stereo_safe = int(analysis.get("channels", 1)) >= 2
    if stereo_safe:
        reasons.append("Stereo-safe mid/side processing will preserve staging cues.")
        if analysis.get("phase_correlation", 1.0) < 0.2:
            intensity = min(intensity, 0.55)
            params["exciter_amount"] = min(params["exciter_amount"], 0.1)
            params["saturation_amount"] = min(params["saturation_amount"], 0.08)
            reasons.append("Reduced enhancement on phase-sensitive stereo material.")

    if len(targets) == 1:
        if target == "restore":
            params["high_boost_db"] = min(params["high_boost_db"], 1.2)
            params["compress_ratio"] = min(params["compress_ratio"], 1.8)
        elif target == "hifi_clean":
            params["high_boost_db"] = float(np.clip(params["high_boost_db"], 0.4, 1.8))
            params["compress_ratio"] = min(params["compress_ratio"], 1.8)
        elif target == "hifi_bright":
            params["high_boost_db"] = max(params["high_boost_db"], 2.2)
            params["mid_cut_db"] = min(params["mid_cut_db"], -0.4)
        elif target == "warm_analog":
            params["low_boost_db"] = max(params["low_boost_db"], 0.8)
            params["high_boost_db"] = min(params["high_boost_db"], -0.4)
            params["compress_ratio"] = float(np.clip(params["compress_ratio"], 1.25, 1.8))
        elif target == "loud_modern":
            params["compress_ratio"] = max(params["compress_ratio"], 2.6)
            params["high_boost_db"] = max(params["high_boost_db"], 1.0)
        elif target == "bass_boost":
            params["lowcut_hz"] = min(params["lowcut_hz"], 38)
            params["low_boost_db"] = float(np.clip(params["low_boost_db"], 1.4, 3.2))
            params["compress_ratio"] = float(np.clip(params["compress_ratio"], 1.35, 2.1))
            params["high_boost_db"] = float(np.clip(params["high_boost_db"], -0.2, 0.8))
        elif target == "voice_focus":
            params["lowcut_hz"] = max(params["lowcut_hz"], 100)
            params["low_boost_db"] = min(params["low_boost_db"], -1.0)
            params["high_boost_db"] = max(params["high_boost_db"], 1.4)

    confidence = 0.72
    if flags:
        confidence -= min(len(flags) * 0.04, 0.18)
    if analysis["duration"] < 1.0:
        confidence -= 0.12
        reasons.append("Short files provide less reliable analysis.")

    intensity = float(np.clip(intensity * ai_amount, 0.0, 1.0))
    params = scale_dsp_params_for_ai_amount(clamp_dsp_params(params), analysis, ai_amount)
    model_denoise = (
        ENABLE_GTCRN_MODEL
        and
        mode == "denoise"
        and analysis.get("duration", 0.0) <= GTCRN_MAX_MODEL_SECONDS
        and (target in {"restore", "voice_focus"} or "high_noise_floor" in flags)
    )
    if mode == "denoise" and not model_denoise:
        reasons.append("Used fast denoise path to keep full-track processing responsive.")
    if not reasons:
        reasons.append("The source is already balanced; applied minimal cleanup.")

    return {
        "target": "+".join(targets),
        "targets": targets,
        "mode": mode,
        "intensity": float(np.clip(intensity, 0.0, 1.0)),
        "ai_amount": ai_amount,
        "dsp_params": params,
        "stereo_safe": stereo_safe,
        "output_sr": output_sr,
        "model_denoise": model_denoise,
        "confidence": float(np.clip(confidence, 0.1, 0.95)),
        "reasons": reasons,
        "advice": summarize_recommendation(params, mode, intensity, flags),
    }


def sync_recommendation_output_sr(recommendation: dict[str, Any], source_sr: int) -> None:
    recommendation["output_sr"] = choose_output_sample_rate(source_sr) if recommendation["mode"] == "upsample" else source_sr


def summarize_recommendation(
    params: dict[str, Any],
    mode: str,
    intensity: float,
    flags: set[str],
) -> str:
    parts = [f"{mode} mode", f"denoise intensity {intensity:.2f}"]
    if params["low_boost_db"]:
        parts.append(f"low {params['low_boost_db']:+.1f} dB")
    if params["mid_cut_db"]:
        parts.append(f"mid {params['mid_cut_db']:+.1f} dB")
    if params["high_boost_db"]:
        parts.append(f"high {params['high_boost_db']:+.1f} dB")
    if params["compress_ratio"] > 1.01:
        parts.append(f"{params['compress_ratio']:.1f}:1 compression")
    parts.append(f"target {params['target_lufs']:.1f} LUFS")
    if params["exciter_amount"] > 0.01:
        parts.append(f"exciter {params['exciter_amount']:.2f}")
    if params["saturation_amount"] > 0.01:
        parts.append(f"saturation {params['saturation_amount']:.2f}")
    if flags:
        parts.append("flags: " + ", ".join(sorted(flags)))
    return "; ".join(parts)


def apply_soft_highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    cutoff_hz = float(cutoff_hz)
    if cutoff_hz <= 20.0 or sr <= 0 or len(audio) < 8:
        return audio

    cutoff_hz = float(np.clip(cutoff_hz, 20.0, sr * 0.45))
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    try:
        return signal.sosfiltfilt(sos, audio).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, audio).astype(np.float32)


def smoothstep_weight(freqs: np.ndarray, start_hz: float, end_hz: float) -> np.ndarray:
    if end_hz <= start_hz:
        return (freqs >= end_hz).astype(np.float64)
    x = np.clip((freqs - start_hz) / (end_hz - start_hz), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def apply_frequency_shaping(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    filtered = apply_soft_highpass(audio, sr, params["lowcut_hz"])
    shaped = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(filtered), 1.0 / sr)

    low_weight = 1.0 - smoothstep_weight(freqs, 180.0, 360.0)
    high_weight = smoothstep_weight(freqs, 3200.0, 6200.0)
    mid_weight = np.clip(1.0 - low_weight - high_weight, 0.0, 1.0)
    total_weight = np.maximum(low_weight + mid_weight + high_weight, 1e-9)
    gain_db = (
        params["low_boost_db"] * low_weight
        + params["mid_cut_db"] * mid_weight
        + params["high_boost_db"] * high_weight
    ) / total_weight
    gain = np.power(10.0, gain_db / 20.0)

    return np.fft.irfft(shaped * gain, n=len(filtered)).astype(np.float32)


def spectral_band_profile(audio: np.ndarray, sr: int) -> dict[str, float]:
    channels = as_channel_matrix(audio)
    if channels.size == 0:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}
    spectrum = np.abs(np.fft.rfft(channels, axis=1))
    spectrum_power = np.mean(np.square(spectrum), axis=0)
    freqs = np.fft.rfftfreq(channels.shape[1], 1.0 / sr)
    low = band_energy(spectrum_power, freqs, 20.0, 250.0)
    mid = band_energy(spectrum_power, freqs, 250.0, 4000.0)
    high = band_energy(spectrum_power, freqs, 4000.0, min(sr / 2, 20000.0))
    _, balance = band_separation_score(low, mid, high)
    return balance


def apply_band_balance_guard(source_audio: np.ndarray, processed_audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    source = spectral_band_profile(source_audio, sr)
    processed = spectral_band_profile(processed_audio, sr)
    drift = max(
        abs(processed["low"] - source["low"]),
        abs(processed["mid"] - source["mid"]),
        abs(processed["high"] - source["high"]),
    )
    if drift < 0.12:
        return processed_audio.astype(np.float32), 0.0

    low_db = float(np.clip((source["low"] - processed["low"]) * 7.0, -2.0, 2.0))
    mid_db = float(np.clip((source["mid"] - processed["mid"]) * 5.0, -1.5, 1.5))
    high_db = float(np.clip((source["high"] - processed["high"]) * 6.0, -2.0, 2.0))
    strength = float(np.clip((drift - 0.08) / 0.20, 0.0, 0.65))
    correction_params = {
        "lowcut_hz": 20.0,
        "low_boost_db": low_db * strength,
        "mid_cut_db": mid_db * strength,
        "high_boost_db": high_db * strength,
    }

    if processed_audio.ndim == 1:
        guarded = apply_frequency_shaping(processed_audio, sr, correction_params)
    else:
        guarded = np.stack(
            [apply_frequency_shaping(channel, sr, correction_params) for channel in processed_audio],
            axis=0,
        )
    applied_db = max(
        abs(correction_params["low_boost_db"]),
        abs(correction_params["mid_cut_db"]),
        abs(correction_params["high_boost_db"]),
    )
    return guarded.astype(np.float32), float(applied_db)


def crest_factor_db(audio: np.ndarray) -> float:
    channels = as_channel_matrix(audio)
    if channels.size == 0:
        return 0.0
    peak = float(np.max(np.abs(channels)))
    level = rms(channels)
    return db(peak / max(level, 1e-9))


def transient_mask(audio: np.ndarray, sr: int) -> np.ndarray:
    channels = as_channel_matrix(audio)
    if channels.size == 0:
        return np.zeros(0, dtype=np.float32)

    mono = np.mean(channels, axis=0).astype(np.float32)
    focused = np.abs(apply_soft_highpass(mono, sr, 120.0))
    if focused.size == 0 or float(np.max(focused)) <= 1e-9:
        return np.zeros_like(focused, dtype=np.float32)

    fast_coeff = float(np.exp(-1.0 / max(sr * 0.003, 1.0)))
    slow_coeff = float(np.exp(-1.0 / max(sr * 0.045, 1.0)))
    fast = signal.lfilter([1.0 - fast_coeff], [1.0, -fast_coeff], focused)
    slow = signal.lfilter([1.0 - slow_coeff], [1.0, -slow_coeff], focused)
    onset = np.maximum(fast - slow, 0.0)
    scale = float(np.percentile(onset, 98)) + 1e-9
    return np.power(np.clip(onset / scale, 0.0, 1.0), 0.65).astype(np.float32)


def apply_transient_preservation(
    source_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, float]:
    source_crest = crest_factor_db(source_audio)
    processed_crest = crest_factor_db(processed_audio)
    crest_loss_db = source_crest - processed_crest
    if crest_loss_db <= 1.2:
        return processed_audio.astype(np.float32), 0.0

    source_channels = as_channel_matrix(source_audio)
    processed_channels = as_channel_matrix(processed_audio)
    channel_count = min(source_channels.shape[0], processed_channels.shape[0])
    sample_count = min(source_channels.shape[1], processed_channels.shape[1])
    if channel_count <= 0 or sample_count <= 0:
        return processed_audio.astype(np.float32), 0.0

    mask = transient_mask(source_channels[:channel_count, :sample_count], sr)
    if mask.size == 0 or float(np.max(mask)) <= 0.01:
        return processed_audio.astype(np.float32), 0.0

    strength = float(np.clip((crest_loss_db - 0.8) / 10.0, 0.0, 0.16))
    restored = processed_channels.copy()
    for index in range(channel_count):
        source_transient = apply_soft_highpass(source_channels[index, :sample_count], sr, 120.0)
        processed_transient = apply_soft_highpass(processed_channels[index, :sample_count], sr, 120.0)
        residual = source_transient - processed_transient
        restored[index, :sample_count] = restored[index, :sample_count] + residual * mask * strength

    if processed_audio.ndim == 1:
        return restored[0].astype(np.float32), crest_loss_db
    return restored.astype(np.float32), crest_loss_db


def apply_soft_compression(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    if ratio <= 1.01:
        return audio

    threshold_db = -15.0
    knee_db = 7.0
    attack_seconds = 0.018
    release_seconds = 0.16
    block_size = max(32, int(sr * 0.004))
    block_count = int(np.ceil(len(audio) / block_size))
    if block_count <= 1:
        return audio.astype(np.float32)

    block_gain_db = np.zeros(block_count, dtype=np.float64)
    for block_index in range(block_count):
        start = block_index * block_size
        end = min(len(audio), start + block_size)
        block = audio[start:end]
        level = max(
            float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0,
            float(np.max(np.abs(block))) * 0.55 if block.size else 0.0,
            1e-9,
        )
        level_db = db(level)
        over_db = level_db - threshold_db
        if over_db <= -knee_db * 0.5:
            gain_reduction_db = 0.0
        elif over_db >= knee_db * 0.5:
            gain_reduction_db = over_db * (1.0 - 1.0 / ratio)
        else:
            knee_pos = over_db + knee_db * 0.5
            gain_reduction_db = (1.0 - 1.0 / ratio) * (knee_pos * knee_pos) / (2.0 * knee_db)
        block_gain_db[block_index] = -min(gain_reduction_db, 7.0)

    attack_coeff = float(np.exp(-block_size / max(sr * attack_seconds, 1.0)))
    release_coeff = float(np.exp(-block_size / max(sr * release_seconds, 1.0)))
    smoothed_gain_db = np.empty_like(block_gain_db)
    current_gain_db = 0.0
    for index, target_gain_db in enumerate(block_gain_db):
        coeff = attack_coeff if target_gain_db < current_gain_db else release_coeff
        current_gain_db = target_gain_db + (current_gain_db - target_gain_db) * coeff
        smoothed_gain_db[index] = current_gain_db

    block_positions = np.arange(block_count, dtype=np.float64) * block_size
    sample_positions = np.arange(len(audio), dtype=np.float64)
    gain_db = np.interp(sample_positions, block_positions, smoothed_gain_db)
    gain = np.power(10.0, gain_db / 20.0)
    return (audio * gain).astype(np.float32)


def apply_spectral_gate(audio: np.ndarray, intensity: float) -> np.ndarray:
    intensity = float(np.clip(intensity, 0.0, 1.0))
    if intensity <= 0.0:
        return audio

    n_fft = min(2048, max(512, int(2 ** np.floor(np.log2(max(len(audio), 512))))))
    hop_length = max(128, n_fft // 4)
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = np.abs(stft), np.angle(stft)
    noise_floor = np.percentile(magnitude, 20, axis=1, keepdims=True)
    threshold = noise_floor * (1.0 + intensity * 2.5)
    mask = magnitude >= threshold
    softened = magnitude * (mask + (1.0 - mask) * (1.0 - intensity * 0.75))
    restored = librosa.istft(softened * np.exp(1j * phase), hop_length=hop_length, length=len(audio))
    return restored.astype(np.float32)


def apply_reference_bleed_cleanup(
    target_audio: np.ndarray,
    reference_audio: np.ndarray,
    sr: int,
    strength: float = STEM_VOCAL_BLEED_CLEANUP_STRENGTH,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 0.7))
    if strength <= 0.0 or sr <= 0:
        return target_audio.astype(np.float32)

    target = np.asarray(target_audio, dtype=np.float32)
    reference = np.asarray(reference_audio, dtype=np.float32)
    if target.size == 0 or reference.size == 0:
        return target.astype(np.float32)

    n_fft = min(4096, max(1024, int(2 ** np.floor(np.log2(max(len(target), 1024))))))
    hop_length = max(256, n_fft // 4)
    target_stft = librosa.stft(target, n_fft=n_fft, hop_length=hop_length)
    reference_stft = librosa.stft(reference, n_fft=n_fft, hop_length=hop_length)

    target_mag = np.abs(target_stft)
    reference_mag = np.abs(reference_stft)
    phase = np.exp(1j * np.angle(target_stft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)[:, np.newaxis]

    high_weight = np.clip((freqs - 3500.0) / 4500.0, 0.0, 1.0)
    low_weight = np.clip((160.0 - freqs) / 120.0, 0.0, 1.0) * 0.45
    mid_weight = np.where((freqs >= 500.0) & (freqs <= 3200.0), 0.12, 0.0)
    band_weight = np.clip(high_weight + low_weight + mid_weight, 0.0, 1.0)

    reference_ratio = reference_mag / (target_mag + reference_mag + 1e-9)
    bleed_mask = np.clip((reference_ratio - 0.52) / 0.36, 0.0, 1.0)
    reduction = np.clip(strength * band_weight * bleed_mask, 0.0, 0.42)
    cleaned_mag = target_mag * (1.0 - reduction)

    restored = librosa.istft(cleaned_mag * phase, hop_length=hop_length, length=len(target))
    return restored.astype(np.float32)


def apply_vocal_stem_bleed_cleanup(
    vocal_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    vocal_sr: int,
    instrumental_sr: int,
) -> tuple[np.ndarray, bool]:
    vocal_matrix = as_channel_matrix(vocal_audio)
    reference = instrumental_audio
    if instrumental_sr != vocal_sr:
        reference = resample_audio(reference, instrumental_sr, vocal_sr)
    reference_matrix = as_fixed_channel_matrix(reference, vocal_matrix.shape[0])
    reference_matrix = fit_audio_length(reference_matrix, vocal_matrix.shape[-1])

    cleaned_channels = []
    for index, channel in enumerate(vocal_matrix):
        cleaned = apply_reference_bleed_cleanup(
            channel,
            reference_matrix[index],
            vocal_sr,
            STEM_VOCAL_BLEED_CLEANUP_STRENGTH,
        )
        cleaned_channels.append(cleaned)

    cleaned_audio = stack_channels(cleaned_channels, vocal_audio.ndim == 1)
    cleanup_delta = float(np.mean(np.abs(vocal_matrix - as_channel_matrix(cleaned_audio))))
    return cleaned_audio.astype(np.float32), cleanup_delta > 1e-7


def apply_vocal_sibilance_guard(
    source_audio: np.ndarray,
    processed_audio: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, float]:
    source_channels = as_channel_matrix(source_audio)
    processed_channels = as_channel_matrix(processed_audio)
    channel_count = min(source_channels.shape[0], processed_channels.shape[0])
    sample_count = min(source_channels.shape[-1], processed_channels.shape[-1])
    if channel_count <= 0 or sample_count <= 0 or sr <= 0:
        return processed_audio.astype(np.float32), 0.0

    guarded = processed_channels.copy()
    max_reduction_db = 0.0
    for index in range(channel_count):
        source_band = bandpass_component(source_channels[index, :sample_count], sr, 5200.0, min(10500.0, sr * 0.45))
        processed_band = bandpass_component(processed_channels[index, :sample_count], sr, 5200.0, min(10500.0, sr * 0.45))
        source_ratio = rms(source_band) / (rms(source_channels[index, :sample_count]) + 1e-9)
        processed_ratio = rms(processed_band) / (rms(processed_channels[index, :sample_count]) + 1e-9)
        allowed_ratio = max(source_ratio * 1.35, 0.10)
        if processed_ratio <= allowed_ratio or processed_ratio <= 1e-9:
            continue

        reduction = float(np.clip(allowed_ratio / processed_ratio, 0.50, 1.0))
        guarded[index, :sample_count] = guarded[index, :sample_count] - processed_band * (1.0 - reduction)
        max_reduction_db = min(max_reduction_db, db(reduction))

    if processed_audio.ndim == 1:
        return guarded[0].astype(np.float32), max_reduction_db
    return guarded.astype(np.float32), max_reduction_db


def apply_stem_specific_guard(
    source_audio: np.ndarray,
    source_sr: int,
    processed_audio: np.ndarray,
    sr: int,
    stem_name: str,
) -> tuple[np.ndarray, list[str]]:
    if source_sr != sr:
        source_audio = resample_audio(source_audio, source_sr, sr)
    guarded = processed_audio.astype(np.float32)
    steps: list[str] = []

    if stem_name == "vocals":
        guarded, sibilance_reduction_db = apply_vocal_sibilance_guard(source_audio, guarded, sr)
        steps.append(
            f"vocal_sibilance_guard_{sibilance_reduction_db:+.1f}db"
            if sibilance_reduction_db < -0.05
            else "vocal_sibilance_checked"
        )
    elif stem_name == "drums":
        guarded, transient_loss_db = apply_transient_preservation(source_audio, guarded, sr)
        steps.append(f"drum_transient_guard_{transient_loss_db:.1f}db")
    elif stem_name == "bass":
        guarded, side_gain_db = apply_low_bass_phase_guard(source_audio, guarded, sr)
        steps.append(f"bass_low_phase_guard_{side_gain_db:+.1f}db" if side_gain_db < -0.05 else "bass_low_phase_checked")
        guarded, band_guard_db = apply_band_balance_guard(source_audio, guarded, sr)
        if band_guard_db >= 0.05:
            steps.append(f"bass_band_guard_{band_guard_db:.1f}db")
    elif stem_name == "other":
        guarded = apply_saturation_smoothing(guarded, sr, 0.08)
        steps.append("other_artifact_smoothing")

    return guarded.astype(np.float32), steps


def apply_stem_auto_gain_balance(
    source_audio: np.ndarray,
    source_sr: int,
    processed_audio: np.ndarray,
    processed_sr: int,
    stem_name: str,
) -> tuple[np.ndarray, float, bool]:
    source_reference = source_audio
    if source_sr != processed_sr:
        source_reference = resample_audio(source_reference, source_sr, processed_sr)
    source_matrix = as_fixed_channel_matrix(source_reference, as_channel_matrix(processed_audio).shape[0])
    source_reference = fit_audio_length(source_matrix, as_channel_matrix(processed_audio).shape[-1])
    if processed_audio.ndim == 1:
        source_reference = source_reference[0]

    return apply_safe_source_loudness_match(
        processed_audio,
        source_reference,
        processed_sr,
        STEM_GAIN_MATCH_CEILING_DB,
        max_boost_db=float(STEM_GAIN_MATCH_MAX_BOOST_DB.get(stem_name, 1.8)),
    )


def run_onnx_in_chunks(audio: np.ndarray, sr: int) -> np.ndarray:
    if upsampler_session is None:
        return audio

    input_name = upsampler_session.get_inputs()[0].name
    chunk_size = sr * 10
    chunks = []

    for index in range(0, len(audio), chunk_size):
        chunk = audio[index : index + chunk_size]
        try:
            chunk_tensor = np.expand_dims(chunk, axis=0).astype(np.float32)
            output = upsampler_session.run(None, {input_name: chunk_tensor})[0].flatten()
            chunks.append(output.astype(np.float32))
        except Exception:
            logger.exception("[ERROR] ONNX chunk processing failed. Falling back to source chunk.")
            chunks.append(chunk)

    return np.concatenate(chunks).astype(np.float32)


def run_gtcrn_denoise(audio: np.ndarray, sr: int, intensity: float) -> tuple[np.ndarray, bool]:
    if gtcrn_session is None:
        return audio, False

    model_sr = 16000
    work_audio = audio
    if sr != model_sr:
        work_audio = librosa.resample(work_audio, orig_sr=sr, target_sr=model_sr).astype(np.float32)

    n_fft = 512
    hop_length = 256
    stft = librosa.stft(work_audio, n_fft=n_fft, hop_length=hop_length, center=True)
    enhanced_frames = np.empty_like(stft)

    inputs = gtcrn_session.get_inputs()
    mix_name = inputs[0].name
    caches = {
        item.name: np.zeros([dim if isinstance(dim, int) else 1 for dim in item.shape], dtype=np.float32)
        for item in inputs[1:]
    }

    for frame_index in range(stft.shape[1]):
        frame = stft[:, frame_index]
        mix = np.stack([frame.real, frame.imag], axis=-1).astype(np.float32)
        feed = {mix_name: mix.reshape(1, n_fft // 2 + 1, 1, 2)}
        feed.update(caches)

        outputs = gtcrn_session.run(None, feed)
        enhanced = outputs[0].reshape(n_fft // 2 + 1, 2)
        enhanced_frames[:, frame_index] = enhanced[:, 0] + 1j * enhanced[:, 1]

        for output_meta, output_value in zip(gtcrn_session.get_outputs()[1:], outputs[1:]):
            cache_name = output_meta.name.removesuffix("_out")
            if cache_name in caches:
                caches[cache_name] = output_value.astype(np.float32)

    denoised = librosa.istft(enhanced_frames, hop_length=hop_length, length=len(work_audio), center=True)
    denoised = np.nan_to_num(denoised, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if sr != model_sr:
        denoised = librosa.resample(denoised, orig_sr=model_sr, target_sr=sr).astype(np.float32)
        if len(denoised) != len(audio):
            denoised = librosa.util.fix_length(denoised, size=len(audio))

    blend = float(np.clip(intensity, 0.0, 1.0))
    processed = (audio * (1.0 - blend) + denoised[: len(audio)] * blend).astype(np.float32)
    return processed, True


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-9:
        return audio
    return (audio / peak * 0.98).astype(np.float32)


def apply_saturation(audio: np.ndarray, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.35))
    if amount <= 0.0:
        return audio

    drive = 1.0 + amount * 5.0
    saturated = np.tanh(audio * drive) / np.tanh(drive)
    return (audio * (1.0 - amount) + saturated * amount).astype(np.float32)


def apply_saturation_smoothing(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.35))
    if amount <= 0.03 or len(audio) < 32 or sr <= 0:
        return audio.astype(np.float32)

    cutoff = min(20000.0, sr * 0.46)
    if cutoff >= sr * 0.49:
        return audio.astype(np.float32)

    sos = signal.butter(2, cutoff, btype="lowpass", fs=sr, output="sos")
    try:
        smoothed = signal.sosfiltfilt(sos, audio).astype(np.float32)
    except ValueError:
        smoothed = signal.sosfilt(sos, audio).astype(np.float32)
    blend = float(np.clip(amount * 0.65, 0.0, 0.18))
    return (audio * (1.0 - blend) + smoothed * blend).astype(np.float32)


def apply_harmonic_exciter(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.35))
    if amount <= 0.0:
        return audio

    if len(audio) < 16 or sr <= 0:
        return audio.astype(np.float32)

    low_hz = min(3800.0, sr * 0.35)
    high_hz = min(12000.0, sr * 0.45)
    if high_hz <= low_hz + 200:
        return audio.astype(np.float32)

    sos = signal.butter(2, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    try:
        air_band = signal.sosfiltfilt(sos, audio).astype(np.float32)
    except ValueError:
        air_band = signal.sosfilt(sos, audio).astype(np.float32)

    excited = np.tanh(air_band * 3.0) - air_band * 0.35
    bright_rms = float(np.sqrt(np.mean(np.square(air_band)))) + 1e-9
    full_rms = float(np.sqrt(np.mean(np.square(audio)))) + 1e-9
    brightness_ratio = bright_rms / full_rms
    de_ess = float(np.clip(0.22 / max(brightness_ratio, 1e-9), 0.45, 1.0))
    return (audio + excited * amount * 0.55 * de_ess).astype(np.float32)


def apply_loudness_normalize(
    audio: np.ndarray,
    target_lufs: float,
    limiter_ceiling_db: float,
    max_limiter_drive_db: float = 1.5,
    sr: int | None = None,
) -> tuple[np.ndarray, float, bool]:
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 1e-9:
        return audio, 0.0, False

    current_lufs = estimate_integrated_loudness(audio, sr)
    desired_gain_db = float(np.clip(target_lufs - current_lufs, -18.0, 18.0))
    true_peak_db = db(estimate_true_peak(audio, oversample_factor=2))
    max_gain_db = (limiter_ceiling_db + max_limiter_drive_db) - true_peak_db
    gain_db = min(desired_gain_db, max_gain_db)
    gain = 10 ** (gain_db / 20.0)
    return (audio * gain).astype(np.float32), gain_db, gain_db < desired_gain_db - 0.05


def apply_soft_limiter(audio: np.ndarray, ceiling_db: float, sr: int) -> np.ndarray:
    ceiling = 10 ** (ceiling_db / 20.0)
    if ceiling <= 0.0:
        return audio

    channels = as_channel_matrix(audio).astype(np.float64)
    linked_peak = np.max(np.abs(channels), axis=0)
    if float(np.max(linked_peak)) <= ceiling:
        return audio.astype(np.float32)

    desired_gain = np.minimum(1.0, ceiling / (linked_peak + 1e-12))
    release_seconds = 0.08
    release_coeff = float(np.exp(-1.0 / max(sr * release_seconds, 1.0)))
    gain = np.empty_like(desired_gain)
    current_gain = 1.0

    for index, target_gain in enumerate(desired_gain):
        if target_gain < current_gain:
            current_gain = float(target_gain)
        else:
            current_gain = float(target_gain + (current_gain - target_gain) * release_coeff)
        gain[index] = current_gain

    limited = channels * gain[np.newaxis, :]
    limited = np.clip(limited, -ceiling, ceiling)
    if audio.ndim == 1:
        return limited[0].astype(np.float32)
    return limited.astype(np.float32)


def apply_true_peak_headroom(
    audio: np.ndarray,
    ceiling_db: float,
    oversample_factor: int = 4,
) -> tuple[np.ndarray, float, float]:
    ceiling = 10 ** (ceiling_db / 20.0)
    true_peak = estimate_true_peak(audio, oversample_factor)
    true_peak_db = db(true_peak)
    if true_peak <= ceiling or true_peak <= 1e-9:
        return audio, 0.0, true_peak_db

    gain = ceiling / true_peak
    gain_db = db(gain)
    return (audio * gain).astype(np.float32), gain_db, true_peak_db


def estimate_lufs_like(audio: np.ndarray, sr: int | None = None) -> float:
    return estimate_integrated_loudness(audio, sr)


def apply_safe_source_loudness_match(
    audio: np.ndarray,
    source_audio: np.ndarray,
    sr: int,
    ceiling_db: float,
    max_boost_db: float = 8.0,
) -> tuple[np.ndarray, float, bool]:
    source_lufs = estimate_lufs_like(source_audio, sr)
    current_lufs = estimate_lufs_like(audio, sr)
    if not np.isfinite(source_lufs) or not np.isfinite(current_lufs):
        return audio.astype(np.float32), 0.0, False

    desired_gain_db = float(np.clip(source_lufs - current_lufs, -18.0, max_boost_db))
    if abs(desired_gain_db) < 0.05:
        return audio.astype(np.float32), 0.0, False

    ceiling = 10 ** (ceiling_db / 20.0)
    true_peak = estimate_true_peak(audio, oversample_factor=2)
    sample_peak = float(np.max(np.abs(as_channel_matrix(audio)))) if audio.size else 0.0
    peak_for_cap = max(true_peak, sample_peak, 1e-9)
    max_safe_gain_db = db(ceiling / peak_for_cap)
    gain_db = min(desired_gain_db, max_safe_gain_db)
    gain = 10 ** (gain_db / 20.0)
    matched = (audio * gain).astype(np.float32)

    matched = apply_soft_limiter(matched, ceiling_db, sr)
    matched, true_peak_trim_db, _ = apply_true_peak_headroom(matched, ceiling_db)
    matched, sample_peak_trim_db = apply_sample_peak_guard(matched)
    total_gain_db = gain_db + true_peak_trim_db + sample_peak_trim_db
    limited = total_gain_db < desired_gain_db - 0.1
    return matched.astype(np.float32), total_gain_db, limited


def finalize_output_safety(
    audio: np.ndarray,
    sr: int,
    ceiling_db: float,
    strict: bool = False,
) -> tuple[np.ndarray, list[str]]:
    finalized = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    steps: list[str] = []
    ceiling = 10 ** (ceiling_db / 20.0)
    allowed_clipping_ratio = 0.0 if strict else 0.0005
    iteration_count = 4 if strict else 2

    for _ in range(iteration_count):
        channels = as_channel_matrix(finalized)
        abs_audio = np.abs(channels)
        clipping_ratio = float(np.mean(abs_audio >= 0.98)) if abs_audio.size else 0.0
        true_peak_db = db(estimate_true_peak(finalized, oversample_factor=4))
        sample_peak = float(np.max(abs_audio)) if abs_audio.size else 0.0

        if clipping_ratio <= allowed_clipping_ratio and true_peak_db <= ceiling_db + 0.02 and sample_peak <= ceiling:
            break

        finalized = apply_soft_limiter(finalized, ceiling_db, sr)
        finalized, true_peak_gain_db, true_peak_before_db = apply_true_peak_headroom(finalized, ceiling_db)
        finalized, sample_peak_gain_db = apply_sample_peak_guard(finalized, ceiling_db)
        if true_peak_gain_db < -0.01:
            steps.append(f"post_render_true_peak_trim_{true_peak_gain_db:+.1f}db_from_{true_peak_before_db:.1f}db")
        if sample_peak_gain_db < -0.01:
            steps.append(f"post_render_sample_guard_{sample_peak_gain_db:+.1f}db")

    channels = as_channel_matrix(finalized)
    sample_peak = float(np.max(np.abs(channels))) if channels.size else 0.0
    true_peak = estimate_true_peak(finalized, oversample_factor=4)
    peak_for_cap = max(sample_peak, true_peak, 1e-9)
    if peak_for_cap > ceiling:
        emergency_gain = ceiling / peak_for_cap
        finalized = (finalized * emergency_gain).astype(np.float32)
        steps.append(f"post_render_emergency_peak_trim_{db(emergency_gain):+.1f}db")

    finalized = np.clip(finalized, -ceiling, ceiling).astype(np.float32)
    if not steps:
        steps.append("post_render_strict_safety_verified" if strict else "post_render_safety_verified")
    return finalized.astype(np.float32), steps


def process_channel_stages(
    channel: np.ndarray,
    sr: int,
    params: dict[str, Any],
    mode: str,
    intensity: float,
    use_model_denoise: bool = True,
) -> tuple[np.ndarray, list[str]]:
    processed = channel.copy()
    steps: list[str] = []

    if abs(float(np.mean(processed))) > 1e-6:
        processed = processed - float(np.mean(processed))
        steps.append("dc_offset_removed")

    if mode == "denoise":
        if intensity <= 0.01:
            steps.append("denoise_skipped_low_ai_amount")
        else:
            processed, used_gtcrn = run_gtcrn_denoise(processed, sr, intensity) if use_model_denoise else (processed, False)
            if used_gtcrn:
                steps.append("gtcrn_denoise")
            else:
                processed = apply_spectral_gate(processed, intensity)
                steps.append("spectral_gate_fallback" if use_model_denoise else "spectral_gate_fast")
    elif mode == "upsample":
        processed = run_onnx_in_chunks(processed, sr)
        steps.append("onnx_upsample_or_passthrough")

    processed = apply_frequency_shaping(processed, sr, params)
    steps.append("frequency_shaping")

    if params["compress_ratio"] > 1.01:
        processed = apply_soft_compression(processed, sr, params["compress_ratio"])
        steps.append("envelope_soft_compression")

    if params["saturation_amount"] > 0.0:
        processed = apply_saturation(processed, params["saturation_amount"])
        steps.append("saturation")
        processed = apply_saturation_smoothing(processed, sr, params["saturation_amount"])
        steps.append("saturation_alias_smoothing")

    if params["exciter_amount"] > 0.0:
        processed = apply_harmonic_exciter(processed, sr, params["exciter_amount"])
        steps.append("harmonic_exciter")

    return processed.astype(np.float32), steps


def process_audio_chain(
    audio: np.ndarray,
    sr: int,
    recommendation: dict[str, Any],
) -> tuple[np.ndarray, int, list[str]]:
    params = recommendation["dsp_params"]
    mode = recommendation["mode"]
    intensity = recommendation["intensity"]
    model_denoise = bool(recommendation.get("model_denoise", True))
    strict_safety = bool(recommendation.get("manual_dsp"))
    limiter_ceiling_db = min(params["limiter_ceiling_db"], MANUAL_DSP_SAFE_CEILING_DB) if strict_safety else params["limiter_ceiling_db"]
    steps: list[str] = []
    was_mono = audio.ndim == 1
    if strict_safety:
        steps.append(f"manual_dsp_strict_ceiling_{limiter_ceiling_db:.1f}db")

    if not was_mono and audio.shape[0] == 2 and recommendation.get("stereo_safe", True):
        mid, side = stereo_to_mid_side(audio)
        side_intensity = min(intensity * 0.35, 0.3) if mode == "denoise" else intensity

        processed_mid, mid_steps = process_channel_stages(
            mid,
            sr,
            params,
            mode,
            intensity,
            use_model_denoise=model_denoise,
        )
        processed_side, side_steps = process_channel_stages(
            side,
            sr,
            params,
            mode,
            side_intensity,
            use_model_denoise=False,
        )
        processed_audio = mid_side_to_stereo(processed_mid, processed_side)
        steps.append("stereo_safe_mid_side")
        steps.extend([f"mid_{step}" for step in mid_steps])
        steps.extend([f"side_{step}" for step in side_steps])
    else:
        processed_channels: list[np.ndarray] = []
        for channel_index, channel in enumerate(audio_channels(audio)):
            processed, channel_steps = process_channel_stages(
                channel,
                sr,
                params,
                mode,
                intensity,
                use_model_denoise=model_denoise,
            )
            processed_channels.append(processed.astype(np.float32))
            for step in channel_steps:
                tagged_step = step if was_mono else f"ch{channel_index + 1}_{step}"
                steps.append(tagged_step)

        processed_audio = stack_channels(processed_channels, was_mono)

    if not was_mono:
        processed_audio = match_channel_balance(audio, processed_audio)
        steps.append("stereo_channel_balance_preserved")
        processed_audio = match_mid_side_balance(audio, processed_audio)
        steps.append("stereo_mid_side_balance_preserved")
        processed_audio, low_bass_side_gain_db = apply_low_bass_phase_guard(audio, processed_audio, sr)
        if low_bass_side_gain_db < -0.05:
            steps.append(f"low_bass_phase_guard_{low_bass_side_gain_db:+.1f}db")
        else:
            steps.append("low_bass_phase_checked")

    processed_audio, band_guard_db = apply_band_balance_guard(audio, processed_audio, sr)
    if band_guard_db >= 0.05:
        steps.append(f"source_band_balance_guard_{band_guard_db:.1f}db")
    else:
        steps.append("source_band_balance_checked")

    processed_audio, transient_loss_db = apply_transient_preservation(audio, processed_audio, sr)
    if transient_loss_db > 1.2:
        steps.append(f"transient_preservation_{transient_loss_db:.1f}db")
    else:
        steps.append("transient_preservation_checked")

    output_sr = int(recommendation.get("output_sr", sr))
    if output_sr != sr:
        processed_audio = resample_audio(processed_audio, sr, output_sr)
        steps.append(f"high_quality_resample_{sr}_to_{output_sr}")

    if params["normalize"]:
        processed_audio, gain_db, gain_limited = apply_loudness_normalize(
            processed_audio,
            params["target_lufs"],
            limiter_ceiling_db,
            sr=output_sr,
        )
        steps.append(f"loudness_normalize_{gain_db:+.1f}db")
        if gain_limited:
            steps.append("peak_aware_gain_limited")

    processed_audio = apply_soft_limiter(processed_audio, limiter_ceiling_db, output_sr)
    steps.append("linked_peak_limiter")
    processed_audio, true_peak_gain_db, true_peak_before_db = apply_true_peak_headroom(
        processed_audio,
        limiter_ceiling_db,
    )
    if true_peak_gain_db < -0.01:
        steps.append(f"true_peak_trim_{true_peak_gain_db:+.1f}db_from_{true_peak_before_db:.1f}db")
    else:
        steps.append("true_peak_checked")

    processed_audio, sample_peak_gain_db = apply_sample_peak_guard(processed_audio)
    if sample_peak_gain_db < -0.01:
        steps.append(f"final_sample_peak_guard_{sample_peak_gain_db:+.1f}db")

    if recommendation.get("volume_mode") == "match_source":
        processed_audio, rematch_gain_db, rematch_limited = apply_safe_source_loudness_match(
            processed_audio,
            audio,
            output_sr,
            limiter_ceiling_db,
        )
        if abs(rematch_gain_db) >= 0.05:
            steps.append(f"post_safety_source_volume_match_{rematch_gain_db:+.1f}db")
        else:
            steps.append("post_safety_source_volume_checked")
        if rematch_limited:
            steps.append("source_volume_match_limited_by_headroom")

    processed_audio, final_safety_steps = finalize_output_safety(
        processed_audio,
        output_sr,
        limiter_ceiling_db,
        strict=strict_safety,
    )
    steps.extend(final_safety_steps)

    return processed_audio.astype(np.float32), output_sr, steps


def build_comparison_report(
    before: dict[str, Any],
    after: dict[str, Any],
    recommendation: dict[str, Any],
    steps: list[str],
) -> dict[str, Any]:
    loudness_delta = after["lufs"] - before["lufs"]
    true_peak_db = after.get("true_peak_db", -180.0)
    clipping_ratio = after.get("clipping_ratio", 0.0)
    stereo_delta = abs(after.get("stereo_width", 0.0) - before.get("stereo_width", 0.0))
    phase_delta = abs(after.get("phase_correlation", 0.0) - before.get("phase_correlation", 0.0))
    volume_matched = abs(loudness_delta) <= (1.25 if recommendation.get("volume_mode") == "match_source" else 1.0)
    clipping_safe = clipping_ratio <= 0.0005 and true_peak_db <= recommendation["dsp_params"]["limiter_ceiling_db"] + 0.2
    headroom_safe = true_peak_db <= -1.0
    stereo_preserved = after.get("channels", 1) < 2 or (stereo_delta <= 0.2 and phase_delta <= 0.25)
    stereo_preservation_score = 100.0 if after.get("channels", 1) < 2 else float(
        np.clip(100.0 - (stereo_delta * 120.0 + phase_delta * 80.0), 0.0, 100.0)
    )
    separation_delta = after.get("band_separation_score", 0.0) - before.get("band_separation_score", 0.0)

    return {
        "before": before,
        "after": after,
        "delta": {
            "lufs": loudness_delta,
            "peak_db": after["peak_db"] - before["peak_db"],
            "noise_floor_db": after["noise_floor_db"] - before["noise_floor_db"],
            "crest_db": after["crest_db"] - before["crest_db"],
            "high_energy": after["high_energy"] - before["high_energy"],
            "stereo_width": after["stereo_width"] - before["stereo_width"],
            "phase_correlation": after["phase_correlation"] - before["phase_correlation"],
            "true_peak_db": after["true_peak_db"] - before["true_peak_db"],
            "band_separation_score": separation_delta,
        },
        "target_lufs": recommendation["dsp_params"]["target_lufs"],
        "limiter_ceiling_db": recommendation["dsp_params"]["limiter_ceiling_db"],
        "loudness_meter": "bs1770_style_k_weighted_gated",
        "recommendation": recommendation,
        "applied_steps": steps,
        "warnings": after.get("quality_flags", []),
        "quality_summary": {
            "volume_matched": volume_matched,
            "loudness_meter": "bs1770_style_k_weighted_gated",
            "loudness_delta_db": loudness_delta,
            "clipping_safe": clipping_safe,
            "headroom_safe": headroom_safe,
            "true_peak_db": true_peak_db,
            "stereo_preserved": stereo_preserved,
            "stereo_preservation_score": stereo_preservation_score,
            "band_separation_score": after.get("band_separation_score", 0.0),
            "band_separation_delta": separation_delta,
            "stereo_width_delta": after.get("stereo_width", 0.0) - before.get("stereo_width", 0.0),
            "phase_correlation_delta": after.get("phase_correlation", 0.0) - before.get("phase_correlation", 0.0),
            "post_safety_volume_match": any(step.startswith("post_safety_source_volume") for step in steps),
            "volume_match_limited_by_headroom": "source_volume_match_limited_by_headroom" in steps,
            "level_match_playback_gain": {
                "original": min(1.0, 10 ** ((min(before["lufs"], after["lufs"]) - before["lufs"]) / 20.0)),
                "enhanced": min(1.0, 10 ** ((min(before["lufs"], after["lufs"]) - after["lufs"]) / 20.0)),
            },
        },
    }


def evaluate_quality_drift(
    before: dict[str, Any],
    after: dict[str, Any],
    recommendation: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    before_channels = int(before.get("channels", 1))
    target = str(recommendation.get("target", ""))
    volume_mode = recommendation.get("volume_mode")
    ceiling_db = float(recommendation.get("dsp_params", {}).get("limiter_ceiling_db", -1.5))

    if before_channels >= 2:
        before_width = float(before.get("stereo_width", 0.0))
        after_width = float(after.get("stereo_width", 0.0))
        if before_width > 0.08 and after_width < before_width * 0.72 and before_width - after_width > 0.10:
            flags.append("stereo_width_loss")
        if before_width > 0.08 and after_width > before_width * 1.55 and after_width - before_width > 0.18:
            flags.append("stereo_width_overexpansion")

        before_phase = float(before.get("phase_correlation", 1.0))
        after_phase = float(after.get("phase_correlation", 1.0))
        if after_phase < -0.10 or (before_phase - after_phase > 0.30 and after_phase < 0.25):
            flags.append("phase_correlation_loss")

    separation_delta = float(after.get("band_separation_score", 0.0)) - float(before.get("band_separation_score", 0.0))
    if separation_delta < -12.0:
        flags.append("band_separation_loss")

    crest_loss = float(before.get("crest_db", 0.0)) - float(after.get("crest_db", 0.0))
    crest_limit = 6.0 if "loud_modern" in target else 4.5
    if crest_loss > crest_limit:
        flags.append("transient_flattening")

    high_delta = float(after.get("high_energy", 0.0)) - float(before.get("high_energy", 0.0))
    flatness_delta = float(after.get("spectral_flatness", 0.0)) - float(before.get("spectral_flatness", 0.0))
    if high_delta > 0.22 and flatness_delta > 0.035 and "hifi_bright" not in target:
        flags.append("high_band_harshness")

    if volume_mode == "match_source" and abs(float(after.get("lufs", 0.0)) - float(before.get("lufs", 0.0))) > 2.25:
        flags.append("source_loudness_drift")

    if float(after.get("clipping_ratio", 0.0)) > 0.0005 or float(after.get("true_peak_db", -180.0)) > ceiling_db + 0.15:
        flags.append("headroom_risk")

    return flags


def apply_post_quality_guard(
    processed_audio: np.ndarray,
    source_audio: np.ndarray,
    source_sr: int,
    output_sr: int,
    before: dict[str, Any],
    recommendation: dict[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any], dict[str, Any]]:
    after = analyze_array(processed_audio, output_sr, {"channels": before.get("channels", 1)})
    drift_flags = evaluate_quality_drift(before, after, recommendation)
    if not drift_flags:
        return processed_audio.astype(np.float32), ["post_quality_guard_checked"], after, {
            "applied": False,
            "flags": [],
            "blend": 0.0,
        }

    target_channels = as_channel_matrix(processed_audio).shape[0]
    target_len = as_channel_matrix(processed_audio).shape[-1]
    processed_ref = fit_audio_length(as_fixed_channel_matrix(processed_audio, target_channels), target_len)
    source_ref = prepare_stem_for_mix(source_audio, source_sr, output_sr, target_channels, target_len)
    blend = float(np.clip(QUALITY_GUARD_BASE_BLEND + 0.055 * len(drift_flags), 0.0, QUALITY_GUARD_MAX_BLEND))
    guarded = processed_ref * (1.0 - blend) + source_ref * blend

    ceiling_db = float(recommendation.get("dsp_params", {}).get("limiter_ceiling_db", -1.5))
    if recommendation.get("volume_mode") == "match_source":
        guarded, match_gain_db, match_limited = apply_safe_source_loudness_match(
            guarded,
            source_ref,
            output_sr,
            ceiling_db,
            max_boost_db=2.0,
        )
    else:
        guarded, match_gain_db, match_limited = apply_loudness_normalize(
            guarded,
            float(recommendation.get("dsp_params", {}).get("target_lufs", -16.0)),
            ceiling_db,
            max_limiter_drive_db=0.5,
            sr=output_sr,
        )

    guarded, safety_steps = finalize_output_safety(guarded, output_sr, ceiling_db, strict=True)
    guarded_after = analyze_array(guarded, output_sr, {"channels": before.get("channels", 1)})
    steps = [f"post_quality_guard_blend_{blend:.2f}_{'_'.join(drift_flags)}"]
    if abs(match_gain_db) >= 0.05:
        steps.append(f"post_quality_guard_loudness_match_{match_gain_db:+.1f}db")
    if match_limited:
        steps.append("post_quality_guard_gain_limited")
    steps.extend([f"post_quality_guard_{step}" for step in safety_steps])
    return guarded.astype(np.float32), steps, guarded_after, {
        "applied": True,
        "flags": drift_flags,
        "blend": blend,
        "loudness_match_gain_db": match_gain_db,
        "gain_limited": match_limited,
        "after_flags": evaluate_quality_drift(before, guarded_after, recommendation),
    }


def analyze_audio_file(path: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    audio, sr, source_info = load_audio(path)
    return audio, sr, analyze_array(audio, sr, source_info)


def resolve_stem_separation_mode(value: str | None) -> str:
    mode = (value or "off").strip().lower()
    return mode if mode in {"off", "2stem", "4stem"} else "off"


def resolve_stem_quality_mode(value: str | None) -> str:
    mode = (value or "balanced").strip().lower()
    return mode if mode in STEM_QUALITY_MODES else "balanced"


def demucs_quality_args(stem_quality: str) -> list[str]:
    quality = resolve_stem_quality_mode(stem_quality)
    if quality == "fast":
        return ["--overlap", "0.10"]
    if quality == "precision":
        return ["-n", "htdemucs_ft", "--overlap", "0.35"]
    return ["--overlap", "0.25"]


def resolve_demucs_command(input_path: str, output_dir: str, stem_mode: str, stem_quality: str = "balanced") -> list[str]:
    demucs_args = ["--two-stems=vocals"] if stem_mode == "2stem" else []
    quality_args = demucs_quality_args(stem_quality)
    runner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demucs_runner.py")
    if os.path.exists(runner_path) and not getattr(sys, "frozen", False) and importlib.util.find_spec("demucs"):
        return [
            sys.executable,
            runner_path,
            *demucs_args,
            *quality_args,
            "-o",
            output_dir,
            input_path,
        ]

    demucs_exe = shutil.which("demucs")
    if demucs_exe:
        return [
            demucs_exe,
            *demucs_args,
            *quality_args,
            "-o",
            output_dir,
            input_path,
        ]

    raise HTTPException(
        status_code=400,
        detail="Stem separation requires Demucs. Install Demucs or turn off stem separation.",
    )


def find_demucs_stem(output_dir: str, filename: str) -> str:
    matches: list[str] = []
    for root, _, files in os.walk(output_dir):
        for item in files:
            if item.lower() == filename:
                matches.append(os.path.join(root, item))
    if not matches:
        raise HTTPException(status_code=500, detail=f"Demucs did not create {filename}.")
    matches.sort(key=lambda path: len(path))
    return matches[0]


def run_demucs_stems(input_path: str, stem_mode: str, stem_quality: str = "balanced") -> tuple[dict[str, str], str]:
    started_at = time.perf_counter()
    work_dir = tempfile.mkdtemp(prefix="demucs_", dir=UPLOAD_DIR)
    command = resolve_demucs_command(input_path, work_dir, stem_mode, stem_quality)
    env = os.environ.copy()
    try:
        import static_ffmpeg

        if static_ffmpeg.add_paths():
            ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffprobe")
            if ffmpeg_path:
                ffmpeg_dir = os.path.dirname(ffmpeg_path)
                env["PATH"] = ffmpeg_dir + os.pathsep + env.get("PATH", "")
    except Exception:
        logger.warning("[STEM] static-ffmpeg path setup failed; falling back to system PATH.", exc_info=True)

    logger.info("[STEM] Running Demucs %s/%s separation: %s", stem_mode, stem_quality, " ".join(command[:6]))

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60 * 20,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=500, detail="Demucs stem separation timed out.") from exc

    if completed.returncode != 0:
        demucs_output = "\n".join(part for part in (completed.stderr, completed.stdout) if part)
        logger.error("[STEM] Demucs failed (%s): %s", completed.returncode, demucs_output[-4000:])
        raise HTTPException(status_code=500, detail="Demucs stem separation failed.")

    logger.info("[STEM] Demucs %s/%s separation completed in %.1fs.", stem_mode, stem_quality, time.perf_counter() - started_at)
    if stem_mode == "4stem":
        return (
            {
                "vocals": find_demucs_stem(work_dir, "vocals.wav"),
                "drums": find_demucs_stem(work_dir, "drums.wav"),
                "bass": find_demucs_stem(work_dir, "bass.wav"),
                "other": find_demucs_stem(work_dir, "other.wav"),
            },
            work_dir,
        )
    return (
        {
            "vocals": find_demucs_stem(work_dir, "vocals.wav"),
            "instrumental": find_demucs_stem(work_dir, "no_vocals.wav"),
        },
        work_dir,
    )


def as_fixed_channel_matrix(audio: np.ndarray, channel_count: int) -> np.ndarray:
    channels = as_channel_matrix(audio)
    if channels.shape[0] == channel_count:
        return channels.astype(np.float32)
    if channel_count == 1:
        return np.mean(channels, axis=0, keepdims=True).astype(np.float32)
    if channels.shape[0] == 1:
        return np.repeat(channels, channel_count, axis=0).astype(np.float32)
    if channels.shape[0] > channel_count:
        return channels[:channel_count].astype(np.float32)

    pad = np.repeat(channels[-1:, :], channel_count - channels.shape[0], axis=0)
    return np.concatenate([channels, pad], axis=0).astype(np.float32)


def fit_audio_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if audio.shape[-1] == target_len:
        return audio.astype(np.float32)
    if audio.shape[-1] > target_len:
        return audio[..., :target_len].astype(np.float32)
    pad_width = [(0, 0)] * audio.ndim
    pad_width[-1] = (0, target_len - audio.shape[-1])
    return np.pad(audio, pad_width, mode="constant").astype(np.float32)


def prepare_stem_for_mix(
    audio: np.ndarray,
    sr: int,
    target_sr: int,
    target_channels: int,
    target_len: int,
) -> np.ndarray:
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
    matrix = as_fixed_channel_matrix(audio, target_channels)
    return fit_audio_length(matrix, target_len)


def remix_processed_stems(
    processed_vocals: np.ndarray,
    vocal_sr: int,
    processed_instrumental: np.ndarray,
    instrumental_sr: int,
    source_audio: np.ndarray,
    source_sr: int,
    output_sr: int,
    vocal_gain: float = STEM_VOCAL_GAIN,
    instrumental_gain: float = STEM_INSTRUMENTAL_GAIN,
) -> np.ndarray:
    source_channels = as_channel_matrix(source_audio)
    target_channels = int(source_channels.shape[0])
    target_len = int(round(source_channels.shape[-1] * (output_sr / max(source_sr, 1))))
    target_len = max(target_len, 1)

    vocals = prepare_stem_for_mix(processed_vocals, vocal_sr, output_sr, target_channels, target_len)
    instrumental = prepare_stem_for_mix(
        processed_instrumental,
        instrumental_sr,
        output_sr,
        target_channels,
        target_len,
    )
    mixed = vocals * float(vocal_gain) + instrumental * float(instrumental_gain)
    return stack_channels([mixed[index] for index in range(target_channels)], target_channels == 1)


def remix_processed_stem_map(
    processed_stems: dict[str, tuple[np.ndarray, int]],
    source_audio: np.ndarray,
    source_sr: int,
    output_sr: int,
) -> np.ndarray:
    source_channels = as_channel_matrix(source_audio)
    target_channels = int(source_channels.shape[0])
    target_len = int(round(source_channels.shape[-1] * (output_sr / max(source_sr, 1))))
    target_len = max(target_len, 1)
    mixed = np.zeros((target_channels, target_len), dtype=np.float32)

    for stem_name, (stem_audio, stem_sr) in processed_stems.items():
        aligned = prepare_stem_for_mix(stem_audio, stem_sr, output_sr, target_channels, target_len)
        gain = float(STEM_REMIX_GAINS.get(stem_name, 0.95))
        mixed += aligned * gain

    return stack_channels([mixed[index] for index in range(target_channels)], target_channels == 1)


def shape_stem_residual(residual: np.ndarray, sr: int) -> np.ndarray:
    matrix = as_channel_matrix(residual).astype(np.float32)
    if matrix.size == 0 or sr <= 0:
        return matrix.astype(np.float32)

    if matrix.shape[0] == 2:
        mid, side = stereo_to_mid_side(matrix)
        shaped_mid = apply_soft_highpass(mid, sr, STEM_RESIDUAL_LOWCUT_HZ * 1.45) * 0.45
        shaped_side = apply_soft_highpass(side, sr, STEM_RESIDUAL_LOWCUT_HZ) * 0.90
        return mid_side_to_stereo(shaped_mid, shaped_side).astype(np.float32)

    shaped_channels = [
        apply_soft_highpass(channel, sr, STEM_RESIDUAL_LOWCUT_HZ) * 0.55
        for channel in matrix
    ]
    return np.stack(shaped_channels, axis=0).astype(np.float32)


def apply_stem_residual_preservation(
    mixed_audio: np.ndarray,
    raw_stems: dict[str, np.ndarray],
    raw_stem_sr: dict[str, int],
    source_audio: np.ndarray,
    source_sr: int,
    output_sr: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_channels = as_channel_matrix(source_audio)
    target_channels = int(source_channels.shape[0])
    target_len = int(round(source_channels.shape[-1] * (output_sr / max(source_sr, 1))))
    target_len = max(target_len, 1)
    source_ref = prepare_stem_for_mix(source_audio, source_sr, output_sr, target_channels, target_len)
    raw_sum = np.zeros((target_channels, target_len), dtype=np.float32)

    for stem_name, stem_audio in raw_stems.items():
        aligned = prepare_stem_for_mix(stem_audio, raw_stem_sr[stem_name], output_sr, target_channels, target_len)
        raw_sum += aligned

    residual = source_ref - raw_sum
    source_rms = rms(source_ref)
    residual_rms = rms(residual)
    if source_rms <= 1e-9 or residual_rms <= source_rms * STEM_RESIDUAL_MIN_SOURCE_RATIO:
        return mixed_audio.astype(np.float32), {
            "applied": False,
            "reason": "residual_below_threshold",
            "source_ratio": 0.0 if source_rms <= 1e-9 else residual_rms / source_rms,
        }

    shaped_residual = shape_stem_residual(residual, output_sr)
    shaped_rms = rms(shaped_residual)
    if shaped_rms <= 1e-9:
        return mixed_audio.astype(np.float32), {
            "applied": False,
            "reason": "shaped_residual_silent",
            "source_ratio": residual_rms / source_rms,
        }

    max_gain = (source_rms * STEM_RESIDUAL_MAX_SOURCE_RATIO) / shaped_rms
    blend_gain = float(np.clip(min(STEM_RESIDUAL_BLEND, max_gain), 0.0, STEM_RESIDUAL_BLEND))
    if blend_gain <= 1e-6:
        return mixed_audio.astype(np.float32), {
            "applied": False,
            "reason": "residual_gain_zero",
            "source_ratio": residual_rms / source_rms,
        }

    mixed_matrix = fit_audio_length(as_fixed_channel_matrix(mixed_audio, target_channels), target_len)
    preserved = mixed_matrix + shaped_residual * blend_gain
    preserved, low_bass_side_gain_db = apply_low_bass_phase_guard(source_ref, preserved, output_sr)
    return stack_channels([preserved[index] for index in range(target_channels)], target_channels == 1), {
        "applied": True,
        "blend_gain": blend_gain,
        "blend_gain_db": db(blend_gain),
        "source_ratio": residual_rms / source_rms,
        "shaped_source_ratio": (shaped_rms * blend_gain) / source_rms,
        "low_bass_side_gain_db": low_bass_side_gain_db,
    }


def combined_stem_reference(
    stem_audio: dict[str, np.ndarray],
    stem_sr: dict[str, int],
    exclude: str,
    target_sr: int,
    target_channels: int,
    target_len: int,
) -> np.ndarray:
    reference = np.zeros((target_channels, target_len), dtype=np.float32)
    for stem_name, audio in stem_audio.items():
        if stem_name == exclude:
            continue
        reference += prepare_stem_for_mix(audio, stem_sr[stem_name], target_sr, target_channels, target_len)
    return stack_channels([reference[index] for index in range(target_channels)], target_channels == 1)


def prepare_recommendation_for_request(
    before: dict[str, Any],
    sr: int,
    target: str,
    intensity: float | None,
    use_denoise: bool,
    volume_mode: str,
    dsp_params: str | None = None,
) -> dict[str, Any]:
    recommendation = recommend_processing(before, target=target, user_intensity=intensity)
    sync_recommendation_output_sr(recommendation, sr)

    volume_mode = volume_mode if volume_mode in {"target", "match_source"} else "target"
    if volume_mode == "match_source":
        source_lufs = float(np.clip(before["lufs"], -24.0, -8.0))
        recommendation["dsp_params"]["target_lufs"] = source_lufs
        recommendation["volume_mode"] = "match_source"
        recommendation["reasons"].append("Matched target loudness to the source loudness.")
        recommendation["advice"] = summarize_recommendation(
            recommendation["dsp_params"],
            recommendation["mode"],
            recommendation["intensity"],
            set(before.get("quality_flags", [])),
        )
    else:
        recommendation["volume_mode"] = "target"

    if not use_denoise and recommendation["mode"] == "denoise":
        recommendation["mode"] = "none"
        recommendation["reasons"].append("Skipped denoise stage by user selection.")
        recommendation["advice"] = summarize_recommendation(
            recommendation["dsp_params"],
            recommendation["mode"],
            recommendation["intensity"],
            set(before.get("quality_flags", [])),
        )

    override_params = parse_manual_dsp_request(dsp_params, recommendation["dsp_params"])
    if override_params is not None:
        recommendation["dsp_params"] = override_params
        recommendation["manual_dsp"] = True
        recommendation["reasons"].append("Manual DSP fine-tuning was applied on top of the selected listening target.")
        recommendation["reasons"].append("Manual DSP fine-tuning uses strict clipping safety.")
        recommendation["advice"] = "Applied manual DSP fine-tuning on top of the selected listening target with strict clipping safety."

    return recommendation


def prepare_stem_recommendation(
    stem_before: dict[str, Any],
    stem_sr: int,
    target: str,
    requested_intensity: float | None,
    use_denoise: bool,
    output_sr: int,
) -> dict[str, Any]:
    stem_intensity = None if requested_intensity is None else min(float(requested_intensity), STEM_INTENSITY_CAP)
    recommendation = prepare_recommendation_for_request(
        stem_before,
        stem_sr,
        target,
        stem_intensity,
        use_denoise,
        "match_source",
        None,
    )
    recommendation["intensity"] = min(float(recommendation.get("intensity", STEM_INTENSITY_CAP)), STEM_INTENSITY_CAP)
    recommendation["ai_amount"] = min(float(recommendation.get("ai_amount", STEM_INTENSITY_CAP)), STEM_INTENSITY_CAP)
    recommendation["output_sr"] = int(output_sr)
    recommendation["model_denoise"] = False
    recommendation["stem_processing"] = True

    params = dict(recommendation["dsp_params"])
    params["limiter_ceiling_db"] = min(float(params.get("limiter_ceiling_db", -1.5)), -1.8)
    params["compress_ratio"] = 1.0 + (float(params.get("compress_ratio", 1.0)) - 1.0) * 0.7
    params["exciter_amount"] = float(params.get("exciter_amount", 0.0)) * 0.55
    params["saturation_amount"] = float(params.get("saturation_amount", 0.0)) * 0.55
    if target == "voice_focus":
        recommendation["intensity"] = min(float(recommendation["intensity"]), STEM_VOCAL_INTENSITY_CAP)
        recommendation["ai_amount"] = min(float(recommendation["ai_amount"]), STEM_VOCAL_INTENSITY_CAP)
        params["high_boost_db"] = min(float(params.get("high_boost_db", 0.0)), 0.9)
        params["compress_ratio"] = min(float(params.get("compress_ratio", 1.0)), 1.35)
        params["exciter_amount"] = min(float(params.get("exciter_amount", 0.0)), 0.025)
        params["saturation_amount"] = min(float(params.get("saturation_amount", 0.0)), 0.015)
        recommendation["reasons"].append("Vocal stem uses bleed-safe conservative tone shaping.")
    recommendation["dsp_params"] = clamp_dsp_params(params)
    recommendation["reasons"].append("Stem processing uses conservative intensity to limit separation artifacts.")
    return recommendation


def finalize_stem_remix(
    mixed_audio: np.ndarray,
    source_audio: np.ndarray,
    output_sr: int,
    recommendation: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    steps: list[str] = []
    limiter_ceiling_db = min(
        float(recommendation["dsp_params"].get("limiter_ceiling_db", STEM_FINAL_CEILING_DB)),
        STEM_FINAL_CEILING_DB,
    )

    mixed_audio = apply_soft_limiter(mixed_audio, limiter_ceiling_db, output_sr)
    steps.append("stem_remix_linked_limiter")
    mixed_audio, true_peak_gain_db, true_peak_before_db = apply_true_peak_headroom(
        mixed_audio,
        limiter_ceiling_db,
    )
    if true_peak_gain_db < -0.01:
        steps.append(f"stem_remix_true_peak_trim_{true_peak_gain_db:+.1f}db_from_{true_peak_before_db:.1f}db")
    else:
        steps.append("stem_remix_true_peak_checked")

    if recommendation.get("volume_mode") == "match_source":
        mixed_audio, rematch_gain_db, rematch_limited = apply_safe_source_loudness_match(
            mixed_audio,
            source_audio,
            output_sr,
            limiter_ceiling_db,
            max_boost_db=4.0,
        )
        steps.append(f"stem_remix_source_volume_match_{rematch_gain_db:+.1f}db")
        if rematch_limited:
            steps.append("stem_remix_volume_match_limited_by_headroom")
    else:
        mixed_audio, gain_db, gain_limited = apply_loudness_normalize(
            mixed_audio,
            float(recommendation["dsp_params"].get("target_lufs", -16.0)),
            limiter_ceiling_db,
            max_limiter_drive_db=0.8,
            sr=output_sr,
        )
        steps.append(f"stem_remix_target_loudness_{gain_db:+.1f}db")
        if gain_limited:
            steps.append("stem_remix_target_gain_limited")

    mixed_audio, safety_steps = finalize_output_safety(
        mixed_audio,
        output_sr,
        limiter_ceiling_db,
        strict=True,
    )
    steps.extend([f"stem_{step}" for step in safety_steps])
    return mixed_audio.astype(np.float32), steps


def process_stem_separated_audio(
    input_path: str,
    source_audio: np.ndarray,
    source_sr: int,
    source_before: dict[str, Any],
    recommendation: dict[str, Any],
    target: str,
    intensity: float | None,
    use_denoise: bool,
    output_sr: int,
    task_id: str,
    bit_depth: str,
    stem_mode: str,
    stem_quality: str = "balanced",
) -> tuple[np.ndarray, int, list[str], dict[str, Any]]:
    started_at = time.perf_counter()
    stem_quality = resolve_stem_quality_mode(stem_quality)
    stem_paths, work_dir = run_demucs_stems(input_path, stem_mode, stem_quality)
    steps: list[str] = [f"demucs_{stem_mode}_{stem_quality}_separation"]

    try:
        logger.info("[STEM] Loading separated stems.")
        stem_audio: dict[str, np.ndarray] = {}
        stem_sr: dict[str, int] = {}
        stem_before: dict[str, dict[str, Any]] = {}
        stem_raw_filenames: dict[str, str] = {}
        stem_enhanced_filenames: dict[str, str] = {}

        for stem_name, stem_path in stem_paths.items():
            audio, audio_sr, source_info = load_audio(stem_path)
            stem_audio[stem_name] = audio
            stem_sr[stem_name] = audio_sr
            stem_before[stem_name] = analyze_array(audio, audio_sr, source_info)
            raw_output_filename = f"{task_id}_{stem_name}_raw.wav"
            stem_raw_filenames[stem_name] = raw_output_filename

        logger.info("[STEM] Writing raw separated stem downloads.")
        for stem_name, stem_path in stem_paths.items():
            shutil.copyfile(stem_path, os.path.join(OUTPUT_DIR, stem_raw_filenames[stem_name]))

        vocal_cleanup_applied = False
        vocal_audio_for_processing = stem_audio["vocals"]
        if "vocals" in stem_audio:
            vocal_matrix = as_channel_matrix(stem_audio["vocals"])
            vocal_reference = combined_stem_reference(
                stem_audio,
                stem_sr,
                "vocals",
                stem_sr["vocals"],
                vocal_matrix.shape[0],
                vocal_matrix.shape[-1],
            )
            vocal_audio_for_processing, vocal_cleanup_applied = apply_vocal_stem_bleed_cleanup(
                stem_audio["vocals"],
                vocal_reference,
                stem_sr["vocals"],
                stem_sr["vocals"],
            )
        if vocal_cleanup_applied:
            steps.append("vocals_reference_bleed_cleanup")

        stem_targets = {
            "vocals": "voice_focus",
            "instrumental": target,
            "drums": "restore",
            "bass": "bass_boost",
            "other": target,
        }
        processed_stems: dict[str, tuple[np.ndarray, int]] = {}
        processed_steps: dict[str, list[str]] = {}
        stem_recommendations: dict[str, dict[str, Any]] = {}
        stem_gain_adjustments: dict[str, dict[str, Any]] = {}
        stem_chain_started_at = time.perf_counter()

        for stem_name in stem_paths:
            stem_input = vocal_audio_for_processing if stem_name == "vocals" else stem_audio[stem_name]
            stem_target = stem_targets.get(stem_name, target)
            recommendation_for_stem = prepare_stem_recommendation(
                stem_before[stem_name],
                stem_sr[stem_name],
                stem_target,
                intensity,
                use_denoise,
                output_sr,
            )
            if stem_name == "drums":
                params = dict(recommendation_for_stem["dsp_params"])
                params["high_boost_db"] = min(float(params.get("high_boost_db", 0.0)), 0.6)
                params["compress_ratio"] = min(float(params.get("compress_ratio", 1.0)), 1.45)
                params["exciter_amount"] = min(float(params.get("exciter_amount", 0.0)), 0.02)
                recommendation_for_stem["dsp_params"] = clamp_dsp_params(params)
            elif stem_name == "bass":
                params = dict(recommendation_for_stem["dsp_params"])
                params["high_boost_db"] = min(float(params.get("high_boost_db", 0.0)), 0.35)
                params["exciter_amount"] = min(float(params.get("exciter_amount", 0.0)), 0.015)
                params["saturation_amount"] = min(float(params.get("saturation_amount", 0.0)), 0.035)
                recommendation_for_stem["dsp_params"] = clamp_dsp_params(params)

            logger.info("[STEM] Processing %s stem.", stem_name)
            processed_audio, processed_sr, stem_steps = process_audio_chain(
                stem_input,
                stem_sr[stem_name],
                recommendation_for_stem,
            )
            processed_audio, guard_steps = apply_stem_specific_guard(
                stem_audio[stem_name],
                stem_sr[stem_name],
                processed_audio,
                processed_sr,
                stem_name,
            )
            stem_steps.extend(guard_steps)
            processed_audio, stem_gain_db, stem_gain_limited = apply_stem_auto_gain_balance(
                stem_audio[stem_name],
                stem_sr[stem_name],
                processed_audio,
                processed_sr,
                stem_name,
            )
            stem_steps.append(f"stem_auto_gain_balance_{stem_gain_db:+.1f}db")
            if stem_gain_limited:
                stem_steps.append("stem_auto_gain_limited_by_headroom")
            stem_gain_adjustments[stem_name] = {
                "gain_db": stem_gain_db,
                "limited_by_headroom": stem_gain_limited,
            }
            processed_stems[stem_name] = (processed_audio, processed_sr)
            processed_steps[stem_name] = stem_steps
            stem_recommendations[stem_name] = recommendation_for_stem

        logger.info("[STEM] Stem DSP completed in %.1fs.", time.perf_counter() - stem_chain_started_at)

        logger.info("[STEM] Writing processed stem downloads.")
        for stem_name, (processed_audio, processed_sr) in processed_stems.items():
            output_filename = f"{task_id}_{stem_name}_enhanced.wav"
            stem_enhanced_filenames[stem_name] = output_filename
            write_audio(os.path.join(OUTPUT_DIR, output_filename), processed_audio, processed_sr, bit_depth=bit_depth)

        logger.info("[STEM] Remixing stems and applying final safety.")
        mixed_audio = remix_processed_stem_map(processed_stems, source_audio, source_sr, output_sr)
        steps.append("stem_align_sample_rate_channels_length")
        steps.append("stem_remix_gain_" + "_".join(f"{name}_{STEM_REMIX_GAINS.get(name, 0.95):.2f}" for name in processed_stems))
        mixed_audio, residual_info = apply_stem_residual_preservation(
            mixed_audio,
            stem_audio,
            stem_sr,
            source_audio,
            source_sr,
            output_sr,
        )
        if residual_info.get("applied"):
            steps.append(f"stem_source_residual_preserved_{residual_info.get('blend_gain_db', 0.0):+.1f}db")
            if float(residual_info.get("low_bass_side_gain_db", 0.0)) < -0.05:
                steps.append(f"stem_residual_low_bass_phase_guard_{residual_info['low_bass_side_gain_db']:+.1f}db")
        else:
            steps.append(f"stem_source_residual_skipped_{residual_info.get('reason', 'unknown')}")
        mixed_audio, final_steps = finalize_stem_remix(mixed_audio, source_audio, output_sr, recommendation)
        for stem_name, stem_steps in processed_steps.items():
            steps.extend([f"{stem_name}_{step}" for step in stem_steps[:12]])
        steps.extend(final_steps)

        stem_report_items: dict[str, Any] = {}
        for stem_name, (processed_audio, processed_sr) in processed_stems.items():
            stem_report_items[stem_name] = {
                "target": stem_targets.get(stem_name, target),
                "before_lufs": stem_before[stem_name].get("lufs"),
                "after_lufs": analyze_array(processed_audio, processed_sr).get("lufs"),
                "sr": processed_sr,
                "steps": processed_steps[stem_name],
                "auto_gain_balance": stem_gain_adjustments.get(stem_name, {}),
                "download_filename": stem_enhanced_filenames[stem_name],
                "raw_download_filename": stem_raw_filenames[stem_name],
                "enhanced_download_filename": stem_enhanced_filenames[stem_name],
            }

        stem_report = {
            "enabled": True,
            "mode": stem_mode,
            "quality_mode": stem_quality,
            "engine": "demucs",
            "stems": stem_report_items,
            "remix_gains": {name: STEM_REMIX_GAINS.get(name, 0.95) for name in processed_stems},
            "auto_gain_balance": stem_gain_adjustments,
            "artifact_guard": "conservative_stem_intensity_vocal_bleed_cleanup_and_source_residual_preservation",
            "vocal_bleed_cleanup": vocal_cleanup_applied,
            "source_residual_preservation": residual_info,
        }
        recommendation["stem_separation"] = {
            "enabled": True,
            "mode": stem_mode,
            "quality_mode": stem_quality,
            "vocal_target": "voice_focus",
            "instrumental_target": target,
            "stem_targets": {name: stem_targets.get(name, target) for name in processed_stems},
        }
        recommendation["reasons"].append(f"Applied Demucs {stem_mode} {stem_quality} separation before final remix.")
        if stem_mode == "4stem":
            recommendation["reasons"].append("4-stem mode processed vocals, drums, bass, and other stems separately.")
        else:
            recommendation["reasons"].append("Vocals used Voice Focus while instrumental used the selected listening target.")
        if vocal_cleanup_applied:
            recommendation["reasons"].append("Applied conservative vocal stem bleed cleanup.")
        if residual_info.get("applied"):
            recommendation["reasons"].append("Preserved low-level source residual during stem remix.")
        logger.info("[STEM] Stem pipeline completed in %.1fs.", time.perf_counter() - started_at)
        return mixed_audio.astype(np.float32), int(output_sr), steps, stem_report
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def process_saved_audio(
    input_path: str,
    original_filename: str,
    target: str,
    intensity: float | None,
    use_denoise: bool,
    volume_mode: str,
    dsp_params: str | None = None,
    output_sample_rate: str | None = "auto",
    output_bit_depth: str | None = "24",
    stem_separation: str | None = "off",
    stem_quality: str | None = "balanced",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    task_id = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{task_id}_enhanced.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    audio, sr, before = analyze_audio_file(input_path)
    recommendation = prepare_recommendation_for_request(
        before,
        sr,
        target,
        intensity,
        use_denoise,
        volume_mode,
        dsp_params,
    )
    bit_depth = resolve_output_bit_depth(output_bit_depth)
    recommendation["output_sr"] = resolve_output_sample_rate(
        output_sample_rate,
        sr,
        int(recommendation.get("output_sr", sr)),
    )
    recommendation["output_bit_depth"] = bit_depth
    stem_mode = resolve_stem_separation_mode(stem_separation)
    stem_quality_mode = resolve_stem_quality_mode(stem_quality)
    stem_report: dict[str, Any] | None = None
    if stem_mode in {"2stem", "4stem"}:
        try:
            processed, output_sr, steps, stem_report = process_stem_separated_audio(
                input_path,
                audio,
                sr,
                before,
                recommendation,
                target,
                intensity,
                use_denoise,
                int(recommendation["output_sr"]),
                task_id,
                bit_depth,
                stem_mode,
                stem_quality_mode,
            )
        except Exception as exc:
            if stem_quality_mode == "precision":
                logger.exception("[STEM] Precision stem quality failed. Falling back to balanced.")
                recommendation["reasons"].append("Precision stem quality failed; used balanced fallback.")
                try:
                    processed, output_sr, steps, stem_report = process_stem_separated_audio(
                        input_path,
                        audio,
                        sr,
                        before,
                        recommendation,
                        target,
                        intensity,
                        use_denoise,
                        int(recommendation["output_sr"]),
                        task_id,
                        bit_depth,
                        stem_mode,
                        "balanced",
                    )
                    if stem_report is not None:
                        stem_report["requested_quality_mode"] = "precision"
                        stem_report["fallback_quality_mode"] = "balanced"
                        stem_report["quality_fallback_reason"] = str(exc)[:240]
                except Exception as balanced_exc:
                    if stem_mode != "4stem":
                        raise balanced_exc
                    logger.exception("[STEM] Balanced 4-stem fallback failed. Falling back to 2-stem.")
                    recommendation["reasons"].append("4-stem balanced fallback failed; used 2-stem fallback.")
                    processed, output_sr, steps, stem_report = process_stem_separated_audio(
                        input_path,
                        audio,
                        sr,
                        before,
                        recommendation,
                        target,
                        intensity,
                        use_denoise,
                        int(recommendation["output_sr"]),
                        task_id,
                        bit_depth,
                        "2stem",
                        "balanced",
                    )
                    steps.insert(0, "4stem_fallback_to_2stem")
                    if stem_report is not None:
                        stem_report["requested_mode"] = "4stem"
                        stem_report["fallback_mode"] = "2stem"
                        stem_report["requested_quality_mode"] = "precision"
                        stem_report["fallback_quality_mode"] = "balanced"
                        stem_report["fallback_reason"] = str(balanced_exc)[:240]
            elif stem_mode != "4stem":
                raise
            else:
                logger.exception("[STEM] 4-stem pipeline failed. Falling back to 2-stem.")
                recommendation["reasons"].append("4-stem processing failed; used 2-stem fallback.")
                processed, output_sr, steps, stem_report = process_stem_separated_audio(
                    input_path,
                    audio,
                    sr,
                    before,
                    recommendation,
                    target,
                    intensity,
                    use_denoise,
                    int(recommendation["output_sr"]),
                    task_id,
                    bit_depth,
                    "2stem",
                    stem_quality_mode,
                )
                steps.insert(0, "4stem_fallback_to_2stem")
                if stem_report is not None:
                    stem_report["requested_mode"] = "4stem"
                    stem_report["fallback_mode"] = "2stem"
                    stem_report["fallback_reason"] = str(exc)[:240]
    else:
        processed, output_sr, steps = process_audio_chain(audio, sr, recommendation)

    processed, quality_guard_steps, after, quality_guard = apply_post_quality_guard(
        processed,
        audio,
        sr,
        output_sr,
        before,
        recommendation,
    )
    steps.extend(quality_guard_steps)

    logger.info("[PROCESS] Writing enhanced audio: %s", output_filename)
    write_audio(output_path, processed, output_sr, bit_depth=bit_depth)
    logger.info("[PROCESS] Building analysis report.")
    report = build_comparison_report(before, after, recommendation, steps)
    report["quality_guard"] = quality_guard
    if stem_report is not None:
        report["stem_separation"] = stem_report
    report["output_format"] = {
        "container": "WAV",
        "bit_depth": int(bit_depth),
        "sample_rate": int(output_sr),
        "subtype": f"PCM_{bit_depth}",
    }
    download_name = f"enhanced_{safe_download_stem(original_filename)}.wav"

    logger.info("[PROCESS] Audio processing completed in %.1fs: %s", time.perf_counter() - started_at, output_filename)
    return {
        "output_path": output_path,
        "output_filename": output_filename,
        "download_name": download_name,
        "source_filename": os.path.basename(original_filename or "audio"),
        "report": report,
    }


def build_stem_downloads(result: dict[str, Any]) -> list[dict[str, str]]:
    stem_report = result.get("report", {}).get("stem_separation") or {}
    stems = stem_report.get("stems") or {}
    if not stem_report.get("enabled") or not stems:
        return []

    source_stem = safe_download_stem(result.get("source_filename", "audio"))
    label_map = {
        "vocals": ("보컬", "vocals"),
        "instrumental": ("반주", "instrumental"),
        "drums": ("드럼", "drums"),
        "bass": ("베이스", "bass"),
        "other": ("기타 악기", "other"),
    }
    downloads: list[dict[str, str]] = []
    stem_order = ["vocals", "instrumental", "drums", "bass", "other"]
    for stem_key in [name for name in stem_order if name in stems]:
        stem_info = stems.get(stem_key) or {}
        korean_name, english_name = label_map.get(stem_key, (stem_key, stem_key))
        for stem_type, filename_key, korean_prefix, english_prefix in [
            ("raw", "raw_download_filename", "원본 분리", "Raw"),
            ("enhanced", "enhanced_download_filename", "개선", "Enhanced"),
        ]:
            filename = stem_info.get(filename_key)
            if not filename:
                continue
            stem_id = f"{stem_type}_{stem_key}"
            download_name = f"{stem_type}_{english_name}_{source_stem}.wav"
            downloads.append(
                {
                    "stem": stem_id,
                    "stem_group": stem_key,
                    "type": stem_type,
                    "label": f"{korean_prefix} {korean_name} Stem",
                    "label_en": f"{english_prefix} {english_name} stem",
                    "download_url": build_download_url(filename, download_name),
                    "output_filename": filename,
                    "filename": download_name,
                }
            )
    return downloads


UPSAMPLER_MODEL_PATH = get_resource_path(os.path.join("models", "light_upsampler.onnx"))
GTCRN_MODEL_PATH = get_resource_path(os.path.join("models", "gtcrn.onnx"))
upsampler_session = initialize_onnx_session(UPSAMPLER_MODEL_PATH)
gtcrn_session = initialize_onnx_session(GTCRN_MODEL_PATH)


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    ext = validate_audio_file(file.filename)
    input_path = save_upload(file, ext)

    try:
        _, _, analysis = analyze_audio_file(input_path)
        return analysis
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[ERROR] Audio analysis failed.")
        raise HTTPException(status_code=500, detail=f"Audio analysis failed: {exc}") from exc
    finally:
        gc.collect()


@app.post("/api/recommend")
async def recommend_audio(
    file: UploadFile = File(...),
    target: str = Form("hifi_clean"),
    intensity: float | None = Form(None),
):
    ext = validate_audio_file(file.filename)
    input_path = save_upload(file, ext)

    try:
        _, _, analysis = analyze_audio_file(input_path)
        recommendation = recommend_processing(analysis, target=target, user_intensity=intensity)
        return {
            "analysis": analysis,
            "recommendation": recommendation,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[ERROR] Audio recommendation failed.")
        raise HTTPException(status_code=500, detail=f"Audio recommendation failed: {exc}") from exc
    finally:
        gc.collect()


@app.post("/api/process")
async def process_audio(
    file: UploadFile = File(...),
    target: str = Form("hifi_clean"),
    intensity: float | None = Form(None),
    use_denoise: bool = Form(True),
    volume_mode: str = Form("match_source"),
    dsp_params: str | None = Form(None),
    output_sample_rate: str = Form("auto"),
    output_bit_depth: str = Form("24"),
    stem_separation: str = Form("off"),
    stem_quality: str = Form("balanced"),
):
    ext = validate_audio_file(file.filename)
    input_path = save_upload(file, ext)

    try:
        result = process_saved_audio(
            input_path,
            file.filename,
            target,
            intensity,
            use_denoise,
            volume_mode,
            dsp_params,
            output_sample_rate,
            output_bit_depth,
            stem_separation,
            stem_quality,
        )

        return {
            "download_url": build_download_url(result["output_filename"], result["download_name"]),
            "filename": result["download_name"],
            "stem_downloads": build_stem_downloads(result),
            "report": result["report"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[ERROR] Audio processing failed.")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {exc}") from exc
    finally:
        gc.collect()


@app.post("/api/process-batch")
async def process_audio_batch(
    files: list[UploadFile] = File(...),
    target: str = Form("hifi_clean"),
    intensity: float | None = Form(None),
    use_denoise: bool = Form(True),
    volume_mode: str = Form("match_source"),
    dsp_params: str | None = Form(None),
    output_sample_rate: str = Form("auto"),
    output_bit_depth: str = Form("24"),
    stem_separation: str = Form("off"),
    stem_quality: str = Form("balanced"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No audio files were uploaded.")
    if len(files) > 30:
        raise HTTPException(status_code=400, detail="Batch processing supports up to 30 files at once.")

    batch_id = str(uuid.uuid4())
    zip_filename = f"{batch_id}_resonix_batch.zip"
    zip_path = os.path.join(OUTPUT_DIR, zip_filename)
    results: list[dict[str, Any]] = []
    used_names: set[str] = set()

    try:
        for index, file in enumerate(files, start=1):
            ext = validate_audio_file(file.filename)
            input_path = save_upload(file, ext)
            result = process_saved_audio(
                input_path,
                file.filename,
                target,
                intensity,
                use_denoise,
                volume_mode,
                dsp_params,
                output_sample_rate=output_sample_rate,
                output_bit_depth=output_bit_depth,
                stem_separation=stem_separation,
                stem_quality=stem_quality,
            )
            base_download_name = result["download_name"]
            arcname = base_download_name
            if arcname in used_names:
                stem, suffix = os.path.splitext(base_download_name)
                arcname = f"{stem}_{index:02d}{suffix}"
            used_names.add(arcname)
            result["archive_name"] = arcname
            results.append(result)

        batch_report = {
            "count": len(results),
            "target": target,
            "ai_amount": intensity,
            "volume_mode": volume_mode,
            "output_sample_rate": output_sample_rate,
            "output_bit_depth": resolve_output_bit_depth(output_bit_depth),
            "manual_dsp": dsp_params is not None,
            "stem_separation": resolve_stem_separation_mode(stem_separation),
            "stem_quality": resolve_stem_quality_mode(stem_quality),
            "items": [
                {
                    "source_filename": item["source_filename"],
                    "archive_name": item["archive_name"],
                    "download_url": build_download_url(item["output_filename"], item["download_name"]),
                    "filename": item["download_name"],
                    "stem_downloads": build_stem_downloads(item),
                    "report": item["report"],
                }
                for item in results
            ],
        }

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive_names = set(used_names)
            for item in results:
                archive.write(item["output_path"], arcname=item["archive_name"])
                archive_names.add(item["archive_name"])
                for stem_download in build_stem_downloads(item):
                    stem_output_filename = os.path.basename(stem_download.get("output_filename", ""))
                    stem_output_path = os.path.join(OUTPUT_DIR, stem_output_filename)
                    if not stem_output_filename or not os.path.exists(stem_output_path):
                        continue
                    stem_arcname = stem_download["filename"]
                    if stem_arcname in archive_names:
                        stem, suffix = os.path.splitext(stem_arcname)
                        stem_arcname = f"{stem}_{len(archive_names):02d}{suffix}"
                    archive_names.add(stem_arcname)
                    archive.write(stem_output_path, arcname=stem_arcname)
            archive.writestr(
                "batch_report.json",
                json.dumps(batch_report, ensure_ascii=False, indent=2),
            )

        summary = [
            {
                "index": index,
                "source_filename": item["source_filename"],
                "archive_name": item["archive_name"],
                "download_url": build_download_url(item["output_filename"], item["download_name"]),
                "filename": item["download_name"],
                "stem_downloads": build_stem_downloads(item),
                "lufs": item["report"]["after"]["lufs"],
                "true_peak_db": item["report"]["after"]["true_peak_db"],
                "sr": item["report"]["after"]["sr"],
                "bit_depth": item["report"]["output_format"]["bit_depth"],
                "channels": item["report"]["after"]["channels"],
                "targets": item["report"]["recommendation"].get("targets")
                or [item["report"]["recommendation"].get("target", target)],
                "report": item["report"],
            }
            for index, item in enumerate(results)
        ]

        return {
            "download_url": build_download_url(zip_filename, "ResonixAI_Batch.zip"),
            "filename": "ResonixAI_Batch.zip",
            "count": len(results),
            "summary": summary,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[ERROR] Batch audio processing failed.")
        raise HTTPException(status_code=500, detail=f"Batch audio processing failed: {exc}") from exc
    finally:
        gc.collect()


@app.post("/api/enhance")
async def enhance_audio(
    file: UploadFile = File(...),
    mode: str = Form(""),
    intensity: float | None = Form(None),
    dsp_params: str | None = Form(None),
):
    ext = validate_audio_file(file.filename)
    input_path = save_upload(file, ext)
    task_id = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{task_id}_enhanced.wav")

    try:
        audio, sr, before = analyze_audio_file(input_path)
        recommendation = recommend_processing(before, user_intensity=intensity)
        if mode in {"denoise", "upsample"}:
            recommendation["mode"] = mode
            sync_recommendation_output_sr(recommendation, sr)

        override_params = parse_manual_dsp_request(dsp_params, recommendation["dsp_params"])
        if override_params is not None:
            recommendation["dsp_params"] = override_params
            recommendation["manual_dsp"] = True

        processed, output_sr, _ = process_audio_chain(audio, sr, recommendation)
        write_audio(output_path, processed, output_sr)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"enhanced_{os.path.splitext(file.filename)[0]}.wav",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[ERROR] Audio enhancement failed.")
        raise HTTPException(status_code=500, detail=f"Audio enhancement failed: {exc}") from exc
    finally:
        gc.collect()


@app.get("/api/download/{filename}")
async def download_output(filename: str, name: str | None = None):
    safe_name = os.path.basename(filename)
    output_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found.")
    download_name = safe_download_filename(name, safe_name)
    suffix = os.path.splitext(safe_name)[1].lower()
    media_type = "application/zip" if suffix == ".zip" else "audio/wav"
    return FileResponse(output_path, media_type=media_type, filename=download_name)


@app.get("/api/version")
async def get_version():
    return {
        "name": "Resonix AI",
        "version": APP_VERSION,
        "app_home": APP_HOME,
        "log_dir": LOG_DIR,
        "distribution": "windows-onedir",
    }


FRONTEND_DIR = get_resource_path("frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
async def index():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="frontend/index.html was not found.")
    return FileResponse(index_path)


@app.get("/app.js")
async def frontend_app_js():
    app_js_path = os.path.join(FRONTEND_DIR, "app.js")
    if not os.path.exists(app_js_path):
        raise HTTPException(status_code=404, detail="frontend/app.js was not found.")
    return FileResponse(app_js_path, media_type="application/javascript")


if __name__ == "__main__":
    def find_available_port(start_port: int = 8000, attempts: int = 20) -> int:
        for port in range(start_port, start_port + attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    continue
        return start_port

    port = find_available_port()
    url = f"http://127.0.0.1:{port}/"
    logger.info("Resonix AI server started at %s", url)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port)
