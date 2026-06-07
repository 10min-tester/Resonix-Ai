import gc
import json
import logging
import os
import shutil
import socket
import sys
import threading
import uuid
import webbrowser
import zipfile
from typing import Any

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


def write_audio(path: str, audio: np.ndarray, sr: int, bit_depth: str | None = "24") -> None:
    audio, _ = apply_sample_peak_guard(audio)
    subtype = None
    if os.path.splitext(path)[1].lower() == ".wav":
        subtype = "PCM_16" if resolve_output_bit_depth(bit_depth) == "16" else "PCM_24"
    if audio.ndim == 1:
        sf.write(path, audio, sr, subtype=subtype)
        return
    sf.write(path, audio.T, sr, subtype=subtype)


def db(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-9)))


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
    stereo_metrics = analyze_stereo_image(audio)
    abs_audio = np.abs(channels)
    rms = float(np.sqrt(np.mean(np.square(channels))))
    peak = float(np.max(abs_audio))
    crest_db = db(peak / max(rms, 1e-9))
    lufs = db(rms) - 0.691

    spectrum = np.abs(np.fft.rfft(channels, axis=1))
    spectrum_power = np.mean(np.square(spectrum), axis=0)
    freqs = np.fft.rfftfreq(channels.shape[1], 1.0 / sr)

    low_energy = band_energy(spectrum_power, freqs, 20, 250)
    mid_energy = band_energy(spectrum_power, freqs, 250, 4000)
    high_energy = band_energy(spectrum_power, freqs, 4000, min(sr / 2, 20000))

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
        "stereo_width": stereo_metrics["stereo_width"],
        "phase_correlation": stereo_metrics["phase_correlation"],
        "mid_lufs": stereo_metrics["mid_lufs"],
        "side_lufs": stereo_metrics["side_lufs"],
        "quality_flags": quality_flags,
    }


def analyze_stereo_image(audio: np.ndarray) -> dict[str, float]:
    if audio.ndim == 1 or audio.shape[0] < 2:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        return {
            "stereo_width": 0.0,
            "phase_correlation": 1.0,
            "mid_lufs": db(rms) - 0.691,
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
        "mid_lufs": db(mid_rms) - 0.691,
        "side_lufs": db(side_rms) - 0.691,
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

    return clamp_dsp_params(params)


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


def apply_frequency_shaping(audio: np.ndarray, sr: int, params: dict[str, Any]) -> np.ndarray:
    filtered = apply_soft_highpass(audio, sr, params["lowcut_hz"])
    shaped = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(filtered), 1.0 / sr)
    gain = np.ones_like(freqs, dtype=np.float32)

    gain[(freqs >= 20) & (freqs < 250)] *= 10 ** (params["low_boost_db"] / 20.0)
    gain[(freqs >= 250) & (freqs < 4000)] *= 10 ** (params["mid_cut_db"] / 20.0)
    gain[freqs >= 4000] *= 10 ** (params["high_boost_db"] / 20.0)

    return np.fft.irfft(shaped * gain, n=len(filtered)).astype(np.float32)


def apply_soft_compression(audio: np.ndarray, ratio: float) -> np.ndarray:
    if ratio <= 1.01:
        return audio

    threshold = 0.35
    sign = np.sign(audio)
    magnitude = np.abs(audio)
    over = magnitude > threshold
    compressed = magnitude.copy()
    compressed[over] = threshold + (magnitude[over] - threshold) / ratio
    return (sign * compressed).astype(np.float32)


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


def apply_harmonic_exciter(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    amount = float(np.clip(amount, 0.0, 0.35))
    if amount <= 0.0:
        return audio

    window_size = max(3, int(sr / 6000))
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    low = np.convolve(audio, kernel, mode="same")
    high = audio - low
    excited = np.tanh(high * 4.0) * 0.45
    return (audio + excited * amount).astype(np.float32)


def apply_loudness_normalize(
    audio: np.ndarray,
    target_lufs: float,
    limiter_ceiling_db: float,
    max_limiter_drive_db: float = 1.5,
) -> tuple[np.ndarray, float, bool]:
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms <= 1e-9:
        return audio, 0.0, False

    current_lufs = db(rms) - 0.691
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
        processed = apply_soft_compression(processed, params["compress_ratio"])
        steps.append("soft_compression")

    if params["saturation_amount"] > 0.0:
        processed = apply_saturation(processed, params["saturation_amount"])
        steps.append("saturation")

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
    steps: list[str] = []
    was_mono = audio.ndim == 1

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

    output_sr = int(recommendation.get("output_sr", sr))
    if output_sr != sr:
        processed_audio = resample_audio(processed_audio, sr, output_sr)
        steps.append(f"high_quality_resample_{sr}_to_{output_sr}")

    if params["normalize"]:
        processed_audio, gain_db, gain_limited = apply_loudness_normalize(
            processed_audio,
            params["target_lufs"],
            params["limiter_ceiling_db"],
        )
        steps.append(f"loudness_normalize_{gain_db:+.1f}db")
        if gain_limited:
            steps.append("peak_aware_gain_limited")

    processed_audio = apply_soft_limiter(processed_audio, params["limiter_ceiling_db"], output_sr)
    steps.append("linked_peak_limiter")
    processed_audio, true_peak_gain_db, true_peak_before_db = apply_true_peak_headroom(
        processed_audio,
        params["limiter_ceiling_db"],
    )
    if true_peak_gain_db < -0.01:
        steps.append(f"true_peak_trim_{true_peak_gain_db:+.1f}db_from_{true_peak_before_db:.1f}db")
    else:
        steps.append("true_peak_checked")

    processed_audio, sample_peak_gain_db = apply_sample_peak_guard(processed_audio)
    if sample_peak_gain_db < -0.01:
        steps.append(f"final_sample_peak_guard_{sample_peak_gain_db:+.1f}db")

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
    volume_matched = recommendation.get("volume_mode") == "match_source" or abs(loudness_delta) <= 1.0
    clipping_safe = clipping_ratio <= 0.0005 and true_peak_db <= recommendation["dsp_params"]["limiter_ceiling_db"] + 0.2
    headroom_safe = true_peak_db <= -1.0
    stereo_preserved = after.get("channels", 1) < 2 or (stereo_delta <= 0.2 and phase_delta <= 0.25)

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
        },
        "target_lufs": recommendation["dsp_params"]["target_lufs"],
        "limiter_ceiling_db": recommendation["dsp_params"]["limiter_ceiling_db"],
        "recommendation": recommendation,
        "applied_steps": steps,
        "warnings": after.get("quality_flags", []),
        "quality_summary": {
            "volume_matched": volume_matched,
            "loudness_delta_db": loudness_delta,
            "clipping_safe": clipping_safe,
            "headroom_safe": headroom_safe,
            "true_peak_db": true_peak_db,
            "stereo_preserved": stereo_preserved,
            "stereo_width_delta": after.get("stereo_width", 0.0) - before.get("stereo_width", 0.0),
            "phase_correlation_delta": after.get("phase_correlation", 0.0) - before.get("phase_correlation", 0.0),
            "level_match_playback_gain": {
                "original": min(1.0, 10 ** ((min(before["lufs"], after["lufs"]) - before["lufs"]) / 20.0)),
                "enhanced": min(1.0, 10 ** ((min(before["lufs"], after["lufs"]) - after["lufs"]) / 20.0)),
            },
        },
    }


def analyze_audio_file(path: str) -> tuple[np.ndarray, int, dict[str, Any]]:
    audio, sr, source_info = load_audio(path)
    return audio, sr, analyze_array(audio, sr, source_info)


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

    override_params = parse_dsp_params(dsp_params)
    if override_params is not None:
        recommendation["dsp_params"] = override_params
        recommendation["advice"] = "Used caller-supplied DSP parameters."

    return recommendation


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
) -> dict[str, Any]:
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
    processed, output_sr, steps = process_audio_chain(audio, sr, recommendation)
    write_audio(output_path, processed, output_sr, bit_depth=bit_depth)
    after = analyze_array(processed, output_sr, {"channels": before["channels"]})
    report = build_comparison_report(before, after, recommendation, steps)
    report["output_format"] = {
        "container": "WAV",
        "bit_depth": int(bit_depth),
        "sample_rate": int(output_sr),
        "subtype": f"PCM_{bit_depth}",
    }
    download_name = f"enhanced_{safe_download_stem(original_filename)}.wav"

    return {
        "output_path": output_path,
        "output_filename": output_filename,
        "download_name": download_name,
        "source_filename": os.path.basename(original_filename or "audio"),
        "report": report,
    }


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
        )

        return {
            "download_url": f"/api/download/{result['output_filename']}",
            "filename": result["download_name"],
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
            "items": [
                {
                    "source_filename": item["source_filename"],
                    "archive_name": item["archive_name"],
                    "download_url": f"/api/download/{item['output_filename']}",
                    "filename": item["download_name"],
                    "report": item["report"],
                }
                for item in results
            ],
        }

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in results:
                archive.write(item["output_path"], arcname=item["archive_name"])
            archive.writestr(
                "batch_report.json",
                json.dumps(batch_report, ensure_ascii=False, indent=2),
            )

        summary = [
            {
                "index": index,
                "source_filename": item["source_filename"],
                "archive_name": item["archive_name"],
                "download_url": f"/api/download/{item['output_filename']}",
                "filename": item["download_name"],
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
            "download_url": f"/api/download/{zip_filename}",
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

        override_params = parse_dsp_params(dsp_params)
        if override_params is not None:
            recommendation["dsp_params"] = override_params

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
async def download_output(filename: str):
    safe_name = os.path.basename(filename)
    output_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(output_path, media_type="audio/wav", filename=safe_name)


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
