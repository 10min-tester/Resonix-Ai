import sys
from pathlib import Path

import soundfile as sf
import torch


def save_audio_with_soundfile(
    wav: torch.Tensor,
    path: str | Path,
    samplerate: int,
    bitrate: int = 320,
    clip: str = "rescale",
    bits_per_sample: int = 16,
    as_float: bool = False,
    preset: int = 2,
) -> None:
    from demucs.audio import prevent_clip

    del bitrate, preset
    path = Path(path)
    wav = prevent_clip(wav.detach().cpu(), mode=clip)
    data = wav.transpose(0, 1).numpy()

    suffix = path.suffix.lower()
    if suffix == ".wav":
        subtype = "FLOAT" if as_float or bits_per_sample == 32 else f"PCM_{bits_per_sample}"
        sf.write(path, data, samplerate, subtype=subtype)
    elif suffix == ".flac":
        sf.write(path, data, samplerate, subtype=f"PCM_{bits_per_sample}")
    else:
        raise ValueError(f"Unsupported Demucs output suffix: {suffix}")


def main() -> None:
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except Exception:
        pass

    import demucs.audio
    import demucs.separate

    demucs.audio.save_audio = save_audio_with_soundfile
    demucs.separate.save_audio = save_audio_with_soundfile
    demucs.separate.main(sys.argv[1:])


if __name__ == "__main__":
    main()
