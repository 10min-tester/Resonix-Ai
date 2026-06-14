import numpy as np

import main


def make_stereo_source(sr: int = 48000, seconds: float = 2.0) -> np.ndarray:
    sample_count = int(sr * seconds)
    t = np.arange(sample_count, dtype=np.float32) / sr
    left = (
        0.14 * np.sin(2 * np.pi * 90 * t)
        + 0.055 * np.sin(2 * np.pi * 1200 * t)
        + 0.018 * np.sin(2 * np.pi * 7200 * t)
    )
    right = (
        0.13 * np.sin(2 * np.pi * 92 * t + 0.2)
        + 0.050 * np.sin(2 * np.pi * 1180 * t + 0.3)
        - 0.016 * np.sin(2 * np.pi * 7200 * t)
    )
    return np.stack([left, right]).astype(np.float32)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_full_mix_quality_chain() -> None:
    sr = 48000
    source = make_stereo_source(sr)
    before = main.analyze_array(source, sr, {"channels": 2})
    recommendation = main.prepare_recommendation_for_request(
        before,
        sr,
        "hifi_clean,bass_boost",
        0.55,
        True,
        "match_source",
        None,
    )
    processed, output_sr, steps = main.process_audio_chain(source, sr, recommendation)
    after = main.analyze_array(processed, output_sr, {"channels": 2})
    report = main.build_comparison_report(before, after, recommendation, steps)

    assert_true(processed.shape == source.shape, "processed shape changed unexpectedly")
    assert_true(np.isfinite(main.as_channel_matrix(processed)).all(), "processed audio has non-finite values")
    assert_true(after["true_peak_db"] <= main.CODEC_SAFE_TRUE_PEAK_CEILING_DB + 0.08, "codec-safe true peak ceiling failed")
    assert_true(any(step.startswith("delta_based_dsp_amount") for step in steps), "delta-based DSP guard did not run")
    assert_true("inter_module_peak_watch_checked" in steps or any(step.startswith("inter_module_peak_watch_") for step in steps), "inter-module peak watch did not run")
    assert_true(report["quality_validation"]["score"] >= 0.0, "quality validation score missing")


def test_manual_strict_safety() -> None:
    sr = 48000
    source = make_stereo_source(sr) * 3.0
    before = main.analyze_array(source, sr, {"channels": 2})
    recommendation = main.prepare_recommendation_for_request(
        before,
        sr,
        "loud_modern",
        1.0,
        True,
        "target",
        '{"manual_mode":"fine_tune","target_lufs_delta":8,"compress_delta":2,"exciter_delta":0.2,"saturation_delta":0.2}',
    )
    processed, output_sr, steps = main.process_audio_chain(source, sr, recommendation)
    after = main.analyze_array(processed, output_sr, {"channels": 2})

    assert_true(recommendation.get("manual_dsp") is True, "manual DSP flag missing")
    assert_true(after["true_peak_db"] <= main.CODEC_SAFE_TRUE_PEAK_CEILING_DB + 0.08, "manual strict true peak safety failed")
    assert_true(any(step.startswith("manual_dsp_strict_ceiling") for step in steps), "manual strict ceiling step missing")
    assert_true(any("post_render" in step or "true_peak" in step for step in steps), "final safety step missing")


def test_stem_risk_map_and_safe_amount() -> None:
    sr = 48000
    source = make_stereo_source(sr)
    before = main.analyze_array(source, sr, {"channels": 2})
    noisy = dict(before)
    noisy.update(
        {
            "spectral_flatness": 0.18,
            "noise_floor_db": -34.0,
            "crest_db": 4.5,
            "phase_correlation": -0.05,
            "stereo_width": 1.6,
        }
    )
    risk_map = main.build_stem_risk_map(
        {
            "vocals": noisy,
            "drums": noisy,
            "bass": noisy,
            "other": noisy,
        }
    )
    safe_amount = main.calculate_stem_safe_amount(
        main.STEM_INTENSITY_CAP,
        1.0,
        0.6,
        risk_map["stems"]["vocals"],
    )

    assert_true(risk_map["average"] > 0.65, "high-risk stems should request fallback")
    assert_true(risk_map["decision"] == "fallback", "stem risk fallback decision failed")
    assert_true(safe_amount < 0.12, "stem safe amount was not attenuated enough")


def test_validation_penalizes_excessive_loudness() -> None:
    sr = 48000
    source = make_stereo_source(sr)
    before = main.analyze_array(source, sr, {"channels": 2})
    after_audio = source * 2.6
    after_audio, _ = main.finalize_output_safety(after_audio, sr, -1.2, strict=True)
    after = main.analyze_array(after_audio, sr, {"channels": 2})
    recommendation = main.prepare_recommendation_for_request(
        before,
        sr,
        "hifi_clean",
        0.6,
        True,
        "match_source",
        None,
    )
    validation = main.build_quality_validation(before, after, recommendation, ["test"])
    risk = validation.get("risk", {})

    assert_true("penalties" in risk, "validation risk details missing")
    assert_true(validation["score"] <= 95.0, "excessive loudness should not receive a perfect score")


def main_entry() -> None:
    test_full_mix_quality_chain()
    test_manual_strict_safety()
    test_stem_risk_map_and_safe_amount()
    test_validation_penalizes_excessive_loudness()
    print("quality_guard_smoke: ok")


if __name__ == "__main__":
    main_entry()
