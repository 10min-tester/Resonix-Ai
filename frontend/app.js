window.__RESONIX_APP_LOADED__ = true;

let originalAudioBlob = null;
let originalAudioUrl = null;
let enhancedAudioUrl = null;
let selectedAudioFiles = [];
let selectedAudioUrls = [];
let batchResults = [];
let isProcessing = false;
let currentReport = null;
let progressTimer = null;
let progressValue = 0;
let progressWaitTick = 0;
let activeJobId = null;
let activeJobCancelled = false;
let appCapabilities = {
    precision_stems: false,
    recommended_stem_quality: "balanced",
};
let currentABMode = "original";
let waveformAnimationFrame = null;
let waveformAudioContext = null;
let waveformResizeTimer = null;
let levelMatchGains = {
    original: 1,
    enhanced: 1,
};

const waveformState = {
    original: null,
    enhanced: null,
};

const API_ORIGIN = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";
const THEME_STORAGE_KEY = "resonix_theme";

const themePresets = {
    "neon-blue": {
        bg: "#030408",
        accent: "#00f0ff",
        accent2: "#8b5cf6",
    },
    "aurora-green": {
        bg: "#03100d",
        accent: "#34d399",
        accent2: "#38bdf8",
    },
    "violet-pulse": {
        bg: "#080511",
        accent: "#a78bfa",
        accent2: "#22d3ee",
    },
    "ruby-night": {
        bg: "#10040b",
        accent: "#fb7185",
        accent2: "#f472b6",
    },
    "amber-studio": {
        bg: "#0d0802",
        accent: "#f59e0b",
        accent2: "#22d3ee",
    },
    "soft-sky": {
        bg: "#dcecf7",
        accent: "#2563eb",
        accent2: "#06b6d4",
    },
    "peach-coral": {
        bg: "#f3ded6",
        accent: "#e11d48",
        accent2: "#f97316",
    },
    "lilac-mist": {
        bg: "#e8ddf4",
        accent: "#7c3aed",
        accent2: "#db2777",
    },
};

function apiUrl(path) {
    return `${API_ORIGIN}${path}`;
}

function resolveApiAssetUrl(path) {
    if (!path || /^(blob:|data:|https?:|file:)/i.test(path)) {
        return path;
    }
    return path.startsWith("/api/") ? apiUrl(path) : path;
}

const text = {
    invalidFile: "\uc9c0\uc6d0\ud558\uc9c0 \uc54a\ub294 \ud30c\uc77c \ud615\uc2dd\uc785\ub2c8\ub2e4. wav, mp3, m4a, flac \ud30c\uc77c\uc744 \uc120\ud0dd\ud574 \uc8fc\uc138\uc694. (Unsupported file type.)",
    processing: "AI \ubd84\uc11d \ubc0f \ubcf5\uc6d0 \uc911... (AI analyzing and restoring...)",
    batchProcessing: "\ubc30\uce58 AI \ubd84\uc11d \ubc0f \ubcf5\uc6d0 \uc911... (Batch AI processing...)",
    processFailed: "\ucc98\ub9ac \uc2e4\ud328 (Processing failed)",
    error: "\uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4. (Error):",
    start: "AI \ubcf5\uc6d0 \uc2dc\uc791 (Start AI restoration)",
    batchStart: "\ubc30\uce58 AI \ubcf5\uc6d0 \uc2dc\uc791 (Start batch restoration)",
    original: "\uc6d0\ubcf8 \uc624\ub514\uc624 (Original audio)",
    enhanced: "\ud5a5\uc0c1\ub41c \uc624\ub514\uc624 (Enhanced audio)",
    loadTimeout: "\uc624\ub514\uc624 \ub85c\ub4dc \uc2dc\uac04\uc774 \ucd08\uacfc\ub418\uc5c8\uc2b5\ub2c8\ub2e4. (Audio load timed out.)",
    loadFailed: "\uc624\ub514\uc624 \ub85c\ub4dc \uc2e4\ud328. (Audio load failed.)",
};

const progressStages = [
    {
        at: 0,
        text: "AI\uac00 \uc6d0\ubcf8 \uc74c\uc6d0\uc744 \ubd84\uc11d\ud558\ub294 \uc911... (Analyzing source audio...)",
    },
    {
        at: 18,
        text: "\uccad\uac10 \ubaa9\ud45c\uc5d0 \ub9de\ub294 \ucc98\ub9ac \uc124\uacc4 \uc911... (Designing target processing...)",
    },
    {
        at: 38,
        text: "\uc2a4\ud14c\ub808\uc624 \uc2a4\ud14c\uc774\uc9d5\uc744 \ubcf4\ud638\ud558\ub294 \uc911... (Protecting stereo staging...)",
    },
    {
        at: 58,
        text: "\ub178\uc774\uc988, \ud1a4, \ub2e4\uc774\ub0b4\ubbf9\uc744 \ubcf5\uc6d0\ud558\ub294 \uc911... (Restoring noise, tone, and dynamics...)",
    },
    {
        at: 78,
        text: "\ucd9c\ub825 \ud30c\uc77c, \ub9ac\ud3ec\ud2b8, \ud5e4\ub4dc\ub8f8\uc744 \ub9c8\ubb34\ub9ac\ud558\ub294 \uc911... (Finalizing output, report, and headroom...)",
    },
    {
        at: 100,
        text: "\ucc98\ub9ac \uc644\ub8cc. A/B \ube44\uad50\ub97c \uc900\ube44\ud588\uc2b5\ub2c8\ub2e4. (Processing complete.)",
    },
];

const stemProgressStages = [
    {
        at: 0,
        text: "\u0041\u0049\uac00 \uc6d0\ubcf8 \uc74c\uc6d0\uc744 \ubd84\uc11d\ud558\ub294 \uc911... (Analyzing source audio...)",
    },
    {
        at: 12,
        text: "\u0044\u0065\u006d\u0075\u0063\u0073\ub85c \u0073\u0074\u0065\u006d\uc744 \ubd84\ub9ac\ud558\ub294 \uc911... 4-stem\uc740 \uba87 \ubd84 \uac78\ub9b4 \uc218 \uc788\uc2b5\ub2c8\ub2e4. (Separating stems...)",
    },
    {
        at: 36,
        text: "\ubd84\ub9ac\ub41c \u0073\u0074\u0065\u006d\ubcc4\ub85c \ubcf4\uc218\uc801\uc778 \uac1c\uc120\uc744 \uc801\uc6a9\ud558\ub294 \uc911... (Enhancing separated stems...)",
    },
    {
        at: 64,
        text: "\u0053\u0074\u0065\u006d\uc744 \ub2e4\uc2dc \ud569\uc131\ud558\uba70 \uc2a4\ud14c\uc774\uc9d5\uc744 \ub9de\ucd94\ub294 \uc911... (Remixing stems and preserving staging...)",
    },
    {
        at: 88,
        text: "\uc11c\ubc84 \uc791\uc5c5 \uc644\ub8cc\ub97c \uae30\ub2e4\ub9ac\ub294 \uc911... Stem \ubd84\ub9ac, \ud569\uc131, \ud5e4\ub4dc\ub8f8 \uac80\uc0ac\uac00 \uc774\uc5b4\uc9c8 \uc218 \uc788\uc2b5\ub2c8\ub2e4. (Waiting for server processing...)",
    },
    {
        at: 100,
        text: "\ucc98\ub9ac \uc644\ub8cc. A/B \ube44\uad50\ub97c \uc900\ube44\ud588\uc2b5\ub2c8\ub2e4. (Processing complete.)",
    },
];

const targetLabels = {
    restore: "\ubcf5\uc6d0 (Restore)",
    hifi_clean: "\ud558\uc774\ud30c\uc774 \ud074\ub9b0 (Hi-Fi Clean)",
    hifi_bright: "\ud558\uc774\ud30c\uc774 \ube0c\ub77c\uc774\ud2b8 (Hi-Fi Bright)",
    warm_analog: "\uc6dc \uc544\ub0a0\ub85c\uadf8 (Warm Analog)",
    loud_modern: "\ub77c\uc6b0\ub4dc \ubaa8\ub358 (Loud Modern)",
    bass_boost: "\uc800\uc74c \ubcf4\uac15 (Bass Boost)",
    voice_focus: "\ubcf4\uc774\uc2a4 \ud3ec\ucee4\uc2a4 (Voice Focus)",
};

const flagLabels = {
    clipping_detected: "\ud074\ub9ac\ud551 \uac10\uc9c0 (Clipping detected)",
    dc_offset: "DC \uc624\ud504\uc14b \uac10\uc9c0 (DC offset)",
    high_noise_floor: "\ub178\uc774\uc988 \ud50c\ub85c\uc5b4 \ub192\uc74c (High noise floor)",
    dull_high_end: "\uace0\uc5ed \ub514\ud14c\uc77c \ubd80\uc871 (Dull high end)",
    very_low_loudness: "\ud3c9\uade0 \uc74c\ub7c9 \ub9e4\uc6b0 \ub0ae\uc74c (Very low loudness)",
    hot_loudness: "\ud3c9\uade0 \uc74c\ub7c9 \ub192\uc74c (Hot loudness)",
    large_silence_sections: "\ubb34\uc74c \uad6c\uac04 \ub9ce\uc74c (Large silence sections)",
    low_sample_rate: "\ub0ae\uc740 \uc0d8\ud50c\ub808\uc774\ud2b8 (Low sample rate)",
};

const reasonLabels = new Map([
    ["Restore target prioritizes cleanup and source preservation.", "\ubcf5\uc6d0 \ubaa9\ud45c\ub294 \uc6d0\ubcf8 \ubcf4\uc874\uacfc \uacb0\ud568 \uc815\ub9ac\ub97c \uc6b0\uc120\ud569\ub2c8\ub2e4. (Restore target prioritizes cleanup and source preservation.)"],
    ["Hi-Fi Clean target keeps a balanced, low-fatigue sound.", "\ud558\uc774\ud30c\uc774 \ud074\ub9b0\uc740 \uade0\ud615\uac10\uacfc \ub0ae\uc740 \ud53c\ub85c\uac10\uc744 \uc720\uc9c0\ud569\ub2c8\ub2e4. (Hi-Fi Clean target keeps a balanced, low-fatigue sound.)"],
    ["Hi-Fi Bright target adds presence and upper detail.", "\ud558\uc774\ud30c\uc774 \ube0c\ub77c\uc774\ud2b8\ub294 \uc874\uc7ac\uac10\uacfc \uace0\uc5ed \ub514\ud14c\uc77c\uc744 \ub354\ud569\ub2c8\ub2e4. (Hi-Fi Bright target adds presence and upper detail.)"],
    ["Warm Analog target adds body and smooths the top end.", "\uc6dc \uc544\ub0a0\ub85c\uadf8\ub294 \ub450\uaed8\uac10\uc744 \ub354\ud558\uace0 \uace0\uc5ed\uc744 \ubd80\ub4dc\ub7fd\uac8c \uc815\ub9ac\ud569\ub2c8\ub2e4. (Warm Analog target adds body and smooths the top end.)"],
    ["Loud Modern target increases density and perceived level.", "\ub77c\uc6b0\ub4dc \ubaa8\ub358\uc740 \ubc00\ub3c4\uc640 \uccb4\uac10 \uc74c\ub7c9\uc744 \ub192\uc785\ub2c8\ub2e4. (Loud Modern target increases density and perceived level.)"],
    ["Bass Boost target adds low-end weight while protecting headroom.", "\uc800\uc74c \ubcf4\uac15\uc740 \ud5e4\ub4dc\ub8f8\uc744 \uc9c0\ud0a4\uba74\uc11c \uc800\uc5ed\uc758 \ubb34\uac8c\uac10\uc744 \ub354\ud569\ub2c8\ub2e4. (Bass Boost target adds low-end weight while protecting headroom.)"],
    ["Voice Focus target emphasizes speech clarity.", "\ubcf4\uc774\uc2a4 \ud3ec\ucee4\uc2a4\ub294 \ub9d0\uc18c\ub9ac \uba85\ub8cc\ub3c4\ub97c \uac15\uc870\ud569\ub2c8\ub2e4. (Voice Focus target emphasizes speech clarity.)"],
    ["Raised denoise intensity because the estimated noise floor is high.", "\ucd94\uc815 \ub178\uc774\uc988\uac00 \ub192\uc544 \ub178\uc774\uc988 \uc81c\uac70 \uac15\ub3c4\ub97c \uc62c\ub838\uc2b5\ub2c8\ub2e4. (Raised denoise intensity because the estimated noise floor is high.)"],
    ["Kept denoise intensity controlled to preserve natural texture.", "\uc790\uc5f0\uc2a4\ub7ec\uc6b4 \uc9c8\uac10\uc744 \ubcf4\uc874\ud558\uae30 \uc704\ud574 \ub178\uc774\uc988 \uc81c\uac70 \uac15\ub3c4\ub97c \uc808\uc81c\ud588\uc2b5\ub2c8\ub2e4. (Kept denoise intensity controlled to preserve natural texture.)"],
    ["Added low-band support because bass energy is weak.", "\uc800\uc5ed \uc5d0\ub108\uc9c0\uac00 \uc57d\ud574 \uc800\uc5ed \ubcf4\uac15\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Added low-band support because bass energy is weak.)"],
    ["Reduced low-band buildup to improve clarity.", "\uba85\ub8cc\ub3c4\ub97c \uc704\ud574 \uc800\uc5ed \ubd80\ud480\uc74c\uc744 \uc904\uc600\uc2b5\ub2c8\ub2e4. (Reduced low-band buildup to improve clarity.)"],
    ["Controlled bass boost because the source already has strong low energy.", "\uc18c\uc2a4\uc758 \uc800\uc5ed\uc774 \uc774\ubbf8 \uac15\ud574 \uc800\uc74c \ubcf4\uac15\uc744 \uc808\uc81c\ud588\uc2b5\ub2c8\ub2e4. (Controlled bass boost because the source already has strong low energy.)"],
    ["Preserved low-end weight while avoiding rumble in restore mode.", "\ubcf5\uc6d0 \ubaa8\ub4dc\uc5d0\uc11c \ub7fc\ube14\uc740 \uc904\uc774\ub418 \uc800\uc5ed \ubb34\uac8c\uac10\uc740 \ubcf4\uc874\ud588\uc2b5\ub2c8\ub2e4. (Preserved low-end weight while avoiding rumble in restore mode.)"],
    ["Applied a small mid cut because the mix is mid-heavy.", "\ubbf9\uc2a4\uc758 \uc911\uc5ed\uc774 \ub450\ud130\uc6cc \uc791\uc740 \uc911\uc5ed \ucef7\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied a small mid cut because the mix is mid-heavy.)"],
    ["Added high-band lift because the source sounds dull.", "\uc18c\uc2a4\uac00 \ub2f5\ub2f5\ud574 \uace0\uc5ed \ub9ac\ud504\ud2b8\ub97c \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Added high-band lift because the source sounds dull.)"],
    ["Reduced high band slightly because the source appears noisy.", "\uc18c\uc2a4\uc758 \ub178\uc774\uc988\uac00 \uc788\uc5b4 \uace0\uc5ed\uc744 \uc57d\uac04 \uc904\uc600\uc2b5\ub2c8\ub2e4. (Reduced high band slightly because the source appears noisy.)"],
    ["Added compression because loudness is very low.", "\ud3c9\uade0 \uc74c\ub7c9\uc774 \ub9e4\uc6b0 \ub0ae\uc544 \ucef4\ud504\ub808\uc158\uc744 \ub354\ud588\uc2b5\ub2c8\ub2e4. (Added compression because loudness is very low.)"],
    ["Added gentle compression because dynamic range is wide.", "\ub2e4\uc774\ub0b4\ubbf9 \ub808\uc778\uc9c0\uac00 \ub113\uc5b4 \ubd80\ub4dc\ub7ec\uc6b4 \ucef4\ud504\ub808\uc158\uc744 \ub354\ud588\uc2b5\ub2c8\ub2e4. (Added gentle compression because dynamic range is wide.)"],
    ["Detected clipping; avoided aggressive gain changes.", "\ud074\ub9ac\ud551\uc774 \uac10\uc9c0\ub418\uc5b4 \uacfc\uaca9\ud55c \uac8c\uc778 \ubcc0\ud654\ub97c \ud53c\ud588\uc2b5\ub2c8\ub2e4. (Detected clipping; avoided aggressive gain changes.)"],
    ["Selected model-assisted upsample mode for low sample-rate material.", "\ub0ae\uc740 \uc0d8\ud50c\ub808\uc774\ud2b8 \uc18c\uc2a4\uc5d0 \ubaa8\ub378 \ubcf4\uc870 \uc5c5\uc0d8\ud50c\ub9c1\uc744 \uc120\ud0dd\ud588\uc2b5\ub2c8\ub2e4. (Selected model-assisted upsample mode for low sample-rate material.)"],
    ["Stereo-safe mid/side processing will preserve staging cues.", "\uc2a4\ud14c\uc774\uc9d5 \ub2e8\uc11c\ub97c \uc720\uc9c0\ud558\uae30 \uc704\ud574 \uc2a4\ud14c\ub808\uc624 \uc138\uc774\ud504 \ubbf8\ub4dc/\uc0ac\uc774\ub4dc \ucc98\ub9ac\ub97c \uc801\uc6a9\ud569\ub2c8\ub2e4. (Stereo-safe mid/side processing will preserve staging cues.)"],
    ["Reduced enhancement on phase-sensitive stereo material.", "\uc704\uc0c1\uc5d0 \ubbfc\uac10\ud55c \uc2a4\ud14c\ub808\uc624 \uc18c\uc2a4\uc5d0\uc11c \ud5a5\uc0c1 \uac15\ub3c4\ub97c \uc904\uc600\uc2b5\ub2c8\ub2e4. (Reduced enhancement on phase-sensitive stereo material.)"],
    ["Short files provide less reliable analysis.", "\uc9e7\uc740 \ud30c\uc77c\uc740 \ubd84\uc11d \uc2e0\ub8b0\ub3c4\uac00 \ub0ae\uc744 \uc218 \uc788\uc2b5\ub2c8\ub2e4. (Short files provide less reliable analysis.)"],
    ["The source is already balanced; applied minimal cleanup.", "\uc18c\uc2a4\uac00 \uc774\ubbf8 \uade0\ud615\uc801\uc774\ub77c \ucd5c\uc18c\ud55c\uc758 \uc815\ub9ac\ub9cc \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (The source is already balanced; applied minimal cleanup.)"],
    ["Matched target loudness to the source loudness.", "\ubaa9\ud45c \uc74c\ub7c9\uc744 \uc6d0\ubcf8 \uc74c\ub7c9\uc5d0 \ub9de\ucd94\uc5c8\uc2b5\ub2c8\ub2e4. (Matched target loudness to the source loudness.)"],
    ["Skipped denoise stage by user selection.", "\uc0ac\uc6a9\uc790 \uc120\ud0dd\uc5d0 \ub530\ub77c \ub178\uc774\uc988 \uc81c\uac70 \ub2e8\uacc4\ub97c \uac74\ub108\ub6f0\uc5c8\uc2b5\ub2c8\ub2e4. (Skipped denoise stage by user selection.)"],
    ["Used fast denoise path to keep full-track processing responsive.", "\uc804\uccb4 \ud2b8\ub799 \ucc98\ub9ac \uc18d\ub3c4\ub97c \uc720\uc9c0\ud558\uae30 \uc704\ud574 \ube60\ub978 \ub178\uc774\uc988 \ucc98\ub9ac \uacbd\ub85c\ub97c \uc0ac\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Used fast denoise path to keep full-track processing responsive.)"],
    ["Applied Demucs 2-stem separation before final remix.", "Demucs 2-stem \ubd84\ub9ac \ud6c4 \ucd5c\uc885 \uc7ac\ud569\uc131\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied Demucs 2-stem separation before final remix.)"],
    ["Applied Demucs 2stem fast separation before final remix.", "Demucs 2-stem fast \ubd84\ub9ac \ud6c4 \ucd5c\uc885 \uc7ac\ud569\uc131\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied Demucs 2-stem fast separation.)"],
    ["Applied Demucs 2stem balanced separation before final remix.", "Demucs 2-stem balanced \ubd84\ub9ac \ud6c4 \ucd5c\uc885 \uc7ac\ud569\uc131\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied Demucs 2-stem balanced separation.)"],
    ["Applied Demucs 2stem precision separation before final remix.", "Demucs 2-stem precision \ubd84\ub9ac \ud6c4 \ucd5c\uc885 \uc7ac\ud569\uc131\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied Demucs 2-stem precision separation.)"],
    ["Applied Demucs 4stem fast separation before final remix.", "Demucs 4-stem fast \ubd84\ub9ac \ud6c4 \ucd5c\uc885 \uc7ac\ud569\uc131\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied Demucs 4-stem fast separation.)"],
    ["Applied Demucs 4stem balanced separation before final remix.", "Demucs 4-stem balanced \ubd84\ub9ac \ud6c4 \ucd5c\uc885 \uc7ac\ud569\uc131\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied Demucs 4-stem balanced separation.)"],
    ["Applied Demucs 4stem precision separation before final remix.", "Demucs 4-stem precision \ubd84\ub9ac \ud6c4 \ucd5c\uc885 \uc7ac\ud569\uc131\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied Demucs 4-stem precision separation.)"],
    ["Vocals used Voice Focus while instrumental used the selected listening target.", "\ubcf4\uceec\uc740 Voice Focus, \ubc18\uc8fc\ub294 \uc120\ud0dd\ud55c \uccad\uac10 \ubaa9\ud45c\ub85c \uac1c\uc120\ud588\uc2b5\ub2c8\ub2e4. (Vocals used Voice Focus while instrumental used the selected listening target.)"],
    ["Stem processing uses conservative intensity to limit separation artifacts.", "Stem \ubd84\ub9ac artifact\ub97c \uc904\uc774\uae30 \uc704\ud574 \ubcf4\uc218\uc801\uc778 \ucc98\ub9ac \uac15\ub3c4\ub97c \uc0ac\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Stem processing uses conservative intensity.)"],
    ["Vocal stem uses bleed-safe conservative tone shaping.", "\ubcf4\uceec stem\uc740 \uc545\uae30 \uc794\uc5ec\uc74c\uc774 \uac15\uc870\ub418\uc9c0 \uc54a\ub3c4\ub85d \ubcf4\uc218\uc801\uc73c\ub85c \ud1a4\uc744 \uc815\ub9ac\ud569\ub2c8\ub2e4. (Vocal stem uses bleed-safe tone shaping.)"],
    ["Applied conservative vocal stem bleed cleanup.", "\ubcf4\uceec stem\uc5d0 \ub0a8\uc740 \uc545\uae30 \uc794\uc5ec\uc74c\uc744 \uc904\uc774\uae30 \uc704\ud574 \uc57d\ud55c bleed cleanup\uc744 \uc801\uc6a9\ud588\uc2b5\ub2c8\ub2e4. (Applied conservative vocal stem bleed cleanup.)"],
    ["Preserved low-level source residual during stem remix.", "\uc6d0\ubcf8\uc758 \uc794\ud5a5\uacfc \uacf5\uac04\uac10\uc744 \uc720\uc9c0\ud558\uae30 \uc704\ud574 stem \uc7ac\ud569\uc131 \uc2dc \uc57d\ud55c residual\uc744 \ubcf4\uc874\ud588\uc2b5\ub2c8\ub2e4. (Preserved low-level source residual during stem remix.)"],
    ["Precision stem quality failed; used balanced fallback.", "Precision stem \ud488\uc9c8 \ubaa8\ub4dc\uac00 \uc2e4\ud328\ud574 balanced \ubaa8\ub4dc\ub85c \uc790\ub3d9 \uc804\ud658\ud588\uc2b5\ub2c8\ub2e4. (Precision stem quality failed; used balanced fallback.)"],
    ["4-stem balanced fallback failed; used 2-stem fallback.", "4-stem balanced fallback\uc774 \uc2e4\ud328\ud574 2-stem\uc73c\ub85c \uc790\ub3d9 \uc804\ud658\ud588\uc2b5\ub2c8\ub2e4. (Used 2-stem fallback.)"],
    ["Precision stem quality is opt-in only; used balanced policy.", "Precision stem? ??? opt-in??? ???? balanced ???? ??????. (Precision stem is opt-in only; used balanced policy.)"],
    ["Vocal stem tuning prioritizes clarity, sibilance safety, and bleed control.", "?? stem? ???, ??? ??, bleed ??? ??????. (Vocal stem prioritizes clarity and bleed control.)"],
    ["Drum stem tuning preserves transient impact and avoids over-compression.", "?? stem? ??? transient? ???? ???? ?????. (Drum stem preserves transient impact.)"],
    ["Bass stem tuning focuses low-end weight while protecting phase and headroom.", "??? stem? ?? ???? ??? ??? ???? ??????. (Bass stem protects phase and headroom.)"],
    ["Other stem tuning preserves ambience and stereo cues.", "?? ?? stem? ??, ???, ???? ??? ??????. (Other stem preserves ambience and stereo cues.)"],
    ["Instrumental stem tuning keeps backing balance and staging stable.", "?? stem? ???? ???? ???? ??????. (Instrumental stem keeps backing balance stable.)"],
    ["Adaptive AI amount reduced processing strength based on source quality.", "소스 품질을 기준으로 AI 처리 강도를 자동으로 낮췄습니다. (Adaptive AI amount reduced processing strength.)"],
    ["Adaptive AI amount increased processing strength based on source quality.", "소스 품질을 기준으로 AI 처리 강도를 자동으로 높였습니다. (Adaptive AI amount increased processing strength.)"],
    ["Stem artifact risk reduced per-stem processing strength.", "Stem artifact 위험을 감지해 해당 stem의 처리 강도를 낮췄습니다. (Reduced per-stem processing strength.)"],
]);

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const processBtn = document.getElementById("process-btn");
const versionPill = document.getElementById("version-pill");
const groqStatusPill = document.getElementById("groq-status-pill");
const intensitySlider = document.getElementById("intensity-slider");
const intensityValue = document.getElementById("intensity-value");
const outputSampleRate = document.getElementById("output-sample-rate");
const outputBitDepth = document.getElementById("output-bit-depth");
const manualDspEnable = document.getElementById("manual-dsp-enable");
const manualDspGrid = document.getElementById("manual-dsp-grid");
const stemSeparationMode = document.getElementById("opt-stem-mode");
const stemQualityMode = document.getElementById("opt-stem-quality");
const targetComboWarning = document.getElementById("target-combo-warning");
const aiIntentTitle = document.getElementById("ai-intent-title");
const aiIntentCopy = document.getElementById("ai-intent-copy");
const aiIntentTargets = document.getElementById("ai-intent-targets");
const aiIntentSafety = document.getElementById("ai-intent-safety");
const aiIntentOutput = document.getElementById("ai-intent-output");
const stemPolicyCopy = document.getElementById("stem-policy-copy");
const resultZone = document.getElementById("result-zone");
const abToggleBtn = document.getElementById("ab-toggle-btn");
const levelMatchToggle = document.getElementById("level-match-toggle");
const masterPlayer = document.getElementById("master-player");
const audioOriginal = document.getElementById("audio-original");
const audioEnhanced = document.getElementById("audio-enhanced");
const downloadLink = document.getElementById("download-link");
const stemDownloadPanel = document.getElementById("stem-download-panel");
const stemDownloadList = document.getElementById("stem-download-list");
const singlePreviewPanel = document.getElementById("single-preview-panel");
const presetPreviewBtn = document.getElementById("preset-preview-btn");
const presetPreviewResults = document.getElementById("preset-preview-results");
const batchResultBox = document.getElementById("batch-result-box");
const batchFeedbackText = document.getElementById("batch-feedback-text");
const batchSummaryList = document.getElementById("batch-summary-list");
const batchZipLink = document.getElementById("batch-zip-link");
const nowPlayingText = document.getElementById("now-playing-text");
const labelOriginal = document.getElementById("label-original");
const labelEnhanced = document.getElementById("label-enhanced");
const waveformOriginal = document.getElementById("waveform-original");
const waveformEnhanced = document.getElementById("waveform-enhanced");
const waveformRowOriginal = document.getElementById("waveform-row-original");
const waveformRowEnhanced = document.getElementById("waveform-row-enhanced");
const aiAnalysisBox = document.getElementById("ai-analysis-box");
const aiFeedbackText = document.getElementById("ai-feedback-text");
const aiDecisionList = document.getElementById("ai-decision-list");
const reportDetailList = document.getElementById("report-detail-list");
const processingPanel = document.getElementById("processing-panel");
const processingStage = document.getElementById("processing-stage");
const processingPercent = document.getElementById("processing-percent");
const processingBar = document.getElementById("processing-bar");
const processingStageItems = Array.from(document.querySelectorAll("#processing-stages li"));
const cancelProcessBtn = document.getElementById("cancel-process-btn");
const settingsBtn = document.getElementById("settings-btn");
const settingsModal = document.getElementById("settings-modal");
const groqApiKeyInput = document.getElementById("groq-api-key");
const modalSave = document.getElementById("modal-save");
const modalCancel = document.getElementById("modal-cancel");
const themePresetSelect = document.getElementById("theme-preset");
const themeBgInput = document.getElementById("theme-bg");
const themeAccentInput = document.getElementById("theme-accent");
const themeAccent2Input = document.getElementById("theme-accent-2");
const themeResetBtn = document.getElementById("theme-reset");

function normalizeHex(value, fallback = "#00f0ff") {
    const raw = String(value || "").trim();
    if (/^#[0-9a-f]{6}$/i.test(raw)) {
        return raw.toLowerCase();
    }
    return fallback;
}

function hexToRgb(hex) {
    const normalized = normalizeHex(hex).slice(1);
    return {
        r: parseInt(normalized.slice(0, 2), 16),
        g: parseInt(normalized.slice(2, 4), 16),
        b: parseInt(normalized.slice(4, 6), 16),
    };
}

function rgbToHex({ r, g, b }) {
    return `#${[r, g, b].map((value) => Math.round(value).toString(16).padStart(2, "0")).join("")}`;
}

function rgbString(hex) {
    const { r, g, b } = hexToRgb(hex);
    return `${r}, ${g}, ${b}`;
}

function mixHex(from, to, amount = 0.5) {
    const a = hexToRgb(from);
    const b = hexToRgb(to);
    const t = Math.max(0, Math.min(1, amount));
    return rgbToHex({
        r: a.r * (1 - t) + b.r * t,
        g: a.g * (1 - t) + b.g * t,
        b: a.b * (1 - t) + b.b * t,
    });
}

function relativeLuminance(hex) {
    const { r, g, b } = hexToRgb(hex);
    const channel = (value) => {
        const normalized = value / 255;
        return normalized <= 0.03928
            ? normalized / 12.92
            : ((normalized + 0.055) / 1.055) ** 2.4;
    };
    return (0.2126 * channel(r)) + (0.7152 * channel(g)) + (0.0722 * channel(b));
}

function buildTheme(seed) {
    const bg = normalizeHex(seed.bg, themePresets["neon-blue"].bg);
    const accent = normalizeHex(seed.accent, themePresets["neon-blue"].accent);
    const accent2 = normalizeHex(seed.accent2, themePresets["neon-blue"].accent2);
    const isLight = relativeLuminance(bg) > 0.46;
    const textColor = isLight ? "#07111f" : "#eef8ff";
    const anchor = isLight ? "#ffffff" : "#12203a";
    const panel = mixHex(bg, anchor, isLight ? 0.58 : 0.62);
    const panelSoft = mixHex(bg, accent, isLight ? 0.08 : 0.14);
    const muted = mixHex(textColor, bg, isLight ? 0.48 : 0.34);
    const accent3 = mixHex(accent, accent2, 0.38);

    return {
        bg,
        panel,
        panelSoft,
        text: textColor,
        muted,
        accent,
        accent2,
        accent3,
    };
}

function setCssVar(name, value) {
    document.documentElement.style.setProperty(name, value);
}

function applyTheme(themeConfig, { persist = true, syncControls = true } = {}) {
    const preset = themeConfig?.preset || "neon-blue";
    const seed = preset === "custom"
        ? {
            bg: themeConfig.bg,
            accent: themeConfig.accent,
            accent2: themeConfig.accent2,
        }
        : (themePresets[preset] || themePresets["neon-blue"]);
    const theme = buildTheme(seed);

    setCssVar("--bg", theme.bg);
    setCssVar("--panel", theme.panel);
    setCssVar("--panel-soft", theme.panelSoft);
    setCssVar("--line", `rgba(${rgbString(theme.accent)}, 0.24)`);
    setCssVar("--text", theme.text);
    setCssVar("--muted", theme.muted);
    setCssVar("--accent", theme.accent);
    setCssVar("--accent-2", theme.accent2);
    setCssVar("--accent-3", theme.accent3);
    setCssVar("--accent-rgb", rgbString(theme.accent));
    setCssVar("--accent-2-rgb", rgbString(theme.accent2));
    setCssVar("--accent-3-rgb", rgbString(theme.accent3));
    setCssVar("--bg-rgb", rgbString(theme.bg));
    setCssVar("--panel-rgb", rgbString(theme.panel));

    if (syncControls) {
        syncThemeControls({
            preset,
            bg: theme.bg,
            accent: theme.accent,
            accent2: theme.accent2,
        });
    }
    if (persist) {
        localStorage.setItem(THEME_STORAGE_KEY, JSON.stringify({
            preset,
            bg: theme.bg,
            accent: theme.accent,
            accent2: theme.accent2,
        }));
    }
    renderWaveforms();
}

function loadSavedTheme() {
    try {
        const saved = JSON.parse(localStorage.getItem(THEME_STORAGE_KEY) || "null");
        if (saved && typeof saved === "object") {
            return saved;
        }
    } catch {}
    return { preset: "neon-blue", ...themePresets["neon-blue"] };
}

function syncThemeControls(themeConfig) {
    if (themePresetSelect) {
        themePresetSelect.value = themeConfig.preset || "neon-blue";
    }
    if (themeBgInput) {
        themeBgInput.value = normalizeHex(themeConfig.bg, themePresets["neon-blue"].bg);
    }
    if (themeAccentInput) {
        themeAccentInput.value = normalizeHex(themeConfig.accent, themePresets["neon-blue"].accent);
    }
    if (themeAccent2Input) {
        themeAccent2Input.value = normalizeHex(themeConfig.accent2, themePresets["neon-blue"].accent2);
    }
}

function getCurrentThemeFromInputs() {
    return {
        preset: "custom",
        bg: themeBgInput?.value || themePresets["neon-blue"].bg,
        accent: themeAccentInput?.value || themePresets["neon-blue"].accent,
        accent2: themeAccent2Input?.value || themePresets["neon-blue"].accent2,
    };
}

function initializeThemeControls() {
    applyTheme(loadSavedTheme(), { persist: false });

    themePresetSelect?.addEventListener("change", () => {
        const preset = themePresetSelect.value;
        applyTheme({
            preset,
            ...(themePresets[preset] || getCurrentThemeFromInputs()),
        });
    });

    [themeBgInput, themeAccentInput, themeAccent2Input].forEach((input) => {
        input?.addEventListener("input", () => {
            applyTheme(getCurrentThemeFromInputs(), { syncControls: false });
            if (themePresetSelect) {
                themePresetSelect.value = "custom";
            }
        });
    });

    themeResetBtn?.addEventListener("click", () => {
        applyTheme({ preset: "neon-blue", ...themePresets["neon-blue"] });
    });
}

settingsBtn.addEventListener("click", () => {
    groqApiKeyInput.value = localStorage.getItem("groq_api_key") || "";
    settingsModal.classList.add("active");
});

modalCancel.addEventListener("click", () => {
    settingsModal.classList.remove("active");
});

modalSave.addEventListener("click", () => {
    localStorage.setItem("groq_api_key", groqApiKeyInput.value.trim());
    updateGroqStatusPill();
    settingsModal.classList.remove("active");
});

settingsModal.addEventListener("click", (event) => {
    if (event.target === settingsModal) {
        settingsModal.classList.remove("active");
    }
});

function updateGroqStatusPill() {
    if (!groqStatusPill) {
        return;
    }
    const key = (localStorage.getItem("groq_api_key") || "").trim();
    groqStatusPill.classList.remove("on", "warn");
    if (!key) {
        groqStatusPill.textContent = "Groq 미설정";
        groqStatusPill.title = "Groq API 키가 저장되어 있지 않습니다. Groq API 설정 버튼에서 키를 저장하세요.";
        return;
    }
    if (key.startsWith("gsk_")) {
        groqStatusPill.textContent = "Groq 활성화";
        groqStatusPill.title = "Groq API 키가 이 브라우저에 저장되어 있습니다.";
        groqStatusPill.classList.add("on");
        return;
    }
    groqStatusPill.textContent = "Groq 확인 필요";
    groqStatusPill.title = "저장된 키가 Groq API 키 형식(gsk_...)처럼 보이지 않습니다.";
    groqStatusPill.classList.add("warn");
}

window.addEventListener("storage", (event) => {
    if (event.key === "groq_api_key") {
        updateGroqStatusPill();
    }
    if (event.key === THEME_STORAGE_KEY) {
        applyTheme(loadSavedTheme(), { persist: false });
    }
});

initializeThemeControls();
loadAppVersion().catch(() => {});
updateGroqStatusPill();

dropZone.addEventListener("click", (event) => {
    if (event.target === fileInput) {
        return;
    }
    ensureFileInputMounted();
    fileInput.click();
});

dropZone.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropZone.classList.add("dragover");
});

dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
});

dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragover");

    if (event.dataTransfer.files.length > 0) {
        handleFileSelect(event.dataTransfer.files);
    }
});

fileInput.addEventListener("change", (event) => {
    if (event.target.files.length > 0) {
        handleFileSelect(event.target.files);
    }
});

function handleFileSelect(fileList) {
    const validExtensions = [".wav", ".mp3", ".m4a", ".flac"];
    const files = Array.from(fileList);
    const invalidFile = files.find((file) => !validExtensions.some((ext) => file.name.toLowerCase().endsWith(ext)));

    if (invalidFile) {
        alert(text.invalidFile);
        return;
    }

    revokeSelectedAudioUrls();

    selectedAudioFiles = files;
    selectedAudioUrls = files.map((file) => URL.createObjectURL(file));
    originalAudioBlob = files[0];
    originalAudioUrl = selectedAudioUrls[0];
    enhancedAudioUrl = null;
    batchResults = [];
    currentReport = null;
    resetLevelMatchGains();
    applyPlaybackVolume();
    resetWaveforms();

    const totalSize = files.reduce((sum, file) => sum + file.size, 0);
    const fileRows = files.slice(0, 8).map((file) => `
        <li>
            <span>${escapeHtml(file.name)}</span>
            <span>${formatFileSize(file.size)}</span>
        </li>
    `).join("");
    const extraCount = files.length > 8 ? `<li><span>+${files.length - 8} more</span><span></span></li>` : "";

    dropZone.innerHTML = `
        <div class="file-info">
            <p>${files.length === 1 ? escapeHtml(files[0].name) : `${files.length} files selected`}</p>
            <p class="file-size">${formatFileSize(totalSize)} total</p>
            <ul class="selected-file-list">${fileRows}${extraCount}</ul>
        </div>
    `;
    ensureFileInputMounted();

    processBtn.disabled = false;
    if (presetPreviewBtn) {
        presetPreviewBtn.disabled = false;
    }
    if (presetPreviewResults) {
        presetPreviewResults.classList.add("hidden");
        presetPreviewResults.innerHTML = "";
    }
    processBtn.querySelector(".btn-text").textContent = files.length > 1 ? text.batchStart : text.start;
    resultZone.classList.add("hidden");
    aiAnalysisBox.classList.add("hidden");
    batchResultBox.classList.add("hidden");
    resetProcessingProgress();
}

function ensureFileInputMounted() {
    if (!fileInput.isConnected || fileInput.parentElement !== dropZone) {
        fileInput.hidden = true;
        dropZone.prepend(fileInput);
    }
}

intensitySlider.addEventListener("input", (event) => {
    intensityValue.textContent = `${event.target.value}%`;
    updateAiIntentPanel();
});

const manualControls = [
    { id: "manual-lowcut", key: "lowcut_offset_hz", suffix: " Hz", decimals: 0, signed: true },
    { id: "manual-low", key: "low_boost_delta_db", suffix: " dB", decimals: 1, signed: true },
    { id: "manual-mid", key: "mid_delta_db", suffix: " dB", decimals: 1, signed: true },
    { id: "manual-high", key: "high_boost_delta_db", suffix: " dB", decimals: 1, signed: true },
    { id: "manual-compress", key: "compress_delta", suffix: ":1", decimals: 2, signed: true },
    { id: "manual-lufs", key: "target_lufs_delta", suffix: " LUFS", decimals: 1, signed: true },
    { id: "manual-exciter", key: "exciter_delta", suffix: "", decimals: 2, signed: true },
    { id: "manual-saturation", key: "saturation_delta", suffix: "", decimals: 2, signed: true },
];

manualControls.forEach((control) => {
    const input = document.getElementById(control.id);
    input.addEventListener("input", () => {
        updateManualControlOutput(control);
        updateAiIntentPanel();
    });
    updateManualControlOutput(control);
});

manualDspEnable.addEventListener("change", () => {
    manualDspGrid.classList.toggle("manual-disabled", !manualDspEnable.checked);
    updateAiIntentPanel();
});

document.querySelectorAll('input[name="target-mode"]').forEach((input) => {
    input.addEventListener("change", () => {
        updateTargetComboWarning();
        updateAiIntentPanel();
    });
});
updateTargetComboWarning();

document.querySelectorAll('input[name="volume-mode"]').forEach((input) => {
    input.addEventListener("change", updateAiIntentPanel);
});
document.getElementById("opt-denoise").addEventListener("change", updateAiIntentPanel);
stemSeparationMode.addEventListener("change", () => {
    updateStemQualityPolicy();
    updateAiIntentPanel();
});
stemQualityMode.addEventListener("change", () => {
    updateStemQualityPolicy();
    updateAiIntentPanel();
});
outputSampleRate.addEventListener("change", updateAiIntentPanel);
outputBitDepth.addEventListener("change", updateAiIntentPanel);
updateStemQualityPolicy();
updateAiIntentPanel();

processBtn.addEventListener("click", async () => {
    if (selectedAudioFiles.length === 0 || isProcessing) {
        return;
    }

    isProcessing = true;
    processBtn.disabled = true;

    const spinner = processBtn.querySelector(".spinner");
    const btnText = processBtn.querySelector(".btn-text");

    spinner.classList.remove("hidden");

    try {
        const isBatch = selectedAudioFiles.length > 1;
        btnText.textContent = isBatch ? text.batchProcessing : text.processing;
        startProcessingProgress();
        const result = isBatch ? await processBatchAudio() : await processSingleAudio();
        completeProcessingProgress();

        if (isBatch) {
            renderBatchResult(result);
        } else {
            renderSingleResult(result);
        }
    } catch (error) {
        failProcessingProgress();
        alert(`${text.error} ${error.message}`);
    } finally {
        isProcessing = false;
        processBtn.disabled = false;
        spinner.classList.add("hidden");
        btnText.textContent = selectedAudioFiles.length > 1 ? text.batchStart : text.start;
        activeJobId = null;
        activeJobCancelled = false;
        cancelProcessBtn?.classList.add("hidden");
    }
});

cancelProcessBtn?.addEventListener("click", async () => {
    if (!activeJobId || activeJobCancelled) {
        return;
    }
    activeJobCancelled = true;
    cancelProcessBtn.disabled = true;
    processingStage.textContent = "취소 요청 중... (Cancelling...)";
    try {
        await fetch(apiUrl(`/api/jobs/${activeJobId}`), { method: "DELETE" });
    } catch (error) {
        console.warn("Cancel request failed", error);
    }
});

presetPreviewResults?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-preview-target]");
    if (!button) {
        return;
    }
    applyPreviewTarget(button.dataset.previewTarget);
});
async function processSingleAudio() {
    const processForm = createProcessingForm();
    processForm.append("file", originalAudioBlob);

    return startAndWaitForJob("/api/jobs/process", processForm);
}

async function processBatchAudio() {
    const processForm = createProcessingForm();
    selectedAudioFiles.forEach((file) => {
        processForm.append("files", file);
    });

    return startAndWaitForJob("/api/jobs/process-batch", processForm);
}

async function startAndWaitForJob(endpoint, formData) {
    const startResponse = await fetch(apiUrl(endpoint), {
        method: "POST",
        body: formData,
    });

    if (!startResponse.ok) {
        throw new Error(await readErrorMessage(startResponse, text.processFailed));
    }

    const job = await startResponse.json();
    activeJobId = job.job_id;
    activeJobCancelled = false;
    if (cancelProcessBtn) {
        cancelProcessBtn.disabled = false;
        cancelProcessBtn.classList.remove("hidden");
    }
    renderJobProgress(job);
    return waitForJob(job.job_id);
}

async function waitForJob(jobId) {
    while (true) {
        await delay(1200);
        const response = await fetch(apiUrl(`/api/jobs/${jobId}`));
        if (!response.ok) {
            throw new Error(await readErrorMessage(response, text.processFailed));
        }
        const job = await response.json();
        renderJobProgress(job);

        if (job.status === "completed") {
            return job.result;
        }
        if (job.status === "cancelled") {
            throw new Error("처리가 취소되었습니다. (Processing cancelled.)");
        }
        if (job.status === "failed") {
            throw new Error(job.error || text.processFailed);
        }
    }
}

function renderJobProgress(job) {
    const percent = Number.isFinite(Number(job.percent)) ? Number(job.percent) : progressValue;
    setProcessingProgress(percent, job.stage || null);
    processingPanel.classList.toggle("is-waiting", job.status === "queued" || job.status === "running");
    if (cancelProcessBtn) {
        cancelProcessBtn.classList.toggle("hidden", !["queued", "running"].includes(job.status));
        cancelProcessBtn.disabled = activeJobCancelled || !["queued", "running"].includes(job.status);
    }
}

function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function createProcessingForm() {
    const processForm = new FormData();
    processForm.append("target", getSelectedTarget());
    processForm.append("intensity", String(Number(intensitySlider.value) / 100));
    processForm.append("use_denoise", document.getElementById("opt-denoise").checked ? "true" : "false");
    processForm.append("stem_separation", getSelectedStemMode());
    processForm.append("stem_quality", getSelectedStemQualityMode());
    processForm.append("volume_mode", getSelectedVolumeMode());
    processForm.append("output_sample_rate", outputSampleRate.value);
    processForm.append("output_bit_depth", outputBitDepth.value);
    const manualParams = getManualDspParams();
    if (manualParams) {
        processForm.append("dsp_params", JSON.stringify(manualParams));
    }
    return processForm;
}

function createPreviewForm() {
    const form = new FormData();
    const previewTargets = getPreviewTargetList();
    form.append("file", selectedAudioFiles[0]);
    form.append("targets", previewTargets.join(","));
    form.append("intensity", String(Number(intensitySlider.value) / 100));
    form.append("use_denoise", document.getElementById("opt-denoise").checked ? "true" : "false");
    form.append("volume_mode", getSelectedVolumeMode());
    form.append("output_sample_rate", outputSampleRate.value);
    form.append("output_bit_depth", outputBitDepth.value);
    return form;
}

function getPreviewTargetList() {
    const selected = getSelectedTargetList();
    if (selected.length >= 2) {
        return selected.slice(0, 3);
    }
    const first = selected[0] || "hifi_clean";
    const defaults = [first, "hifi_clean", "warm_analog", "bass_boost", "hifi_bright"];
    return Array.from(new Set(defaults)).slice(0, 3);
}

async function createPresetPreviews() {
    if (!selectedAudioFiles.length || !presetPreviewBtn || !presetPreviewResults) {
        return;
    }
    presetPreviewBtn.disabled = true;
    presetPreviewResults.classList.remove("hidden");
    presetPreviewResults.innerHTML = `<div class="preset-preview-item"><div><strong>미리듣기 생성 중</strong><span>(Creating preset previews...)</span></div></div>`;

    try {
        const response = await fetch(apiUrl("/api/preview-targets"), {
            method: "POST",
            body: createPreviewForm(),
        });
        if (!response.ok) {
            throw new Error(await readErrorMessage(response, "Preset preview failed"));
        }
        const result = await response.json();
        renderPresetPreviews(result);
    } catch (error) {
        presetPreviewResults.innerHTML = `<div class="preset-preview-item"><div><strong>미리듣기 실패</strong><span>${escapeHtml(error.message || String(error))}</span></div></div>`;
    } finally {
        presetPreviewBtn.disabled = false;
    }
}

function renderPresetPreviews(result) {
    if (!presetPreviewResults) {
        return;
    }
    const items = Array.isArray(result.items) ? result.items : [];
    if (!items.length) {
        presetPreviewResults.innerHTML = `<div class="preset-preview-item"><div><strong>결과 없음</strong><span>(No preview results.)</span></div></div>`;
        return;
    }
    presetPreviewResults.innerHTML = items.map((item) => {
        const target = item.target || "hifi_clean";
        const label = translateTarget(target);
        const score = Number.isFinite(Number(item.validation_score)) ? `${Number(item.validation_score).toFixed(0)}점` : "-";
        const overall = item.validation_overall || "review";
        const risk = item.validation_risk || {};
        const penalty = Number.isFinite(Number(risk.penalties)) ? Number(risk.penalties).toFixed(1) : "0.0";
        const url = resolveApiAssetUrl(item.download_url);
        const filename = item.filename || `preview_${target}.wav`;
        return `
            <div class="preset-preview-item">
                <div>
                    <strong>${escapeHtml(label)}</strong>
                    <span>검증 ${score} / ${escapeHtml(overall)} · Risk ${escapeHtml(penalty)} · LUFS ${formatNumber(item.lufs, 1)} · TP ${formatNumber(item.true_peak_db, 1)} dBTP</span>
                </div>
                <div class="preset-preview-actions">
                    <button type="button" class="preset-preview-select" data-preview-target="${escapeHtml(target)}">현재 목표로 적용</button>
                    <a href="${escapeHtml(url)}" download="${escapeHtml(filename)}">재생/저장</a>
                </div>
            </div>
        `;
    }).join("");
}
function applyPreviewTarget(target) {
    if (!target) {
        return;
    }
    const inputs = Array.from(document.querySelectorAll('input[name="target-mode"]'));
    let matched = false;
    inputs.forEach((input) => {
        input.checked = input.value === target;
        matched = matched || input.checked;
    });
    if (!matched) {
        const fallback = inputs.find((input) => input.value === "hifi_clean");
        if (fallback) {
            fallback.checked = true;
        }
    }
    updateTargetComboWarning();
    updateAiIntentPanel();
}
function getManualDspParams() {
    if (!manualDspEnable.checked) {
        return null;
    }

    const params = {
        manual_mode: "fine_tune",
        limiter_ceiling_db: -1.5,
        normalize: true,
    };
    manualControls.forEach((control) => {
        params[control.key] = Number(document.getElementById(control.id).value);
    });
    return params;
}

function updateManualControlOutput(control) {
    const input = document.getElementById(control.id);
    const output = input.nextElementSibling;
    const value = Number(input.value);
    const formatted = control.signed
        ? formatSigned(value, control.decimals)
        : value.toFixed(control.decimals);
    output.textContent = `${formatted}${control.suffix}`;
}

function renderSingleResult(result, sourceFile = originalAudioBlob, sourceUrl = originalAudioUrl, options = {}) {
    processingPanel.classList.add("hidden");
    originalAudioBlob = sourceFile;
    originalAudioUrl = sourceUrl;
    currentReport = result.report;
    enhancedAudioUrl = resolveApiAssetUrl(result.download_url);

    audioOriginal.src = originalAudioUrl;
    audioEnhanced.src = enhancedAudioUrl;
    masterPlayer.src = originalAudioUrl;
    updateLevelMatchGains(currentReport);
    applyPlaybackVolume();

    downloadLink.href = enhancedAudioUrl;
    downloadLink.download = result.filename || `enhanced_${originalAudioBlob.name.replace(/\.[^.]+$/, ".wav")}`;
    setDownloadLabel("\ud5a5\uc0c1\ub41c \ud30c\uc77c \ub2e4\uc6b4\ub85c\ub4dc", "Download enhanced file");
    batchZipLink.classList.toggle("hidden", !options.keepBatchVisible);
    renderStemDownloads(result.stem_downloads || []);

    singlePreviewPanel.classList.remove("hidden");
    batchResultBox.classList.toggle("hidden", !options.keepBatchVisible);
    renderReport(currentReport);
    resultZone.classList.remove("hidden");
    setABMode("original");
    prepareWaveformComparison().catch(() => {
        resetWaveforms();
    });
}

function renderBatchResult(result) {
    processingPanel.classList.add("hidden");
    currentReport = null;
    enhancedAudioUrl = null;
    batchResults = result.summary || [];
    resetLevelMatchGains();
    masterPlayer.pause();
    masterPlayer.removeAttribute("src");
    audioOriginal.removeAttribute("src");
    audioEnhanced.removeAttribute("src");
    resetWaveforms();

    batchZipLink.href = resolveApiAssetUrl(result.download_url);
    batchZipLink.download = result.filename || "ResonixAI_Batch.zip";
    batchZipLink.classList.remove("hidden");

    singlePreviewPanel.classList.add("hidden");
    aiAnalysisBox.classList.add("hidden");
    batchFeedbackText.textContent = `${result.count || selectedAudioFiles.length}\uac1c \ud30c\uc77c\uc744 \uac19\uc740 AI \uc124\uc815\uc73c\ub85c \ucc98\ub9ac\ud588\uc2b5\ub2c8\ub2e4. (Processed ${result.count || selectedAudioFiles.length} files with the same AI settings.)`;
    batchSummaryList.innerHTML = batchResults.map((item, index) => `
        <li>
            <button class="batch-summary-item" type="button" data-index="${index}">
                <div>
                    <strong>${escapeHtml(item.source_filename || item.archive_name || "audio")}</strong>
                    <span>${(item.targets || []).map(translateTarget).join(" + ")}</span>
                </div>
                <span>${formatSampleRate(item.sr || 0)} / ${item.bit_depth || 24}-bit / ${Number(item.lufs || 0).toFixed(1)} LUFS</span>
            </button>
        </li>
    `).join("");
    batchResultBox.classList.remove("hidden");
    resultZone.classList.remove("hidden");
    if (batchResults.length > 0) {
        selectBatchResult(0);
    }
}

function selectBatchResult(index) {
    const item = batchResults[index];
    const sourceFile = selectedAudioFiles[index];
    const sourceUrl = selectedAudioUrls[index];
    if (!item || !sourceFile || !sourceUrl || !item.report || !item.download_url) {
        return;
    }

    Array.from(batchSummaryList.querySelectorAll(".batch-summary-item")).forEach((button, buttonIndex) => {
        button.classList.toggle("active", buttonIndex === index);
    });

    renderSingleResult(
        {
            download_url: resolveApiAssetUrl(item.download_url),
            filename: item.filename || item.archive_name,
            stem_downloads: item.stem_downloads || [],
            report: item.report,
        },
        sourceFile,
        sourceUrl,
        { keepBatchVisible: true },
    );
}

batchSummaryList.addEventListener("click", (event) => {
    const button = event.target.closest(".batch-summary-item");
    if (!button) {
        return;
    }
    selectBatchResult(Number(button.dataset.index));
});

function setDownloadLabel(korean, english) {
    downloadLink.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        ${korean}<br>(${english})
    `;
}

function renderStemDownloads(stemDownloads) {
    if (!stemDownloadPanel || !stemDownloadList) {
        return;
    }
    if (!Array.isArray(stemDownloads) || stemDownloads.length === 0) {
        stemDownloadPanel.classList.add("hidden");
        stemDownloadList.innerHTML = "";
        return;
    }

    stemDownloadList.innerHTML = stemDownloads.map((item) => {
        const url = resolveApiAssetUrl(item.download_url);
        const filename = item.filename || `${item.stem || "stem"}_enhanced.wav`;
        const label = item.label || item.stem || "Stem";
        const labelEn = item.label_en || "Stem";
        const type = item.type === "raw" ? "raw" : "enhanced";
        const typeLabel = type === "raw" ? "RAW" : "ENHANCED";
        return `
            <a class="stem-download-link" href="${escapeHtml(url)}" download="${escapeHtml(filename)}" data-type="${escapeHtml(type)}">
                <span class="stem-download-kind">${typeLabel}</span>
                <span class="stem-download-label">${escapeHtml(label)}<br>(${escapeHtml(labelEn)})</span>
            </a>
        `;
    }).join("");
    stemDownloadPanel.classList.remove("hidden");
}

function getSelectedTarget() {
    const selectedTargets = getSelectedTargetList();

    return selectedTargets.length > 0
        ? selectedTargets.join(",")
        : "hifi_clean";
}

function getSelectedTargetList() {
    return Array.from(document.querySelectorAll('input[name="target-mode"]:checked'))
        .map((item) => item.value);
}

function updateAiIntentPanel() {
    if (!aiIntentTitle || !aiIntentCopy || !aiIntentTargets || !aiIntentSafety || !aiIntentOutput) {
        return;
    }

    const targets = getSelectedTargetList();
    const effectiveTargets = targets.length > 0 ? targets : ["hifi_clean"];
    const targetText = effectiveTargets.map(translateTarget).join(" + ");
    const amount = Number(intensitySlider.value);
    const denoiseEnabled = document.getElementById("opt-denoise").checked;
    const selectedStemMode = getSelectedStemMode();
    const selectedStemQuality = getSelectedStemQualityMode();
    const stemEnabled = selectedStemMode !== "off";
    const volumeMode = getSelectedVolumeMode();
    const manualEnabled = manualDspEnable.checked;

    aiIntentTitle.textContent = buildAiIntentTitle(effectiveTargets);
    aiIntentCopy.textContent = buildAiIntentCopy(
        effectiveTargets,
        amount,
        volumeMode,
        denoiseEnabled,
        manualEnabled,
        selectedStemMode,
        selectedStemQuality,
    );
    aiIntentTargets.textContent = `${targetText} / AI ${amount}%`;
    aiIntentSafety.textContent = [
        stemEnabled ? (selectedStemMode === "4stem" ? `4-stem ${selectedStemQuality}` : `2-stem ${selectedStemQuality}`) : null,
        volumeMode === "match_source" ? "원본 음량 유지" : "AI 목표 음량",
        "트루 피크 헤드룸",
        "스테레오 세이프",
        denoiseEnabled ? "노이즈 처리 사용" : "노이즈 처리 생략",
        manualEnabled ? "수동 DSP 보정 포함" : null,
    ].filter(Boolean).join(" / ");
    aiIntentOutput.textContent = `${formatOutputSampleRateLabel(outputSampleRate.value)} / ${outputBitDepth.value}-bit PCM`;
}

function updateStemQualityPolicy() {
    if (!stemQualityMode) {
        return;
    }

    const precisionOption = Array.from(stemQualityMode.options).find((option) => option.value === "precision");
    if (precisionOption) {
        precisionOption.disabled = !appCapabilities.precision_stems;
        precisionOption.textContent = appCapabilities.precision_stems
            ? "정밀 - 실험용 (Precision - experimental)"
            : "정밀 - 모바일 제한 (Precision - desktop only)";
    }
    if (stemQualityMode.value === "precision" && !appCapabilities.precision_stems) {
        stemQualityMode.value = appCapabilities.recommended_stem_quality || "balanced";
    }

    if (!stemPolicyCopy) {
        return;
    }
    const mode = getSelectedStemMode();
    const quality = getSelectedStemQualityMode();
    const estimate = estimateStemProcessingTime(mode, quality);
    if (mode === "off") {
        stemPolicyCopy.textContent = "Stem 분리를 끄면 전체 믹스 기준으로 빠르게 처리합니다. (Stem separation is off; processing uses the full mix.)";
    } else if (quality === "precision") {
        stemPolicyCopy.textContent = `${mode} precision은 실험용입니다. 첫 실행 시 모델 다운로드와 긴 처리 시간이 발생할 수 있습니다. 예상: ${estimate}.`;
    } else {
        const precisionScope = appCapabilities.precision_stems
            ? "이 PC 로컬 접속에서는 Precision을 선택할 수 있습니다."
            : "모바일 접속에서는 Precision을 제한합니다.";
        stemPolicyCopy.textContent = `${mode} ${quality} 권장. 실사용 기본은 balanced이며, 예상 처리 시간은 ${estimate}입니다. ${precisionScope}`;
    }
}

function estimateStemProcessingTime(mode, quality) {
    if (mode === "off") {
        return "1분 이내";
    }
    if (mode === "2stem") {
        return quality === "fast" ? "1-3분" : "2-5분";
    }
    if (quality === "fast") {
        return "2-5분";
    }
    if (quality === "precision") {
        return "8분 이상";
    }
    return "3-7분";
}

function buildAiIntentTitle(targets) {
    if (targets.length > 1) {
        return "선택한 청감 목표를 혼합해 분석";
    }
    return {
        restore: "원본 보존을 우선하는 복원 분석",
        hifi_clean: "하이파이 클린 기준으로 분석",
        hifi_bright: "디테일과 공기감을 우선하는 분석",
        warm_analog: "밀도와 부드러움을 우선하는 분석",
        loud_modern: "체감 음압과 에너지를 우선하는 분석",
        bass_boost: "저역 무게감과 헤드룸을 함께 분석",
        voice_focus: "목소리 명료도를 우선하는 분석",
    }[targets[0]] || "AI 청감 목표 기준으로 분석";
}

function buildAiIntentCopy(targets, amount, volumeMode, denoiseEnabled, manualEnabled, stemMode = "off", stemQuality = "balanced") {
    const volumeText = volumeMode === "match_source"
        ? "원본 음량을 유지"
        : "선택한 목표의 체감 음량에 맞춤";
    const denoiseText = denoiseEnabled
        ? "노이즈 플로어를 함께 판단"
        : "노이즈 제거를 생략";
    const stemText = stemMode === "4stem"
        ? ` Demucs 4-stem ${stemQuality} 분리 후 보컬, 드럼, 베이스, 기타 악기를 역할별로 개선하고 재합성합니다.`
        : stemMode === "2stem"
            ? ` Demucs 2-stem ${stemQuality} 분리 후 보컬과 반주를 각각 보수적으로 개선하고 재합성합니다.`
            : "";
    const manualText = manualEnabled
        ? " 수동 DSP 보정값을 최종 의도 위에 얹습니다."
        : "";

    if (targets.length > 1) {
        return `AI 적용량 ${amount}%로 여러 청감 목표를 혼합합니다. ${volumeText}하면서 스테이징, 헤드룸, 대역 분리도를 함께 보호합니다. ${denoiseText}합니다.${stemText}${manualText}`;
    }
    return `AI 적용량 ${amount}%로 현재 목표에 맞춰 톤, 다이내믹, 스테이징을 조정합니다. ${volumeText}하고, 트루 피크 헤드룸과 스테레오 이미지를 우선 보호합니다. ${denoiseText}합니다.${stemText}${manualText}`;
}

function formatOutputSampleRateLabel(value) {
    return {
        auto: "자동 샘플레이트",
        source: "원본 샘플레이트 유지",
        "44100": "44.1 kHz",
        "48000": "48 kHz",
        "96000": "96 kHz",
    }[value] || value;
}

function updateTargetComboWarning() {
    if (!targetComboWarning) {
        return;
    }
    const selected = new Set(
        Array.from(document.querySelectorAll('input[name="target-mode"]:checked')).map((item) => item.value)
    );
    const opposingPairs = [
        ["restore", "loud_modern"],
        ["hifi_bright", "warm_analog"],
        ["bass_boost", "voice_focus"],
    ];
    const hasOpposingPair = opposingPairs.some(([left, right]) => selected.has(left) && selected.has(right));
    targetComboWarning.classList.toggle("hidden", !hasOpposingPair);
    if (hasOpposingPair) {
        targetComboWarning.title = "반대 성향의 청감 목표가 함께 선택되어 개성이 중화될 수 있습니다.";
    }
}

function getSelectedVolumeMode() {
    const selectedVolume = document.querySelector('input[name="volume-mode"]:checked');
    return selectedVolume ? selectedVolume.value : "match_source";
}

function getSelectedStemMode() {
    const mode = stemSeparationMode?.value || "off";
    return ["off", "2stem", "4stem"].includes(mode) ? mode : "off";
}

function getSelectedStemQualityMode() {
    const mode = stemQualityMode?.value || "balanced";
    if (mode === "precision" && !appCapabilities.precision_stems) {
        return "balanced";
    }
    return ["fast", "balanced", "precision"].includes(mode) ? mode : "balanced";
}

async function loadAppVersion() {
    if (!versionPill) {
        return;
    }
    const response = await fetch(apiUrl("/api/version"));
    if (!response.ok) {
        return;
    }
    const version = await response.json();
    appCapabilities = {
        ...appCapabilities,
        ...(version.capabilities || {}),
    };
    versionPill.textContent = `v${version.version || "1.0.0"}`;
    versionPill.title = version.log_dir
        ? `Logs: ${version.log_dir}`
        : "Resonix AI";
    updateStemQualityPolicy();
    updateAiIntentPanel();
}

function resetLevelMatchGains() {
    levelMatchGains = {
        original: 1,
        enhanced: 1,
    };
}

function updateLevelMatchGains(report) {
    const summary = report?.quality_summary || {};
    const playbackGain = summary.level_match_playback_gain || {};
    const originalGain = Number(playbackGain.original);
    const enhancedGain = Number(playbackGain.enhanced);

    if (Number.isFinite(originalGain) && Number.isFinite(enhancedGain)) {
        levelMatchGains = {
            original: clampPlaybackGain(originalGain),
            enhanced: clampPlaybackGain(enhancedGain),
        };
        return;
    }

    const beforeLufs = Number(report?.before?.lufs);
    const afterLufs = Number(report?.after?.lufs);
    if (!Number.isFinite(beforeLufs) || !Number.isFinite(afterLufs)) {
        resetLevelMatchGains();
        return;
    }

    const referenceLufs = Math.min(beforeLufs, afterLufs);
    levelMatchGains = {
        original: clampPlaybackGain(10 ** ((referenceLufs - beforeLufs) / 20)),
        enhanced: clampPlaybackGain(10 ** ((referenceLufs - afterLufs) / 20)),
    };
}

function clampPlaybackGain(value) {
    if (!Number.isFinite(value)) {
        return 1;
    }
    return Math.max(0.05, Math.min(1, value));
}

function getPlaybackGain(mode) {
    if (!levelMatchToggle || !levelMatchToggle.checked) {
        return 1;
    }
    return mode === "enhanced" ? levelMatchGains.enhanced : levelMatchGains.original;
}

function applyPlaybackVolume() {
    masterPlayer.volume = getPlaybackGain(currentABMode);
}

function renderReport(report) {
    const after = report.after;
    const recommendation = report.recommendation;
    const delta = report.delta;
    const qualitySummary = formatQualitySummary(report);

    document.getElementById("metric-lufs").textContent = `${after.lufs.toFixed(1)}`;
    document.getElementById("metric-crest").textContent = `${after.crest_db.toFixed(1)}`;
    document.getElementById("metric-sr").textContent = formatSampleRate(after.sr);
    document.getElementById("metric-true-peak").textContent = Number.isFinite(after.true_peak_db)
        ? `${after.true_peak_db.toFixed(1)}`
        : "-";
    document.getElementById("metric-stereo").textContent = Number.isFinite(after.stereo_width)
        ? `${after.stereo_width.toFixed(2)}`
        : "-";

    const flags = after.quality_flags.length > 0
        ? after.quality_flags.slice(0, 2).map(translateFlag).join(", ")
        : "\ud2b9\uc774 \uc0ac\ud56d \uc5c6\uc74c (No major flags)";
    const targets = recommendation.targets
        ? recommendation.targets.map(translateTarget).join(" + ")
        : translateTarget(recommendation.target);
    const volumeMode = recommendation.volume_mode === "match_source"
        ? "\uc6d0\ubcf8 \uc74c\ub7c9 \uc720\uc9c0 (Match source)"
        : "AI \ubaa9\ud45c \uc74c\ub7c9 (AI target)";

    aiFeedbackText.textContent =
        `\ucc98\ub9ac \uc694\uc57d (Overview): ${targets}. ${volumeMode}.\n` +
        `${qualitySummary}\n` +
        `LUFS ${formatDelta(delta.lufs)}, True peak ${formatNumber(after.true_peak_db, 1)} dBTP, Stereo ${formatNumber(after.stereo_width, 2)} / Phase ${formatNumber(after.phase_correlation, 2)}. ` +
        `\uc8fc\uc758 (Flags): ${flags}.`;

    renderReportDetails(report);
    renderDecisionNotes(report);
    aiAnalysisBox.classList.remove("hidden");
}

function renderDecisionNotes(report) {
    if (!aiDecisionList) {
        return;
    }
    const recommendation = report.recommendation || {};
    const summary = report.quality_summary || {};
    const reasons = Array.isArray(recommendation.reasons)
        ? recommendation.reasons.slice(0, 3).map(translateReason)
        : [];
    const fallback = [
        summary.headroom_safe
            ? "트루 피크 헤드룸을 우선 보호했습니다. (Protected true-peak headroom.)"
            : "헤드룸 주의가 필요해 출력 게인을 보수적으로 제한했습니다. (Limited output gain for headroom.)",
        summary.stereo_preserved
            ? "스테레오 이미지를 보존하는 방향으로 처리했습니다. (Preserved stereo image.)"
            : "스테레오 변화가 감지되어 A/B 비교 확인이 필요합니다. (Stereo change needs A/B review.)",
        summary.volume_matched
            ? "원본과 가까운 청감 음량으로 맞췄습니다. (Matched source loudness.)"
            : "선택한 AI 목표 음량에 맞춰 레벨을 조정했습니다. (Adjusted to AI target loudness.)",
    ];
    const notes = (reasons.length > 0 ? reasons : fallback).slice(0, 3);
    aiDecisionList.innerHTML = notes.map((note) => `<li>${escapeHtml(note)}</li>`).join("");
}

function formatQualitySummary(report) {
    const summary = report.quality_summary || {};
    const volume = summary.volume_matched
        ? "\uc74c\ub7c9 \uc720\uc9c0 (Volume matched)"
        : `\uc74c\ub7c9 \ucc28\uc774 ${formatDelta(summary.loudness_delta_db || 0)} dB (Level changed)`;
    const headroom = summary.headroom_safe
        ? "\ud5e4\ub4dc\ub8f8 \uc548\uc815 (Headroom safe)"
        : "\ud5e4\ub4dc\ub8f8 \uc8fc\uc758 (Check headroom)";
    const clipping = summary.clipping_safe
        ? "\ud074\ub9ac\ud551 \uc704\ud5d8 \ub0ae\uc74c (Low clipping risk)"
        : "\ud074\ub9ac\ud551 \uc8fc\uc758 (Clipping check)";
    const stereo = summary.stereo_preserved
        ? "\uc2a4\ud14c\ub808\uc624 \ubcf4\uc874 (Stereo preserved)"
        : "\uc2a4\ud14c\ub808\uc624 \ubcc0\ud654 \uc788\uc74c (Stereo changed)";
    const validation = Number.isFinite(Number(summary.validation_score))
        ? ` 자동 검증 ${formatNumber(summary.validation_score, 0)}점 (${summary.validation_overall || "review"}).`
        : "";
    return `${volume}. ${headroom}. ${clipping}. ${stereo}.${validation}`;
}

function formatQualityValidation(report) {
    const validation = report.quality_validation || {};
    const checks = Array.isArray(validation.checks) ? validation.checks : [];
    if (!checks.length) {
        return "자동 A/B 검증 정보가 없습니다. (No automatic A/B validation.)";
    }
    const statusLabel = {
        pass: "통과",
        review: "검토",
        fail: "주의",
    }[validation.overall] || "검토";
    const weakChecks = checks
        .filter((item) => item.status !== "pass")
        .slice(0, 2)
        .map((item) => `${item.label}: ${item.detail}`)
        .join(" / ");
    const detail = weakChecks || "핵심 항목 안정";
    return `${statusLabel} ${formatNumber(validation.score, 0)} / 100. ${detail}. (Automatic A/B validation)`;
}

function formatAdaptiveAi(recommendation) {
    const adaptive = recommendation?.adaptive_ai || {};
    if (!adaptive.enabled) {
        return "자동 보정 정보 없음 (No adaptive AI data)";
    }
    const requested = Math.round(Number(adaptive.requested || 0) * 100);
    const effective = Math.round(Number(adaptive.effective || 0) * 100);
    const delta = effective - requested;
    const condition = {
        already_polished: "원본 품질 양호",
        needs_recovery: "복원 필요",
        fragile: "민감한 소스",
        balanced: "균형 소스",
        bypass: "보정 생략",
    }[adaptive.source_condition] || adaptive.source_condition || "균형 소스";
    const reason = Array.isArray(adaptive.reasons) && adaptive.reasons.length
        ? adaptive.reasons[0]
        : "Source stayed close to requested strength.";
    return `${requested}% -> ${effective}% (${formatDelta(delta)}%). ${condition}. ${reason}`;
}

function formatDspBudget(recommendation) {
    const budget = recommendation?.dsp_budget || {};
    const reductions = budget.reductions || {};
    if (!budget.enabled || Object.keys(reductions).length === 0) {
        return "DSP 예산 제한 정보 없음 (No DSP budget data)";
    }
    const limited = Object.entries(reductions)
        .filter(([, value]) => Number(value) < 0.999)
        .map(([key, value]) => `${key} ${(Number(value) * 100).toFixed(0)}%`);
    return limited.length
        ? `과처리 방지 예산 적용: ${limited.join(" / ")} (DSP budget limited)`
        : "과처리 방지 예산 내 처리 (Within DSP budget)";
}

function formatHarmonicSafety(recommendation) {
    const guard = recommendation?.harmonic_safety || {};
    if (!guard.enabled) {
        return "하모닉 보호 정보 없음 (No harmonic safety data)";
    }
    const factor = Number.isFinite(Number(guard.factor)) ? `${(Number(guard.factor) * 100).toFixed(0)}%` : "-";
    const reasons = Array.isArray(guard.reasons) && guard.reasons.length
        ? ` / ${guard.reasons.join(", ")}`
        : "";
    return `Exciter/Saturation 안전 계수 ${factor}${reasons}`;
}

function formatStemRisk(report) {
    const stem = report?.stem_separation || {};
    const riskMap = stem.stem_risk_map || stem.fallback_risk_map || {};
    if (!riskMap.enabled) {
        return stem.bypassed
            ? "Stem 위험으로 full-mix fallback 적용 (Stem branch bypassed)"
            : "Stem 위험도 정보 없음 (No stem risk data)";
    }
    const average = formatNumber(riskMap.average, 2);
    const max = formatNumber(riskMap.max, 2);
    const decision = riskMap.decision || "review";
    return `평균 ${average}, 최대 ${max}, 판단 ${decision} (Stem risk map)`;
}

function formatRemixOptimization(report) {
    const opt = report?.stem_separation?.remix_optimization || {};
    if (!opt.applied) {
        return "Stem remix 기본 gain 사용 (Default stem remix gain)";
    }
    const gainText = Object.entries(opt.gains || {})
        .map(([name, value]) => `${name} ${Number(value).toFixed(2)}`)
        .join(" / ");
    const reduction = Number.isFinite(Number(opt.error_reduction))
        ? `${(Number(opt.error_reduction) * 100).toFixed(0)}%`
        : "-";
    return `${gainText}. 원본 대비 오차 개선 ${reduction} (Source-match remix optimization)`;
}

function formatQualityGuard(report) {
    const guard = report?.quality_guard || {};
    if (!guard.applied) {
        return "품질 drift 보호 확인 완료 (Quality drift checked)";
    }
    const blend = `${(Number(guard.blend || 0) * 100).toFixed(0)}%`;
    const severity = guard.severity || "warning";
    const flags = Array.isArray(guard.flags) ? guard.flags.join(", ") : "";
    return `${severity}: 원본 보존 blend ${blend} / ${flags}`;
}

function renderReportDetails(report) {
    const before = report.before || {};
    const after = report.after || {};
    const summary = report.quality_summary || {};
    const recommendation = report.recommendation || {};
    const outputFormat = report.output_format || {};
    const bitDepth = outputFormat.bit_depth ? `${outputFormat.bit_depth}-bit` : "24-bit";
    const sampleRate = outputFormat.sample_rate || after.sr;

    const rows = [
        {
            label: "품질 판정 (Quality)",
            value: formatQualitySummary(report),
        },
        {
            label: "자동 A/B 검증 (A/B validation)",
            value: formatQualityValidation(report),
        },
        {
            label: "AI 자동 보정 (Adaptive AI)",
            value: formatAdaptiveAi(recommendation),
        },
        {
            label: "DSP 예산 (DSP budget)",
            value: formatDspBudget(recommendation),
        },
        {
            label: "하모닉 안전 (Harmonic safety)",
            value: formatHarmonicSafety(recommendation),
        },
        {
            label: "품질 보호 (Quality guard)",
            value: formatQualityGuard(report),
        },
        {
            label: "핵심 수치 (Core)",
            value: `LUFS ${formatNumber(before.lufs, 1)} -> ${formatNumber(after.lufs, 1)} (${formatDelta((after.lufs || 0) - (before.lufs || 0))}), True peak ${formatNumber(after.true_peak_db, 1)} dBTP`,
        },
        {
            label: "공간/분리 (Image)",
            value: `Stereo ${formatNumber(after.stereo_width, 2)} / Phase ${formatNumber(after.phase_correlation, 2)}, Separation ${formatNumber(summary.band_separation_score, 0)} / 100`,
        },
        {
            label: "출력 (Output)",
            value: `WAV ${bitDepth} / ${formatSampleRate(sampleRate)}`,
        },
    ];

    const stem = report.stem_separation || {};
    if (stem.enabled || stem.bypassed || stem.fallback_risk_map) {
        const mode = stem.mode === "4stem"
            ? "Demucs 4-stem: 보컬 / 드럼 / 베이스 / 기타 악기 개별 처리"
            : stem.bypassed
                ? "Stem branch bypass: full-mix DSP fallback"
                : "Demucs 2-stem: 보컬 / 반주 개별 처리";
        const quality = stem.quality_mode ? ` / 품질 ${stem.quality_mode}` : "";
        const fallback = stem.fallback_mode ? ` / ${stem.requested_mode || "requested"} -> ${stem.fallback_mode} fallback` : "";
        rows.splice(1, 0, {
            label: "Stem 분리 (Stem)",
            value: `${mode}${quality}${fallback}`,
        });
        rows.splice(2, 0, {
            label: "Stem 위험도 (Stem risk)",
            value: formatStemRisk(report),
        });
        rows.splice(3, 0, {
            label: "Stem 재합성 (Stem remix)",
            value: formatRemixOptimization(report),
        });
    }

    reportDetailList.innerHTML = rows.map((row) => `
        <li>
            <strong>${escapeHtml(row.label)}</strong>
            <span>${escapeHtml(row.value)}</span>
        </li>
    `).join("");
}
function startProcessingProgress() {
    clearProgressTimer();
    progressWaitTick = 0;
    processingPanel.classList.remove("hidden");
    processingPanel.classList.remove("is-waiting");
    setProcessingProgress(1, "작업을 서버에 전송하는 중... (Sending job to server...)");
}

function completeProcessingProgress() {
    clearProgressTimer();
    processingPanel.classList.remove("is-waiting");
    setProcessingProgress(100);
}

function failProcessingProgress() {
    clearProgressTimer();
    processingPanel.classList.remove("hidden");
    processingPanel.classList.remove("is-waiting");
    processingStage.textContent = "\ucc98\ub9ac \uc911 \uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4. (Processing failed.)";
    cancelProcessBtn?.classList.add("hidden");
}

function resetProcessingProgress() {
    clearProgressTimer();
    progressValue = 0;
    progressWaitTick = 0;
    processingPanel.classList.remove("is-waiting");
    processingPanel.classList.add("hidden");
    setProcessingProgress(0);
    cancelProcessBtn?.classList.add("hidden");
}

function clearProgressTimer() {
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
}

function setProcessingProgress(value, stageText = null) {
    progressValue = Math.max(0, Math.min(100, Math.round(value)));
    processingPercent.textContent = `${progressValue}%`;
    processingBar.style.width = `${progressValue}%`;
    processingPanel.classList.toggle("is-waiting", progressValue > 0 && progressValue < 100);

    const activeStages = getActiveProgressStages();
    const textStageIndex = getProgressStageIndex(progressValue, activeStages);
    const itemStageIndex = Math.min(textStageIndex, processingStageItems.length - 1);
    processingStage.textContent = stageText || activeStages[textStageIndex].text;
    processingStageItems.forEach((item, index) => {
        item.classList.toggle("done", index < itemStageIndex || progressValue === 100);
        item.classList.toggle("active", index === itemStageIndex && progressValue < 100);
    });
}

function refreshProcessingWaitingCopy() {
    progressWaitTick += 1;
    processingPanel.classList.add("is-waiting");
    const activeStages = getActiveProgressStages();
    const textStageIndex = getProgressStageIndex(progressValue, activeStages);
    const dots = ".".repeat((progressWaitTick % 3) + 1);
    processingStage.textContent = `${activeStages[textStageIndex].text} ${dots}`;
}

function getActiveProgressStages() {
    return getSelectedStemMode() !== "off" ? stemProgressStages : progressStages;
}

function getProgressStageIndex(value, stages = getActiveProgressStages()) {
    let activeIndex = 0;
    stages.forEach((stage, index) => {
        if (value >= stage.at) {
            activeIndex = index;
        }
    });
    return activeIndex;
}

function translateTarget(target) {
    return targetLabels[target] || target;
}

function translateFlag(flag) {
    return flagLabels[flag] || flag;
}

function translateReason(reason) {
    if (reason.startsWith("Blended listening targets:")) {
        const rawTargets = reason
            .replace("Blended listening targets:", "")
            .replace(".", "")
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean);
        return `\uc120\ud0dd\ud55c \uccad\uac10 \ubaa9\ud45c\ub97c \ud63c\ud569\ud588\uc2b5\ub2c8\ub2e4 (Blended listening targets): ${rawTargets.map(translateTarget).join(" + ")}.`;
    }
    return reasonLabels.get(reason) || reason;
}

function translateAdvice(recommendation) {
    const params = recommendation.dsp_params || {};
    const modeLabel = {
        denoise: "\ub178\uc774\uc988 \uc81c\uac70 (Denoise)",
        upsample: "\uc5c5\uc0d8\ud50c\ub9c1 (Upsample)",
        none: "\ub178\uc774\uc988 \uc81c\uac70 \uc0dd\ub7b5 (No denoise)",
    }[recommendation.mode] || recommendation.mode;
    const parts = [
        `\ucc98\ub9ac \ubaa8\ub4dc (Mode): ${modeLabel}`,
        `AI \uc801\uc6a9\ub7c9 (AI amount): ${Math.round(Number(recommendation.ai_amount ?? recommendation.intensity ?? 0) * 100)}%`,
    ];

    if (Number.isFinite(params.low_boost_db) && Math.abs(params.low_boost_db) > 0.01) {
        parts.push(`\uc800\uc5ed (Low) ${formatSigned(params.low_boost_db)} dB`);
    }
    if (Number.isFinite(params.mid_cut_db) && Math.abs(params.mid_cut_db) > 0.01) {
        parts.push(`\uc911\uc5ed (Mid) ${formatSigned(params.mid_cut_db)} dB`);
    }
    if (Number.isFinite(params.high_boost_db) && Math.abs(params.high_boost_db) > 0.01) {
        parts.push(`\uace0\uc5ed (High) ${formatSigned(params.high_boost_db)} dB`);
    }
    if (Number.isFinite(params.compress_ratio) && params.compress_ratio > 1.01) {
        parts.push(`\ucef4\ud504\ub808\uc158 (Compression) ${params.compress_ratio.toFixed(1)}:1`);
    }
    if (Number.isFinite(params.target_lufs)) {
        parts.push(`\ubaa9\ud45c (Target) ${params.target_lufs.toFixed(1)} LUFS`);
    }
    if (Number.isFinite(params.exciter_amount) && params.exciter_amount > 0.01) {
        parts.push(`\uc775\uc0ac\uc774\ud130 (Exciter) ${params.exciter_amount.toFixed(2)}`);
    }
    if (Number.isFinite(params.saturation_amount) && params.saturation_amount > 0.01) {
        parts.push(`\uc0c8\uce04\ub808\uc774\uc158 (Saturation) ${params.saturation_amount.toFixed(2)}`);
    }
    if (Number.isFinite(recommendation.output_sr)) {
        parts.push(`\ucd9c\ub825 \uc0d8\ud50c\ub808\uc774\ud2b8 (Output sample rate) ${formatSampleRate(recommendation.output_sr)}`);
    }

    return parts.join("; ");
}

async function readErrorMessage(response, fallback) {
    try {
        const errorData = await response.json();
        return errorData.detail || fallback;
    } catch {
        return `${fallback}: HTTP ${response.status}`;
    }
}

function formatSampleRate(sr) {
    return sr >= 1000 ? `${(sr / 1000).toFixed(sr % 1000 === 0 ? 0 : 1)}k` : String(sr);
}

function revokeSelectedAudioUrls() {
    selectedAudioUrls.forEach((url) => {
        try {
            URL.revokeObjectURL(url);
        } catch {}
    });
    selectedAudioUrls = [];
    originalAudioUrl = null;
}

function formatNumber(value, decimals = 1) {
    return Number.isFinite(Number(value)) ? Number(value).toFixed(decimals) : "-";
}

function formatGain(value) {
    const gain = Number(value);
    return Number.isFinite(gain) ? `${Math.round(gain * 100)}%` : "100%";
}

function formatStepName(step) {
    return String(step)
        .replace(/_/g, " ")
        .replace(/\bdb\b/g, "dB")
        .slice(0, 42);
}

function formatFileSize(bytes) {
    if (bytes >= 1024 * 1024) {
        return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    }
    return `${(bytes / 1024).toFixed(1)} KB`;
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatDelta(value) {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${value.toFixed(1)}`;
}

function formatSigned(value, decimals = 1) {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${Number(value).toFixed(decimals)}`;
}

function getWaveformAudioContext() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) {
        return null;
    }
    if (!waveformAudioContext) {
        waveformAudioContext = new AudioContextClass();
    }
    return waveformAudioContext;
}

async function prepareWaveformComparison() {
    const audioContext = getWaveformAudioContext();
    if (!audioContext || !originalAudioBlob || !enhancedAudioUrl) {
        return;
    }

    const [originalBuffer, enhancedBuffer] = await Promise.all([
        decodeAudioBlob(originalAudioBlob, audioContext),
        decodeAudioUrl(enhancedAudioUrl, audioContext),
    ]);

    waveformState.original = createWaveformPeaks(originalBuffer);
    waveformState.enhanced = createWaveformPeaks(enhancedBuffer);
    renderWaveforms();
}

async function decodeAudioBlob(blob, audioContext) {
    const arrayBuffer = await blob.arrayBuffer();
    return audioContext.decodeAudioData(arrayBuffer.slice(0));
}

async function decodeAudioUrl(url, audioContext) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(text.loadFailed);
    }
    const arrayBuffer = await response.arrayBuffer();
    return audioContext.decodeAudioData(arrayBuffer.slice(0));
}

function createWaveformPeaks(audioBuffer, pointCount = 1800) {
    const channels = Array.from({ length: audioBuffer.numberOfChannels }, (_, index) => audioBuffer.getChannelData(index));
    const length = audioBuffer.length;
    const segmentSize = Math.max(1, Math.floor(length / pointCount));
    const peaks = [];

    for (let start = 0; start < length; start += segmentSize) {
        const end = Math.min(length, start + segmentSize);
        let min = 1;
        let max = -1;

        for (let index = start; index < end; index += 1) {
            channels.forEach((channel) => {
                const sample = channel[index] || 0;
                min = Math.min(min, sample);
                max = Math.max(max, sample);
            });
        }

        peaks.push({
            min: Number.isFinite(min) ? min : 0,
            max: Number.isFinite(max) ? max : 0,
        });
    }

    return {
        peaks,
        duration: audioBuffer.duration,
    };
}

function resetWaveforms() {
    waveformState.original = null;
    waveformState.enhanced = null;
    renderWaveforms();
}

function renderWaveforms() {
    const progress = getPlaybackProgress();
    drawWaveform(waveformOriginal, waveformState.original, getCssColor("--accent", "#00f0ff"), progress, currentABMode === "original");
    drawWaveform(waveformEnhanced, waveformState.enhanced, getCssColor("--accent-2", "#8b5cf6"), progress, currentABMode === "enhanced");
}

function getCssColor(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

function getCssRgb(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

function drawWaveform(canvas, waveform, color, progress, isActive) {
    if (!canvas) {
        return;
    }

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(rect.width * dpr));
    const height = Math.max(1, Math.floor(rect.height * dpr));

    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }

    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "rgba(2, 7, 17, 0.82)";
    ctx.fillRect(0, 0, width, height);

    const center = height / 2;
    ctx.strokeStyle = "rgba(159, 184, 207, 0.18)";
    ctx.lineWidth = Math.max(1, dpr);
    ctx.beginPath();
    ctx.moveTo(0, center);
    ctx.lineTo(width, center);
    ctx.stroke();

    if (!waveform || waveform.peaks.length === 0) {
        return;
    }

    const peaks = waveform.peaks;
    const step = width / peaks.length;
    ctx.strokeStyle = color;
    ctx.globalAlpha = isActive ? 0.95 : 0.58;
    ctx.lineWidth = Math.max(1, dpr);
    ctx.beginPath();

    peaks.forEach((peak, index) => {
        const x = index * step;
        const yMin = center + peak.min * center * 0.86;
        const yMax = center + peak.max * center * 0.86;
        ctx.moveTo(x, yMin);
        ctx.lineTo(x, yMax);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;

    if (progress > 0) {
        const playhead = Math.max(0, Math.min(width, width * progress));
        const gradient = ctx.createLinearGradient(0, 0, playhead, 0);
        const accentRgb = getCssRgb("--accent-rgb", "0, 240, 255");
        const accent2Rgb = getCssRgb("--accent-2-rgb", "139, 92, 246");
        gradient.addColorStop(0, `rgba(${accentRgb}, 0.12)`);
        gradient.addColorStop(1, isActive ? `rgba(${accent2Rgb}, 0.14)` : `rgba(${accentRgb}, 0.06)`);
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, playhead, height);
        ctx.strokeStyle = isActive ? color : "rgba(159, 184, 207, 0.46)";
        ctx.lineWidth = Math.max(1, 2 * dpr);
        ctx.beginPath();
        ctx.moveTo(playhead, 0);
        ctx.lineTo(playhead, height);
        ctx.stroke();
    }
}

function getPlaybackProgress() {
    if (!masterPlayer.duration || !Number.isFinite(masterPlayer.duration)) {
        return 0;
    }
    return Math.max(0, Math.min(1, masterPlayer.currentTime / masterPlayer.duration));
}

async function switchAudioMode(mode, keepTime = true) {
    if (!masterPlayer.src || !enhancedAudioUrl) {
        return;
    }

    const isPlaying = !masterPlayer.paused;
    const currentTime = keepTime ? masterPlayer.currentTime : 0;
    setABMode(mode);
    applyPlaybackVolume();

    const nextSource = mode === "enhanced" ? enhancedAudioUrl : originalAudioUrl;
    if (masterPlayer.src !== new URL(nextSource, window.location.href).href) {
        masterPlayer.src = nextSource;
        masterPlayer.load();
    }

    try {
        await waitForAudioReady(masterPlayer);
        masterPlayer.currentTime = Math.min(currentTime, masterPlayer.duration || currentTime);
    } catch {
        masterPlayer.currentTime = 0;
    }

    if (isPlaying) {
        masterPlayer.play().catch(() => {});
    }
    renderWaveforms();
}

async function seekWaveform(mode, event) {
    if (!enhancedAudioUrl) {
        return;
    }

    const canvas = mode === "enhanced" ? waveformEnhanced : waveformOriginal;
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    await switchAudioMode(mode);
    if (masterPlayer.duration && Number.isFinite(masterPlayer.duration)) {
        masterPlayer.currentTime = ratio * masterPlayer.duration;
    }
    renderWaveforms();
}

function startWaveformAnimation() {
    if (waveformAnimationFrame) {
        return;
    }

    const tick = () => {
        renderWaveforms();
        waveformAnimationFrame = masterPlayer.paused ? null : requestAnimationFrame(tick);
    };
    waveformAnimationFrame = requestAnimationFrame(tick);
}

function stopWaveformAnimation() {
    if (waveformAnimationFrame) {
        cancelAnimationFrame(waveformAnimationFrame);
        waveformAnimationFrame = null;
    }
    renderWaveforms();
}

abToggleBtn.addEventListener("click", async (event) => {
    event.preventDefault();
    const nextMode = abToggleBtn.dataset.mode === "enhanced" ? "original" : "enhanced";
    await switchAudioMode(nextMode);
});

function setABMode(mode) {
    const isEnhanced = mode === "enhanced";
    currentABMode = mode;
    abToggleBtn.dataset.mode = mode;
    abToggleBtn.classList.toggle("restored", isEnhanced);
    labelOriginal.classList.toggle("active", !isEnhanced);
    labelEnhanced.classList.toggle("active", isEnhanced);
    waveformRowOriginal.classList.toggle("active", !isEnhanced);
    waveformRowEnhanced.classList.toggle("active", isEnhanced);
    nowPlayingText.textContent = isEnhanced ? text.enhanced : text.original;
    applyPlaybackVolume();
    renderWaveforms();
}

if (levelMatchToggle) {
    levelMatchToggle.addEventListener("change", applyPlaybackVolume);
}

waveformOriginal.addEventListener("click", (event) => {
    seekWaveform("original", event).catch(() => {});
});

waveformEnhanced.addEventListener("click", (event) => {
    seekWaveform("enhanced", event).catch(() => {});
});

masterPlayer.addEventListener("play", startWaveformAnimation);
masterPlayer.addEventListener("pause", stopWaveformAnimation);
masterPlayer.addEventListener("ended", stopWaveformAnimation);
masterPlayer.addEventListener("seeked", renderWaveforms);
masterPlayer.addEventListener("loadedmetadata", renderWaveforms);

window.addEventListener("resize", () => {
    clearTimeout(waveformResizeTimer);
    waveformResizeTimer = setTimeout(renderWaveforms, 120);
});

function waitForAudioReady(audioElement) {
    return new Promise((resolve, reject) => {
        if (audioElement.readyState >= 3) {
            resolve();
            return;
        }

        const timeoutId = setTimeout(() => {
            cleanup();
            reject(new Error(text.loadTimeout));
        }, 5000);

        const cleanup = () => {
            clearTimeout(timeoutId);
            audioElement.removeEventListener("canplay", onReady);
            audioElement.removeEventListener("error", onError);
        };

        const onReady = () => {
            cleanup();
            resolve();
        };

        const onError = () => {
            cleanup();
            reject(new Error(text.loadFailed));
        };

        audioElement.addEventListener("canplay", onReady, { once: true });
        audioElement.addEventListener("error", onError, { once: true });
    });
}
