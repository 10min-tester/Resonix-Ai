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
let currentABMode = "original";
let waveformAnimationFrame = null;
let waveformAudioContext = null;
let waveformResizeTimer = null;

const waveformState = {
    original: null,
    enhanced: null,
};

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
]);

const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const processBtn = document.getElementById("process-btn");
const versionPill = document.getElementById("version-pill");
const intensitySlider = document.getElementById("intensity-slider");
const intensityValue = document.getElementById("intensity-value");
const outputSampleRate = document.getElementById("output-sample-rate");
const outputBitDepth = document.getElementById("output-bit-depth");
const manualDspEnable = document.getElementById("manual-dsp-enable");
const manualDspGrid = document.getElementById("manual-dsp-grid");
const resultZone = document.getElementById("result-zone");
const abToggleBtn = document.getElementById("ab-toggle-btn");
const masterPlayer = document.getElementById("master-player");
const audioOriginal = document.getElementById("audio-original");
const audioEnhanced = document.getElementById("audio-enhanced");
const downloadLink = document.getElementById("download-link");
const singlePreviewPanel = document.getElementById("single-preview-panel");
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
const reportDetailList = document.getElementById("report-detail-list");
const processingPanel = document.getElementById("processing-panel");
const processingStage = document.getElementById("processing-stage");
const processingPercent = document.getElementById("processing-percent");
const processingBar = document.getElementById("processing-bar");
const processingStageItems = Array.from(document.querySelectorAll("#processing-stages li"));
const settingsBtn = document.getElementById("settings-btn");
const settingsModal = document.getElementById("settings-modal");
const groqApiKeyInput = document.getElementById("groq-api-key");
const modalSave = document.getElementById("modal-save");
const modalCancel = document.getElementById("modal-cancel");

settingsBtn.addEventListener("click", () => {
    groqApiKeyInput.value = localStorage.getItem("groq_api_key") || "";
    settingsModal.classList.add("active");
});

modalCancel.addEventListener("click", () => {
    settingsModal.classList.remove("active");
});

modalSave.addEventListener("click", () => {
    localStorage.setItem("groq_api_key", groqApiKeyInput.value.trim());
    settingsModal.classList.remove("active");
});

settingsModal.addEventListener("click", (event) => {
    if (event.target === settingsModal) {
        settingsModal.classList.remove("active");
    }
});

loadAppVersion().catch(() => {});

dropZone.addEventListener("click", () => fileInput.click());

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

    processBtn.disabled = false;
    processBtn.querySelector(".btn-text").textContent = files.length > 1 ? text.batchStart : text.start;
    resultZone.classList.add("hidden");
    aiAnalysisBox.classList.add("hidden");
    batchResultBox.classList.add("hidden");
    resetProcessingProgress();
}

intensitySlider.addEventListener("input", (event) => {
    intensityValue.textContent = `${event.target.value}%`;
});

const manualControls = [
    { id: "manual-lowcut", key: "lowcut_hz", suffix: " Hz", decimals: 0 },
    { id: "manual-low", key: "low_boost_db", suffix: " dB", decimals: 1, signed: true },
    { id: "manual-mid", key: "mid_cut_db", suffix: " dB", decimals: 1, signed: true },
    { id: "manual-high", key: "high_boost_db", suffix: " dB", decimals: 1, signed: true },
    { id: "manual-compress", key: "compress_ratio", suffix: ":1", decimals: 2 },
    { id: "manual-lufs", key: "target_lufs", suffix: " LUFS", decimals: 1 },
    { id: "manual-exciter", key: "exciter_amount", suffix: "", decimals: 2 },
    { id: "manual-saturation", key: "saturation_amount", suffix: "", decimals: 2 },
];

manualControls.forEach((control) => {
    const input = document.getElementById(control.id);
    input.addEventListener("input", () => updateManualControlOutput(control));
    updateManualControlOutput(control);
});

manualDspEnable.addEventListener("change", () => {
    manualDspGrid.classList.toggle("manual-disabled", !manualDspEnable.checked);
});

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
    }
});

async function processSingleAudio() {
    const processForm = createProcessingForm();
    processForm.append("file", originalAudioBlob);

    const processResponse = await fetch("/api/process", {
        method: "POST",
        body: processForm,
    });

    if (!processResponse.ok) {
        throw new Error(await readErrorMessage(processResponse, text.processFailed));
    }

    return processResponse.json();
}

async function processBatchAudio() {
    const processForm = createProcessingForm();
    selectedAudioFiles.forEach((file) => {
        processForm.append("files", file);
    });

    const processResponse = await fetch("/api/process-batch", {
        method: "POST",
        body: processForm,
    });

    if (!processResponse.ok) {
        throw new Error(await readErrorMessage(processResponse, text.processFailed));
    }

    return processResponse.json();
}

function createProcessingForm() {
    const processForm = new FormData();
    processForm.append("target", getSelectedTarget());
    processForm.append("intensity", String(Number(intensitySlider.value) / 100));
    processForm.append("use_denoise", document.getElementById("opt-denoise").checked ? "true" : "false");
    processForm.append("volume_mode", getSelectedVolumeMode());
    processForm.append("output_sample_rate", outputSampleRate.value);
    processForm.append("output_bit_depth", outputBitDepth.value);
    const manualParams = getManualDspParams();
    if (manualParams) {
        processForm.append("dsp_params", JSON.stringify(manualParams));
    }
    return processForm;
}

function getManualDspParams() {
    if (!manualDspEnable.checked) {
        return null;
    }

    const params = {
        limiter_ceiling_db: -1.0,
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
        ? formatSigned(value)
        : value.toFixed(control.decimals);
    output.textContent = `${formatted}${control.suffix}`;
}

function renderSingleResult(result, sourceFile = originalAudioBlob, sourceUrl = originalAudioUrl, options = {}) {
    originalAudioBlob = sourceFile;
    originalAudioUrl = sourceUrl;
    currentReport = result.report;
    enhancedAudioUrl = result.download_url;

    audioOriginal.src = originalAudioUrl;
    audioEnhanced.src = enhancedAudioUrl;
    masterPlayer.src = originalAudioUrl;

    downloadLink.href = result.download_url;
    downloadLink.download = result.filename || `enhanced_${originalAudioBlob.name.replace(/\.[^.]+$/, ".wav")}`;
    setDownloadLabel("\ud5a5\uc0c1\ub41c \ud30c\uc77c \ub2e4\uc6b4\ub85c\ub4dc", "Download enhanced file");

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
    currentReport = null;
    enhancedAudioUrl = null;
    batchResults = result.summary || [];
    masterPlayer.pause();
    masterPlayer.removeAttribute("src");
    audioOriginal.removeAttribute("src");
    audioEnhanced.removeAttribute("src");
    resetWaveforms();

    batchZipLink.href = result.download_url;
    batchZipLink.download = result.filename || "ResonixAI_Batch.zip";

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
            download_url: item.download_url,
            filename: item.filename || item.archive_name,
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

function getSelectedTarget() {
    const selectedTargets = Array.from(document.querySelectorAll('input[name="target-mode"]:checked'));

    return selectedTargets.length > 0
        ? selectedTargets.map((item) => item.value).join(",")
        : "hifi_clean";
}

function getSelectedVolumeMode() {
    const selectedVolume = document.querySelector('input[name="volume-mode"]:checked');
    return selectedVolume ? selectedVolume.value : "target";
}

async function loadAppVersion() {
    if (!versionPill) {
        return;
    }
    const response = await fetch("/api/version");
    if (!response.ok) {
        return;
    }
    const version = await response.json();
    versionPill.textContent = `v${version.version || "0.9.0"}`;
    versionPill.title = version.log_dir
        ? `Logs: ${version.log_dir}`
        : "Resonix AI";
}

function renderReport(report) {
    const after = report.after;
    const recommendation = report.recommendation;
    const delta = report.delta;

    document.getElementById("metric-lufs").textContent = `${after.lufs.toFixed(1)}`;
    document.getElementById("metric-crest").textContent = `${after.crest_db.toFixed(1)}`;
    document.getElementById("metric-sr").textContent = formatSampleRate(after.sr);
    document.getElementById("metric-true-peak").textContent = Number.isFinite(after.true_peak_db)
        ? `${after.true_peak_db.toFixed(1)}`
        : "-";
    document.getElementById("metric-stereo").textContent = Number.isFinite(after.stereo_width)
        ? `${after.stereo_width.toFixed(2)}`
        : "-";

    const reasons = recommendation.reasons.slice(0, 3).map(translateReason).join(" ");
    const flags = after.quality_flags.length > 0
        ? ` \ub0a8\uc740 \uc8fc\uc758 \ud56d\ubaa9 (Remaining flags): ${after.quality_flags.map(translateFlag).join(", ")}.`
        : "";
    const targets = recommendation.targets
        ? recommendation.targets.map(translateTarget).join(" + ")
        : translateTarget(recommendation.target);
    const volumeMode = recommendation.volume_mode === "match_source"
        ? "\uc6d0\ubcf8 \uc74c\ub7c9 \uc720\uc9c0 (Match source)"
        : "AI \ubaa9\ud45c \uc74c\ub7c9 (AI target)";
    const stereoText = Number.isFinite(after.stereo_width) && Number.isFinite(after.phase_correlation)
        ? ` \uc2a4\ud14c\ub808\uc624 \ud3ed (Stereo width) ${after.stereo_width.toFixed(2)} ${formatDelta(delta.stereo_width || 0)}, \uc704\uc0c1 \uc0c1\uad00 (Phase correlation) ${after.phase_correlation.toFixed(2)} ${formatDelta(delta.phase_correlation || 0)}.`
        : "";
    const truePeakText = Number.isFinite(after.true_peak_db)
        ? ` \ud2b8\ub8e8 \ud53c\ud06c (True peak) ${after.true_peak_db.toFixed(1)} dBTP ${formatDelta(delta.true_peak_db || 0)}.`
        : "";

    aiFeedbackText.textContent =
        `${targets}: ${volumeMode}. ${translateAdvice(recommendation)}. LUFS ${formatDelta(delta.lufs)} -> ${report.target_lufs.toFixed(1)}, \ub178\uc774\uc988 \ud50c\ub85c\uc5b4 (Noise floor) ${formatDelta(delta.noise_floor_db)} dB.${stereoText}${truePeakText} ${reasons}${flags}`;

    renderReportDetails(report);
    aiAnalysisBox.classList.remove("hidden");
}

function renderReportDetails(report) {
    const before = report.before || {};
    const after = report.after || {};
    const outputFormat = report.output_format || {};
    const steps = Array.isArray(report.applied_steps) ? report.applied_steps : [];
    const bitDepth = outputFormat.bit_depth ? `${outputFormat.bit_depth}-bit` : "24-bit";
    const sampleRate = outputFormat.sample_rate || after.sr;
    const stepText = steps.slice(0, 5).map(formatStepName).join(", ");

    const rows = [
        {
            label: "\ucd9c\ub825 \ud3ec\ub9f7 (Output format)",
            value: `WAV ${bitDepth} / ${formatSampleRate(sampleRate)}`,
        },
        {
            label: "LUFS",
            value: `${formatNumber(before.lufs, 1)} -> ${formatNumber(after.lufs, 1)} (${formatDelta((after.lufs || 0) - (before.lufs || 0))})`,
        },
        {
            label: "\ud2b8\ub8e8 \ud53c\ud06c (True peak)",
            value: `${formatNumber(before.true_peak_db, 1)} -> ${formatNumber(after.true_peak_db, 1)} dBTP`,
        },
        {
            label: "\uc2a4\ud14c\ub808\uc624 / \uc704\uc0c1 (Stereo / phase)",
            value: `${formatNumber(after.stereo_width, 2)} / ${formatNumber(after.phase_correlation, 2)}`,
        },
        {
            label: "\uc801\uc6a9 \ub2e8\uacc4 (Applied steps)",
            value: stepText || "-",
        },
    ];

    reportDetailList.innerHTML = rows.map((row) => `
        <li>
            <strong>${row.label}</strong>
            <span>${row.value}</span>
        </li>
    `).join("");
}

function startProcessingProgress() {
    clearProgressTimer();
    processingPanel.classList.remove("hidden");
    setProcessingProgress(3);
    progressTimer = setInterval(() => {
        if (progressValue >= 92) {
            return;
        }
        const remaining = 92 - progressValue;
        const step = Math.max(1, Math.ceil(remaining * 0.08));
        setProcessingProgress(Math.min(92, progressValue + step));
    }, 650);
}

function completeProcessingProgress() {
    clearProgressTimer();
    setProcessingProgress(100);
}

function failProcessingProgress() {
    clearProgressTimer();
    processingPanel.classList.remove("hidden");
    processingStage.textContent = "\ucc98\ub9ac \uc911 \uc624\ub958\uac00 \ubc1c\uc0dd\ud588\uc2b5\ub2c8\ub2e4. (Processing failed.)";
}

function resetProcessingProgress() {
    clearProgressTimer();
    progressValue = 0;
    processingPanel.classList.add("hidden");
    setProcessingProgress(0);
}

function clearProgressTimer() {
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
}

function setProcessingProgress(value) {
    progressValue = Math.max(0, Math.min(100, Math.round(value)));
    processingPercent.textContent = `${progressValue}%`;
    processingBar.style.width = `${progressValue}%`;

    const textStageIndex = getProgressStageIndex(progressValue);
    const itemStageIndex = Math.min(textStageIndex, processingStageItems.length - 1);
    processingStage.textContent = progressStages[textStageIndex].text;
    processingStageItems.forEach((item, index) => {
        item.classList.toggle("done", index < itemStageIndex || progressValue === 100);
        item.classList.toggle("active", index === itemStageIndex && progressValue < 100);
    });
}

function getProgressStageIndex(value) {
    let activeIndex = 0;
    progressStages.forEach((stage, index) => {
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

function formatSigned(value) {
    const sign = value >= 0 ? "+" : "";
    return `${sign}${Number(value).toFixed(1)}`;
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
    drawWaveform(waveformOriginal, waveformState.original, "#00f0ff", progress, currentABMode === "original");
    drawWaveform(waveformEnhanced, waveformState.enhanced, "#8b5cf6", progress, currentABMode === "enhanced");
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
        gradient.addColorStop(0, "rgba(0, 240, 255, 0.12)");
        gradient.addColorStop(1, isActive ? "rgba(139, 92, 246, 0.14)" : "rgba(0, 240, 255, 0.06)");
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
    renderWaveforms();
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
