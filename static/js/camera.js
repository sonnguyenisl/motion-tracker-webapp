// Camera page controller: webcam, optional live skeleton (MediaPipe Tasks JS),
// MediaRecorder capture, and upload-on-stop that hands off to the server for
// scoring. Live progress arrives via socket.js window events.

import { ready as socketReady, getRoom, emitStop } from "./socket.js";

// Read config from a <script id="ftConfig" type="application/json"> tag.
var cfgEl = document.getElementById("ftConfig");
var cfg = cfgEl ? JSON.parse(cfgEl.textContent) : {};

const preview = document.getElementById("preview");
const overlay = document.getElementById("overlay");
const octx = overlay.getContext("2d");
const refOverlay = document.getElementById("refOverlay");
const rctx = refOverlay.getContext("2d");

const btnStartCam = document.getElementById("btnStartCam");
const btnSkeleton = document.getElementById("btnSkeleton");
const btnRefOverlay = document.getElementById("btnRefOverlay");
const btnRecord = document.getElementById("btnRecord");
const btnStop = document.getElementById("btnStop");
const statusEl = document.getElementById("status");
const badge = document.getElementById("recordingBadge");
const timerEl = document.getElementById("timer");
const countdownEl = document.getElementById("countdown");
const countdownNum = document.getElementById("countdownNum");
const referenceSelect = document.getElementById("referenceSelect");

const modeLive = document.getElementById("modeLive");
const modeUpload = document.getElementById("modeUpload");
const livePanel = document.getElementById("livePanel");
const uploadPanel = document.getElementById("uploadPanel");
const videoFileInput = document.getElementById("videoFileInput");
const videoFileName = document.getElementById("videoFileName");

const COUNTDOWN_SECONDS = 3;
// Max length for any scored clip; mirrors config.MAX_VIDEO_SECONDS (passed via
// ftConfig). Live recording auto-stops here and the trimmer window is capped.
const MAX_VIDEO_SECONDS = cfg.maxVideoSeconds || 30;

const progressWrap = document.getElementById("progressWrap");
const progressBar = document.getElementById("progressBar");
const progressMsg = document.getElementById("progressMsg");

let stream = null;
let recorder = null;
let chunks = [];
let timerId = null;
let seconds = 0;
let countdownId = null;

let skeletonOn = false;
let landmarker = null;
let rafId = null;
let drawingUtils = null;
let refOverlayOn = false;

// --- Trimmer state ---
let videoTrimmer = null;
let pendingRecordingBlob = null;     // live recording blob awaiting trim confirmation
let pendingRecordingFilename = null; // live recording filename
let pendingObjectUrl = null;         // blob URL that must be revoked after use

// Hidden inputs for trim values (appended to the form data on submit)
function ensureTrimInputs() {
  let si = document.getElementById("ftTrimStartInput");
  let ei = document.getElementById("ftTrimEndInput");
  if (!si) {
    si = document.createElement("input");
    si.type = "hidden";
    si.id = "ftTrimStartInput";
    si.name = "trim_start";
    si.value = "0";
    document.body.appendChild(si);
  }
  if (!ei) {
    ei = document.createElement("input");
    ei.type = "hidden";
    ei.id = "ftTrimEndInput";
    ei.name = "trim_end";
    ei.value = "0";
    document.body.appendChild(ei);
  }
  return { startInput: si, endInput: ei };
}

// Build a trim preview area after a given element
function createTrimArea() {
  const wrap = document.createElement("div");
  wrap.className = "ft-trim-area hidden mt-4";
  wrap.id = "ftTrimArea";

  const vid = document.createElement("video");
  vid.controls = true;
  vid.preload = "metadata";
  vid.className = "w-full rounded-xl border border-gray-700 bg-black";
  wrap.appendChild(vid);

  return { wrap, video: vid };
}

function destroyTrimmer() {
  if (videoTrimmer) {
    videoTrimmer.destroy();
    videoTrimmer = null;
  }
  const area = document.getElementById("ftTrimArea");
  if (area) {
    area.remove();
  }
}

// Show trimming UI for a given blob/video URL.
// The Confirm button starts disabled and is enabled once the video loads.
function showTrimmer(videoUrl, onConfirm, onCancel) {
  destroyTrimmer();

  const { wrap, video } = createTrimArea();
  video.src = videoUrl;
  uploadPanel.parentNode.insertBefore(wrap, uploadPanel.nextSibling);

  // Confirm/Cancel buttons — Confirm starts disabled
  const btnRow = document.createElement("div");
  btnRow.className = "flex gap-3 mt-3";
  btnRow.innerHTML =
    '<button type="button" id="ftTrimConfirmBtn" disabled class="flex-1 from-gray-700 to-gray-700 text-gray-500 cursor-not-allowed py-3 rounded-2xl font-bold uppercase tracking-wider text-sm">Confirm & Score</button>' +
    '<button type="button" id="ftTrimCancelBtn" class="flex-1 bg-gray-800 border border-gray-700 hover:border-gray-500 transition py-3 rounded-2xl font-bold uppercase tracking-wider text-sm text-gray-300">Cancel</button>';
  wrap.appendChild(btnRow);

  wrap.classList.remove("hidden");

  const confirmBtn = document.getElementById("ftTrimConfirmBtn");

  video.addEventListener("loadedmetadata", function initTrimmer() {
    video.removeEventListener("loadedmetadata", initTrimmer);
    videoTrimmer = new VideoTrimmer(video, {
      maxDuration: MAX_VIDEO_SECONDS,
      onTrimChange: function (start, end) {
        const ins = ensureTrimInputs();
        ins.startInput.value = start;
        ins.endInput.value = end;
      },
    });
    // Enable the confirm button once the video is loaded
    confirmBtn.disabled = false;
    modeLive.disabled = true;
    videoFileInput.disabled = true;
    modeLive.classList.add('opacity-50', 'cursor-not-allowed');
    confirmBtn.className = "flex-1 bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 transition py-3 rounded-2xl font-bold uppercase tracking-wider text-sm shadow-lg shadow-orange-500/30";
  });

  confirmBtn.addEventListener("click", function () {
    if (videoTrimmer) {
      const vals = videoTrimmer.getTrimValues();
      const ins = ensureTrimInputs();
      ins.startInput.value = vals.start;
      ins.endInput.value = vals.end;
    }
    wrap.classList.add("hidden");
    if (onConfirm) onConfirm();
  });

  document.getElementById("ftTrimCancelBtn").addEventListener("click", function () {
    destroyTrimmer();
    const ins = ensureTrimInputs();
    ins.startInput.value = "0";
    ins.endInput.value = "0";
    if (onCancel) onCancel();
  });
}

function setStatus(text, color) {
  statusEl.textContent = text;
  statusEl.className = "text-center text-sm mt-8 mb-5 " + (color || "text-gray-400");
}

function fmt(s) {
  const m = String(Math.floor(s / 60)).padStart(2, "0");
  const r = String(s % 60).padStart(2, "0");
  return `${m}:${r}`;
}

// --- Camera -------------------------------------------------------------
function stopCamera() {
  if (!stream) return;
  stopSkeleton();
  cancelCountdown();
  if (recorder && recorder.state !== "inactive") recorder.stop();
  clearInterval(timerId);
  badge.classList.add("hidden");
  stream.getTracks().forEach((t) => t.stop());
  stream = null;
  preview.srcObject = null;
  btnRecord.disabled = true;
  btnStop.disabled = true;
  btnSkeleton.disabled = true;
  setRefOverlay(false);
  btnRefOverlay.disabled = true;
  btnStartCam.textContent = "Start Camera";
}

btnStartCam.addEventListener("click", async () => {
  if (stream) {
    stopCamera();
    setStatus("Camera off.");
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    preview.srcObject = stream;
    await preview.play().catch(() => {});
    overlay.width = preview.videoWidth || 640;
    overlay.height = preview.videoHeight || 480;
    btnRecord.disabled = !cfg.hasReference;
    btnSkeleton.disabled = false;
    btnRefOverlay.disabled = !cfg.hasReference;
    if (refOverlayOn) drawReferenceOverlay();
    btnStartCam.textContent = "Stop Camera";
    setStatus(cfg.hasReference
      ? "Camera ready. Hit Record when you're set."
      : "Camera ready — but upload a reference before recording.");
  } catch (err) {
    setStatus("Could not access webcam: " + err.message, "text-red-400");
  }
});

// --- Live skeleton (optional) ------------------------------------------
btnSkeleton.addEventListener("click", async () => {
  if (skeletonOn) {
    stopSkeleton();
    return;
  }
  setStatus("Loading pose model…");
  try {
    if (!landmarker) await initLandmarker();
    skeletonOn = true;
    btnSkeleton.textContent = "Skeleton: On";
    setStatus("Live skeleton on.");
    renderLoop();
  } catch (err) {
    setStatus("Live skeleton unavailable: " + err.message, "text-red-400");
  }
});

async function initLandmarker() {
  const vision = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14"
  );
  const { PoseLandmarker, FilesetResolver, DrawingUtils } = vision;
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  landmarker = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  drawingUtils = new DrawingUtils(octx);
  window._PoseLandmarker = PoseLandmarker; // for connection list
}

function stopSkeleton() {
  skeletonOn = false;
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  btnSkeleton.textContent = "Skeleton: Off";
}

function renderLoop() {
  if (!skeletonOn || !stream) return;
  if (preview.readyState >= 2) {
    const result = landmarker.detectForVideo(preview, performance.now());
    octx.clearRect(0, 0, overlay.width, overlay.height);
    const PoseLandmarker = window._PoseLandmarker;
    for (const landmarks of result.landmarks || []) {
      drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
        color: "#3b82f6", lineWidth: 3,
      });
      drawingUtils.drawLandmarks(landmarks, { color: "#22c55e", radius: 3 });
    }
  }
  rafId = requestAnimationFrame(renderLoop);
}

// --- Reference alignment overlay (static first-frame ref pose) ---------
// Connections between the 12 tracked landmarks (indices into the stored pose):
// 0 LElbow 1 RElbow 2 LShoulder 3 RShoulder 4 LHip 5 RHip
// 6 LAnkle 7 RAnkle 8 LWrist 9 RWrist 10 LKnee 11 RKnee
const REF_CONNECTIONS = [
  [2, 3], [2, 0], [0, 8], [3, 1], [1, 9],
  [2, 4], [3, 5], [4, 5], [4, 10], [10, 6], [5, 11], [11, 7],
];

function clearReferenceOverlay() {
  rctx.clearRect(0, 0, refOverlay.width, refOverlay.height);
}

function drawReferenceOverlay() {
  const id = referenceSelect ? referenceSelect.value : null;
  const pts = id && cfg.refPoses ? cfg.refPoses[id] : null;

  const w = preview.videoWidth || refOverlay.clientWidth || 640;
  const h = preview.videoHeight || refOverlay.clientHeight || 480;
  refOverlay.width = w;
  refOverlay.height = h;
  clearReferenceOverlay();
  if (!pts || !pts.length) return;

  // Fit the reference pose to the frame: scale uniformly to fill ~75% of the
  // view and centre it, so it never spills out or sits off to one side.
  const FIT_RATIO = 0.6; // fraction of the frame the reference pose fills
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const poseW = Math.max(maxX - minX, 1e-6);
  const poseH = Math.max(maxY - minY, 1e-6);
  const scale = Math.min((w * FIT_RATIO) / poseW, (h * FIT_RATIO) / poseH);
  const offX = (w - poseW * scale) / 2;
  const offY = (h - poseH * scale) / 2;
  const P = pts.map(([x, y]) => [(x - minX) * scale + offX, (y - minY) * scale + offY]);

  // Distinct pink, dashed — clearly different from the green/blue live skeleton.
  rctx.strokeStyle = "rgba(236, 72, 153, 0.9)";
  rctx.lineWidth = 4;
  rctx.setLineDash([10, 6]);
  for (const [a, b] of REF_CONNECTIONS) {
    if (P[a] && P[b]) {
      rctx.beginPath();
      rctx.moveTo(P[a][0], P[a][1]);
      rctx.lineTo(P[b][0], P[b][1]);
      rctx.stroke();
    }
  }
  rctx.setLineDash([]);
  rctx.fillStyle = "rgba(236, 72, 153, 1)";
  for (const [x, y] of P) {
    rctx.beginPath();
    rctx.arc(x, y, 5, 0, Math.PI * 2);
    rctx.fill();
  }
}

function setRefOverlay(on) {
  refOverlayOn = on;
  btnRefOverlay.textContent = on ? "Reference: On" : "Reference: Off";
  if (on) {
    drawReferenceOverlay();
  } else {
    clearReferenceOverlay();
  }
}

btnRefOverlay.addEventListener("click", () => {
  setRefOverlay(!refOverlayOn);
  if (refOverlayOn) {
    setStatus("Reference pose shown — align yourself to the pink skeleton.");
  }
});

if (referenceSelect) {
  referenceSelect.addEventListener("change", () => {
    if (refOverlayOn) drawReferenceOverlay();
  });
}

// --- Recording ----------------------------------------------------------
// Clicking Record starts a 3-second countdown, then begins recording.
btnRecord.addEventListener("click", () => {
  if (!stream || countdownId) return;
  btnRecord.disabled = true;

  let count = COUNTDOWN_SECONDS;
  countdownNum.textContent = count;
  countdownEl.classList.remove("hidden");
  setStatus("Get ready…");

  countdownId = setInterval(() => {
    count -= 1;
    if (count > 0) {
      countdownNum.textContent = count;
    } else {
      cancelCountdown();
      beginRecording();
    }
  }, 1000);
});

function cancelCountdown() {
  if (countdownId) clearInterval(countdownId);
  countdownId = null;
  countdownEl.classList.add("hidden");
}

function beginRecording() {
  if (!stream) {
    btnRecord.disabled = false;
    return;
  }
  chunks = [];
  const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
    ? "video/webm;codecs=vp9"
    : "video/webm";
  recorder = new MediaRecorder(stream, { mimeType: mime });

  recorder.ondataavailable = (e) => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };
  recorder.onstop = onRecordingStop;

  recorder.start();
  seconds = 0;
  timerEl.textContent = fmt(0);
  timerId = setInterval(() => {
    seconds += 1;
    timerEl.textContent = fmt(seconds);
    // Hard cap: stop automatically once the limit is reached.
    if (seconds >= MAX_VIDEO_SECONDS) {
      stopRecording();
      setStatus(`Reached the ${MAX_VIDEO_SECONDS}s limit — recording stopped. Trim and score it.`);
    }
  }, 1000);

  badge.classList.remove("hidden");
  btnRecord.disabled = true;
  btnStop.disabled = false;
  modeUpload.disabled = true
  modeUpload.classList.add('opacity-50', 'cursor-not-allowed');
  setStatus(`Recording… (max ${MAX_VIDEO_SECONDS}s)`);
}

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
  clearInterval(timerId);
  badge.classList.add("hidden");
  btnStop.disabled = true;
  btnRefOverlay.disabled = true;
  btnSkeleton.disabled = true;
  btnStartCam.disabled = true;
  
  btnStartCam.classList.add('opacity-50', 'cursor-not-allowed');
  emitStop();
}
btnStop.addEventListener("click", stopRecording);

// After recording stops, show trimmer before uploading
function revokePendingUrl() {
  if (pendingObjectUrl) {
    URL.revokeObjectURL(pendingObjectUrl);
    pendingObjectUrl = null;
  }
}

function onRecordingStop() {
  const blob = new Blob(chunks, { type: "video/webm" });
  pendingRecordingBlob = blob;
  pendingRecordingFilename = `recording-${Date.now()}.webm`;

  revokePendingUrl();
  const url = URL.createObjectURL(blob);
  pendingObjectUrl = url;
  setStatus("Trim your recording, then confirm to score it.");

  showTrimmer(url, function onConfirm() {
    revokePendingUrl();
    // User confirmed trim — upload the blob
    submitForScoring(pendingRecordingBlob, pendingRecordingFilename, () => {
      btnRecord.disabled = false;
    });
    pendingRecordingBlob = null;
  }, function onCancel() {
    revokePendingUrl();
    // User cancelled — allow re-recording
    btnRecord.disabled = false;
    btnStartCam.disabled = false;
    modeUpload.disabled = false;
    btnStartCam.classList.remove('opacity-50', 'cursor-not-allowed');
    modeUpload.classList.remove('opacity-50', 'cursor-not-allowed');
    pendingRecordingBlob = null;
    setStatus("Recording cancelled.");
  });
}

// Shared submit path for a live recording OR an uploaded video file.
async function submitForScoring(fileBlob, filename, onError) {
  const room = await socketReady;

  const fd = new FormData();
  fd.append("recording", fileBlob, filename);
  fd.append("reference_id", referenceSelect ? referenceSelect.value : "");
  fd.append("room", room || "");

  // Append trim values
  const si = document.getElementById("ftTrimStartInput");
  const ei = document.getElementById("ftTrimEndInput");
  if (si) fd.append("trim_start", si.value || "0");
  if (ei) fd.append("trim_end", ei.value || "0");

  progressWrap.classList.remove("hidden");
  setProgress(0.02, "Uploading…");
  setStatus("Processing your movement…");

  try {
    const res = await fetch(cfg.sessionsUrl, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Upload failed");
    // If no live socket, fall back to redirecting once processing likely done.
    if (!room) {
      setStatus("Uploaded. Open your dashboard to see the result.", "text-green-400");
    }
  } catch (err) {
    setStatus("Error: " + err.message, "text-red-400");
    progressWrap.classList.add("hidden");
    if (window.showAppError) window.showAppError(err.message);
    if (onError) onError();
  }
}

function setProgress(fraction, message) {
  progressBar.style.width = Math.round(fraction * 100) + "%";
  if (message) progressMsg.textContent = message;
}

// --- Mode switching: live camera vs. upload a prerecorded video --------
const MODE_ACTIVE = ["bg-gradient-to-r", "from-orange-500", "to-red-600", "text-white"];
const MODE_INACTIVE = ["bg-gray-800", "border", "border-gray-700", "text-gray-300"];

function applyModeButtons(live) {
  [modeLive, modeUpload].forEach((b) =>
    b.classList.remove(...MODE_ACTIVE, ...MODE_INACTIVE));
  modeLive.classList.add(...(live ? MODE_ACTIVE : MODE_INACTIVE));
  modeUpload.classList.add(...(live ? MODE_INACTIVE : MODE_ACTIVE));
}

function setMode(mode) {
  const live = mode === "live";
  livePanel.classList.toggle("hidden", !live);
  uploadPanel.classList.toggle("hidden", live);
  applyModeButtons(live);

  if (live) {
    setStatus(stream ? "Camera ready." : 'Click "Start Camera" to begin.');
  } else {
    stopCamera(); // release the webcam while in upload mode
    setStatus(cfg.hasReference
      ? "Pick a video to score against your reference."
      : "Upload a reference before scoring a video.");
  }
}

modeLive.addEventListener("click", () => setMode("live"));
modeUpload.addEventListener("click", () => setMode("upload"));

// --- Upload video panel with trimmer ---
videoFileInput.addEventListener("change", () => {
  const file = videoFileInput.files[0];
  videoFileName.textContent = file ? file.name : "Choose a prerecorded video";

  // Show trimmer for the selected file
  if (file) {
    revokePendingUrl();
    const url = URL.createObjectURL(file);
    pendingObjectUrl = url;

    // Check duration before showing trimmer
    var checkVideo = document.createElement('video');
    checkVideo.preload = 'metadata';
    checkVideo.src = url;
    checkVideo.addEventListener('loadedmetadata', function onMeta() {
      checkVideo.removeEventListener('loadedmetadata', onMeta);
      if (checkVideo.duration > MAX_VIDEO_SECONDS + 0.5) {
        URL.revokeObjectURL(url);
        pendingObjectUrl = null;
        videoFileInput.value = "";
        videoFileName.textContent = "Choose a prerecorded video";
        if (window.showAppError) {
          window.showAppError('Video exceeds the ' + MAX_VIDEO_SECONDS + '-second limit. Please choose a shorter clip.');
        }
        return;
      }
      setStatus("Trim your video, then confirm to score it.");

      showTrimmer(url, function onConfirm() {
        revokePendingUrl();
        // Trim confirmed — score immediately.
        submitForScoring(file, file.name, function onError() {
          // Re-enable the file input on error so user can retry
          setStatus("Upload failed. Try again.");
        });
      }, function onCancel() {
        revokePendingUrl();
        // Cancelled — clear file
        modeLive.disabled = false
        modeLive.classList.remove('opacity-50', 'cursor-not-allowed');
        videoFileInput.disabled = false
        videoFileInput.value = "";
        videoFileName.textContent = "Choose a prerecorded video";
        setStatus(cfg.hasReference
          ? "Pick a video to score against your reference."
          : "Upload a reference before scoring a video.");
      });
    });
  } else {
    destroyTrimmer();
  }
});

// --- Page unload: release all pending resources --------------------------
window.addEventListener("beforeunload", function () {
  if (stream) {
    stream.getTracks().forEach(function (t) { t.stop(); });
    stream = null;
  }
  destroyTrimmer();
  revokePendingUrl();
  pendingRecordingBlob = null;
});

// Start in live-camera mode.
setMode("live");

// --- Server progress events --------------------------------------------
window.addEventListener("ft:progress", (e) => {
  setProgress(e.detail.fraction, e.detail.message);
});
window.addEventListener("ft:done", (e) => {
  setProgress(1, "Done!");
  window.location.href = e.detail.redirect;
});
window.addEventListener("ft:error", (e) => {
  setStatus("Error: " + e.detail.message, "text-red-400");
  progressWrap.classList.add("hidden");
  btnRecord.disabled = false;
});