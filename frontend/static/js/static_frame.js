import {
  FilesetResolver,
  HandLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0";

const video     = document.getElementById("cam");
const overlay   = document.getElementById("overlay");
const ctx       = overlay.getContext("2d");
const statusEl  = document.getElementById("status");

// ROI size as a fraction of the video frame (centered box)
const BOX_W_RATIO = 0.50;
const BOX_H_RATIO = 0.50;

// MediaPipe state
let handLandmarker = null;
let running = false;

/**
 * Compute the centered ROI box in video/canvas pixel space.
 */
function getBox() {
  const w = overlay.width, h = overlay.height;
  const cx = (w / 2) | 0, cy = (h / 2) | 0;
  const bw = (w * BOX_W_RATIO) | 0, bh = (h * BOX_H_RATIO) | 0;
  const x1 = (cx - bw / 2) | 0, y1 = (cy - bh / 2) | 0;
  const x2 = x1 + bw, y2 = y1 + bh;
  return { x1, y1, x2, y2, bw, bh, cx, cy };
}

/**
 * Provide a static ROI box (used by capture code to crop).
 * Coordinates are in the same pixel space as the video/canvas.
 */
window.__staticRoiBox = function () {
  const { x1, y1, bw, bh } = getBox();
  return { x1, y1, bw, bh };
};

/**
 * Draw the overlay: dim background, cutout box, brackets, crosshair, and hint text.
 * The frame turns green when the detected nail center sits inside the box.
 */
function drawBox(inBox) {
  if (!ctx) return;

  const { x1, y1, x2, y2, cx, cy, bw, bh } = getBox();

  // Clear the entire canvas
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  // Dimmed backdrop with a transparent cutout (the ROI)
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.35)";
  ctx.fillRect(0, 0, overlay.width, overlay.height);
  ctx.clearRect(x1, y1, bw, bh);
  ctx.restore();

  // Frame color: green if in-box, otherwise white
  ctx.strokeStyle = inBox ? "#1db954" : "#ffffff";
  ctx.lineWidth = 4;
  const c = 22; // corner length

  // Corner brackets (top-left, top-right, bottom-left, bottom-right)
  ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x1 + c, y1); ctx.moveTo(x1, y1); ctx.lineTo(x1, y1 + c); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x2, y1); ctx.lineTo(x2 - c, y1); ctx.moveTo(x2, y1); ctx.lineTo(x2, y1 + c); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x1, y2); ctx.lineTo(x1 + c, y2); ctx.moveTo(x1, y2); ctx.lineTo(x1, y2 - c); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - c, y2); ctx.moveTo(x2, y2); ctx.lineTo(x2, y2 - c); ctx.stroke();

  // Crosshair at center
  ctx.beginPath();
  ctx.moveTo(cx - 24, cy); ctx.lineTo(cx + 24, cy);
  ctx.moveTo(cx, cy - 24); ctx.lineTo(cx, cy + 24);
  ctx.stroke();

  // Hint text
  ctx.font = "18px system-ui, -apple-system, Segoe UI, Roboto";
  ctx.fillStyle = inBox ? "#1db954" : "#ffffff";
  ctx.fillText("Put your nail into the frame!", cx - 110, y1 - 12);
}

/**
 * Decide if the nail “center” (midpoint of landmarks 7 and 8) is inside the ROI.
 * landmarks come normalized in [0,1] relative to the video size.
 */
function nailInBox(landmarks, w, h, box) {
  if (!landmarks) return false;

  const p7 = landmarks[7]; // base of index fingertip
  const p8 = landmarks[8]; // index fingertip tip
  const nailX = ((p7.x + p8.x) / 2) * w;
  const nailY = ((p7.y + p8.y) / 2) * h;

  return nailX >= box.x1 && nailX <= box.x2 && nailY >= box.y1 && nailY <= box.y2;
}

/**
 * Per-frame loop: run landmark detection and redraw the overlay.
 */
async function loop() {
  if (!running) return;

  const now = performance.now();
  const results = handLandmarker.detectForVideo(video, now);
  const box = getBox();

  let inBox = false;
  if (results.landmarks && results.landmarks[0]) {
    inBox = nailInBox(results.landmarks[0], overlay.width, overlay.height, box);
  }
  drawBox(inBox);

  requestAnimationFrame(loop);
}

/**
 * Initialize camera, size the overlay to video resolution, and create the hand landmarker.
 */
async function init() {
  // Request camera stream
  const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                  facingMode: { ideal: "environment" }
                }
              });

  video.srcObject = stream;

  // Ensure metadata is loaded so we know the actual dimensions
  await new Promise((resolve) => (video.onloadedmetadata = resolve));
  await video.play();

  // Match overlay canvas size to the video’s intrinsic resolution
  overlay.width = video.videoWidth;
  overlay.height = video.videoHeight;

  // (Optional) For high-DPI displays, scale the drawing by devicePixelRatio:
  // const dpr = window.devicePixelRatio || 1;
  // overlay.width = Math.round(video.videoWidth * dpr);
  // overlay.height = Math.round(video.videoHeight * dpr);
  // overlay.style.width = video.videoWidth + "px";
  // overlay.style.height = video.videoHeight + "px";
  // ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  // Load MediaPipe Tasks runtime and hand model
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "/static/models/hand_landmarker.task" },
    runningMode: "VIDEO",
    numHands: 1,
  });

  running = true;
  if (statusEl) statusEl.textContent = "Camera ready. Align your nail in the box.";
  requestAnimationFrame(loop);
}

// Bootstrap and basic error reporting
init().catch((e) => {
  console.error(e);
  if (statusEl) statusEl.textContent = "Init failed.";
});
