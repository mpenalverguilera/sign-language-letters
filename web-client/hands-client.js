/* -------------------------------------------------------------
    ASL Sign‑to‑Text – client logic
    • Every 500 ms compare current landmarks with the last sent.
    • If L2 distance > THRESHOLD send to backend.
    • Only accept predictions whose confidence ≥ CONF_THRESHOLD.
    • Debounce so we don't append same letter repeatedly.
--------------------------------------------------------------*/
const BACKEND_URL    = "https://l3eygbhqj0.execute-api.eu-west-3.amazonaws.com/Prod/predict";
const SEND_EVERY_MS  = 500;      // interval between attempts
const THRESHOLD      = 0.01;     // mean L2 distance threshold
const CONF_THRESHOLD = 0.75;     // min confidence to accept
const DEBOUNCE_MS    = 2000;      // ignore duplicate predictions within this window

const video   = document.getElementById('inputVideo');
const canvas  = document.getElementById('outputCanvas');
const ctx     = canvas.getContext('2d');
const logArea = document.getElementById('log');
const typed   = document.getElementById('typedText');

let lastFrameLandmarks = null;   // latest 63‑float array from MediaPipe
let lastSentLandmarks  = null;   // landmarks sent to backend previously
let lastAccepted       = null;   // last accepted letter
let lastAcceptTime     = 0;      // time in ms

/* -------- helper functions -------- */
const flatten = lms => lms.flatMap(p => [p.x, p.y, p.z]);
const l2 = (a, b) => {
  if (!a || !b || a.length !== b.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum / a.length);
};

async function sendPayload(flat) {
  try {
    const res  = await fetch(BACKEND_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks: flat })
    });
    const { prediction, confidence } = await res.json();
    handlePrediction(prediction, confidence);
    lastSentLandmarks = flat;
  } catch (err) {
    logArea.textContent = `❌ ${err.message}`;
    console.error(err);
  }
}

function appendLetter(letter) {
  if (letter === 'space') {
    typed.textContent += ' ';
  } else if (letter === 'del') {
    typed.textContent = typed.textContent.slice(0, -1);
  } else {
    typed.textContent += letter;
  }
}

function handlePrediction(prediction, confidence) {
  if (confidence >= CONF_THRESHOLD && prediction) {
    const now = Date.now();
    if (prediction !== lastAccepted || now - lastAcceptTime > DEBOUNCE_MS) {
      appendLetter(prediction);
      lastAccepted   = prediction;
      lastAcceptTime = now;
    }
    logArea.textContent = `✔️ ${new Date().toLocaleTimeString()} – ${prediction} (${confidence.toFixed(2)})`;
  } else {
    logArea.textContent = `ℹ️ ${new Date().toLocaleTimeString()} – ${prediction} with low confidence (${confidence?.toFixed(2) ?? 'n/a'})`;
  }
}

/* -------- drawing helpers -------- */
function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
function drawLandmarks(lms, color) {
  ctx.fillStyle = color;
  lms.forEach(p => ctx.fillRect(p.x * canvas.width - 2, p.y * canvas.height - 2, 4, 4));
}

/* -------- MediaPipe callback -------- */
function onResults(res) {
  lastFrameLandmarks = res.multiHandLandmarks?.length ? flatten(res.multiHandLandmarks[0]) : null;

  // overlay always draws current video frame
  ctx.save();
  clearCanvas();
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  if (res.multiHandLandmarks?.length) drawLandmarks(res.multiHandLandmarks[0], '#00e676');
  ctx.restore();
}

/* -------- MediaPipe setup -------- */
const hands = new Hands({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` });
hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.75, minTrackingConfidence: 0.5, staticImageMode: false });
hands.onResults(onResults);

/* -------- Camera -------- */
async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
    logArea.textContent = 'Camera ready – detecting…';

    const cam = new Camera(video, { width: 720, height: 540, onFrame: async () => hands.send({ image: video }) });
    cam.start();
  } catch (err) {
    logArea.textContent = 'Camera permission denied.';
    console.error(err);
  }
}

/* -------- Periodic decision -------- */
setInterval(() => {
  if (!lastFrameLandmarks) return; // no hand
  const delta = l2(lastFrameLandmarks, lastSentLandmarks);
  if (delta > THRESHOLD) {
    sendPayload(lastFrameLandmarks);
  }
}, SEND_EVERY_MS);

initCamera();