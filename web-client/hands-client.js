const INTERVAL_MS    = 400;
const ENABLE_OVERLAY = true;

const video   = document.getElementById('inputVideo');
const canvas  = document.getElementById('outputCanvas');
const ctx     = canvas.getContext('2d');
const logArea = document.getElementById('log');
let latestLandmarks = null;

function flattenLandmarks(lms) {
  return lms.flatMap(p => [p.x, p.y, p.z]);
}

function sendPayload(flat) {
  const payload = { landmarks: flat, ts: Date.now() };
  console.log('[payload]', payload);
  logArea.textContent =
    `sent (${flat.length} floats) @ ${new Date().toLocaleTimeString()}`;
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function drawFrameWithLandmarks(results) {
  ctx.save();
  clearCanvas();

  // Always draw the current video frame
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // Draw landmarks only if present
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    ctx.fillStyle = '#00ff00';
    const lms = results.multiHandLandmarks[0];
    lms.forEach(p =>
      ctx.fillRect(p.x * canvas.width - 2, p.y * canvas.height - 2, 4, 4)
    );
  }

  ctx.restore();
}

function onResults(results) {
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lms = results.multiHandLandmarks[0];
    latestLandmarks = flattenLandmarks(lms);
  } else {
    latestLandmarks = null;
  }

  if (ENABLE_OVERLAY) {
    drawFrameWithLandmarks(results);
  }
}

const hands = new Hands({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}`
});
hands.setOptions({
  maxNumHands: 1,
  modelComplexity: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
  staticImageMode: false
});
hands.onResults(onResults);

async function initCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    await video.play();
    logArea.textContent = 'camera ready – processing…';

    const mpCamera = new Camera(video, {
      onFrame: async () => await hands.send({ image: video }),
      width: 640,
      height: 480
    });
    mpCamera.start();
  } catch (err) {
    logArea.textContent = 'Camera permission denied.';
    console.error(err);
  }
}

setInterval(() => {
  if (latestLandmarks) sendPayload(latestLandmarks);
}, INTERVAL_MS);

initCamera();
