const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const overlayCanvas = document.getElementById('overlayCanvas');
const overlayCtx = overlayCanvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const alertBox = document.getElementById('alertBox');
const statusLabel = document.getElementById('statusLabel');
const arousalVal = document.getElementById('arousalVal');
const valenceVal = document.getElementById('valenceVal');

let stream = null;
let captureInterval = null;
let consecutiveLowFocus = 0;

// API Configurations
const API_URL = "http://localhost:8000/predict";
const FPS = 2; // Frames to capture per second

async function initCamera() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusLabel.textContent = "Monitoring...";

        // Wait for video metadata to set canvas size
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Match overlay canvas size to the video intrinsic size
            overlayCanvas.width = video.videoWidth;
            overlayCanvas.height = video.videoHeight;

            startTracking();
        };
    } catch (err) {
        console.error("Camera access denied!", err);
        alert("Please enable camera access for emotion tracking.");
    }
}

function stopTracking() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
    clearInterval(captureInterval);

    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusLabel.textContent = "Off";

    arousalVal.textContent = "0.00";
    valenceVal.textContent = "0.00";
    alertBox.classList.remove('active');

    // Clear overlay
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

async function sendFrame() {
    if (!stream) return;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.8);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: dataUrl,
                user_id: 1, // Mock User ID
                session_id: 1 // Mock Session ID
            })
        });

        const data = await response.json();
        if (data && data.arousal !== undefined && data.valence !== undefined) {
            updateDashboard(data);
        }
    } catch (error) {
        console.error("Error sending frame:", error);
    }
}

function updateDashboard(data) {
    const { arousal, valence, status, bbox } = data;

    arousalVal.textContent = arousal.toFixed(2);
    valenceVal.textContent = valence.toFixed(2);
    statusLabel.textContent = status;

    // Draw Bounding Box logic
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    if (bbox) {
        // Clear previous drawings
        const [x, y, w, h] = bbox;

        overlayCtx.strokeStyle = "#00FF00"; // Bright green
        overlayCtx.lineWidth = 4;
        overlayCtx.strokeRect(x, y, w, h);

        // Add status text above the box
        overlayCtx.fillStyle = "#00FF00";
        overlayCtx.font = "16px Arial";
        overlayCtx.fillText(status, x, y - 10);
    }

    // Simple logic for alerts: low arousal and negative valence
    if (arousal < 0.3 || valence < -0.3) {
        consecutiveLowFocus++;
        if (consecutiveLowFocus >= 3) {
            alertBox.classList.add('active');
        }
    } else {
        consecutiveLowFocus = 0;
        alertBox.classList.remove('active');
    }
}

function startTracking() {
    captureInterval = setInterval(() => {
        sendFrame();
    }, 1000 / FPS);
}

startBtn.addEventListener('click', initCamera);
stopBtn.addEventListener('click', stopTracking);