import cv2
import numpy as np
from flask import Flask, jsonify, send_from_directory, render_template_string
import os
import time
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

app = Flask(__name__)

# Static folder for saving pulse plot image
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

def bandpass_filter(signal, low=0.75, high=4.0, fs=30, order=5):
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_pulse(duration_sec=25, fs=30):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
    
    if not cap.isOpened():
        return None, "Cannot open webcam"
    
    green_signals = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB (or keep BGR if preferred)
        # Extract green channel mean as pulse signal proxy
        green_mean = np.mean(frame[:, :, 1])
        green_signals.append(green_mean)

        # Check if duration reached
        if time.time() - start_time > duration_sec:
            break

        # Optional: show video feed for debug (comment out in production)
        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    # cv2.destroyAllWindows()

    if len(green_signals) < fs * duration_sec * 0.8:
        return None, "Insufficient frames captured"

    # Process signal: detrend and bandpass filter
    signal = np.array(green_signals)
    signal = signal - np.mean(signal)
    filtered = bandpass_filter(signal, fs=fs)

    # Find peaks to estimate pulse rate
    peaks, _ = find_peaks(filtered, distance=fs*0.5)  # minimum 0.5 sec between peaks (max 120 bpm)
    num_peaks = len(peaks)
    bpm = (num_peaks / duration_sec) * 60

    # Plot signal and detected peaks
    plt.figure(figsize=(10, 4))
    plt.plot(filtered, label='Filtered Signal (Green Channel)')
    plt.plot(peaks, filtered[peaks], "x", label="Peaks")
    plt.title(f"Pulse Detection - Estimated BPM: {bpm:.1f}")
    plt.xlabel("Frames")
    plt.ylabel("Signal Intensity")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(STATIC_FOLDER, 'pulse_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return round(bpm, 1), None

@app.route('/')
def index():
    # Minimal frontend, you can replace with your full HTML page
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Pulse Detection</title>
  <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-b from-green-100 to-green-300 min-h-screen flex flex-col justify-center items-center font-sans">
  <div class="bg-white rounded-2xl shadow-xl p-10 max-w-xl w-full text-center">
    <h1 class="text-4xl font-extrabold text-green-700 mb-6">🫀 Pulse Detection via Webcam</h1>
    <p class="text-gray-600 mb-8">Click below to detect your pulse using your face. Remain still and face the camera.</p>
    <button id="startBtn" class="bg-green-600 hover:bg-green-700 text-white text-lg font-semibold py-3 px-6 rounded-full transition duration-300 ease-in-out">▶️ Start Measurement</button>
    <div id="loader" class="mt-6 hidden">
      <div class="flex justify-center">
        <svg class="animate-spin h-8 w-8 text-green-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
          <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
        </svg>
      </div>
      <p class="text-green-700 mt-4">Analyzing... Please wait 25 seconds</p>
    </div>
    <div id="result" class="mt-6 text-xl font-medium text-gray-800 hidden"></div>
    <div class="mt-4">
      <img id="pulsePlot" src="" alt="Pulse Plot" class="rounded-lg shadow-md mx-auto hidden" width="450" />
    </div>
  </div>

  <script>
    const startBtn = document.getElementById('startBtn');
    const loader = document.getElementById('loader');
    const result = document.getElementById('result');
    const pulsePlot = document.getElementById('pulsePlot');

    startBtn.addEventListener('click', () => {
      result.classList.add('hidden');
      pulsePlot.classList.add('hidden');
      loader.classList.remove('hidden');

      fetch('/start')
        .then(res => res.json())
        .then(data => {
          loader.classList.add('hidden');
          if (data.status === 'success') {
            result.textContent = `✅ Estimated BPM: ${data.bpm}`;
            result.classList.remove('hidden');
            pulsePlot.src = '/static/pulse_plot.png?' + new Date().getTime();
            pulsePlot.classList.remove('hidden');
          } else {
            result.textContent = `⚠️ Error: ${data.message}`;
            result.classList.remove('hidden');
          }
        })
        .catch(err => {
          loader.classList.add('hidden');
          result.textContent = `❌ Request failed: ${err}`;
          result.classList.remove('hidden');
        });
    });
  </script>
</body>
</html>
    """)

@app.route('/start')
def start():
    bpm, error = detect_pulse(duration_sec=25, fs=30)
    if bpm is not None:
        return jsonify(status='success', bpm=bpm)
    else:
        return jsonify(status='error', message=error or "Unknown error")

# Serve static files (pulse_plot.png)
@app.route('/static/<path:path>')
def static_files(path):
    return send_from_directory(STATIC_FOLDER, path)

if __name__ == '__main__':
    app.run(debug=True)
