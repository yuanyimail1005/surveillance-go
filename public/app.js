const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const wsBase = `${proto}://${location.host}`;

const videoImg = document.getElementById('video');
const videoOverlay = document.getElementById('videoOverlay');
const faceStatus = document.getElementById('faceStatus');
const cameraDevice = document.getElementById('cameraDevice');
const resolutionSelect = document.getElementById('resolution');
const fpsInput = document.getElementById('fps');
const microphoneSelect = document.getElementById('microphone');
const speakerSelect = document.getElementById('speaker');
const volumeInput = document.getElementById('volume');

let audioCtx;
let audioSocket;
let talkSocket;
let mediaRecorder;
const cameraResolutionsByDevice = new Map();
let currentVideoObjectUrl = null;
let pendingVideoBlob = null;
let videoDecodeInFlight = false;
let nextAudioPlayTime = 0;
let droppedAudioChunks = 0;
let speakerMeterSmooth = 0;
let micMeterSmooth = 0;
let latestFaces = [];
let latestFaceImageSize = { width: 0, height: 0 };

const AUDIO_MIN_LEAD_SECONDS = 0.02;
const AUDIO_MAX_BUFFER_SECONDS = 0.4;

const syncVideoOverlaySize = () => {
  const w = videoImg.clientWidth;
  const h = videoImg.clientHeight;
  if (!w || !h) return;
  if (videoOverlay.width !== w) videoOverlay.width = w;
  if (videoOverlay.height !== h) videoOverlay.height = h;
};

/**
 * Draw server-sent face boxes onto the overlay canvas.
 * faces: array of { name, confidence, left, top, right, bottom } in image coordinates.
 * imageWidth/Height: original JPEG dimensions used to scale into display dimensions.
 */
const drawFaceOverlay = (faces, imageWidth, imageHeight) => {
  const ctx = videoOverlay.getContext('2d');
  if (!ctx) return;
  syncVideoOverlaySize();
  ctx.clearRect(0, 0, videoOverlay.width, videoOverlay.height);
  if (!Array.isArray(faces) || !faces.length) return;

  const dw = videoOverlay.width;
  const dh = videoOverlay.height;
  const scaleX = imageWidth > 0 ? dw / imageWidth : 1;
  const scaleY = imageHeight > 0 ? dh / imageHeight : 1;

  ctx.lineWidth = 2;
  ctx.font = '13px sans-serif';

  for (const face of faces) {
    const x = Math.max(0, Math.round(face.left * scaleX));
    const y = Math.max(0, Math.round(face.top * scaleY));
    const w = Math.max(1, Math.round((face.right - face.left) * scaleX));
    const h = Math.max(1, Math.round((face.bottom - face.top) * scaleY));
    const isKnown = face.name && face.name !== 'Unknown';
    const colour = isKnown ? '#47d169' : '#f0a500';
    const label = isKnown
      ? `${face.name} (${Math.round(face.confidence * 100)}%)`
      : 'Unknown';

    ctx.strokeStyle = colour;
    ctx.fillStyle = isKnown ? 'rgba(71, 209, 105, 0.12)' : 'rgba(240, 165, 0, 0.12)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeRect(x, y, w, h);

    const textW = Math.ceil(ctx.measureText(label).width) + 10;
    const textX = x;
    const textY = Math.max(16, y - 4);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.78)';
    ctx.fillRect(textX, textY - 14, textW, 17);
    ctx.fillStyle = isKnown ? '#ddffe6' : '#ffe9b0';
    ctx.fillText(label, textX + 5, textY - 1);
  }
};

/** Handle a face_data JSON message pushed by the server over the video WebSocket. */
const handleServerFaceData = (msg) => {
  const result = msg.result || {};
  latestFaceImageSize = { width: result.image_width || 0, height: result.image_height || 0 };
  latestFaces = Array.isArray(result.faces) ? result.faces : [];
  drawFaceOverlay(latestFaces, latestFaceImageSize.width, latestFaceImageSize.height);

  if (!msg.enabled) {
    setFaceStatus('Face AI: disabled');
  } else if (!msg.available) {
    setFaceStatus(`Face AI: ${msg.message || 'unavailable'}`);
  } else {
    const n = latestFaces.length;
    const knownStr = msg.known_faces_count > 0 ? ` · ${msg.known_faces_count} known` : '';
    setFaceStatus(`Face AI: ready${knownStr} · ${n} face${n === 1 ? '' : 's'} detected`);
  }
};

const setFaceStatus = (text) => {
  if (faceStatus) faceStatus.textContent = text;
};

const renderLatestVideoFrame = () => {
  if (videoDecodeInFlight || !pendingVideoBlob) return;

  videoDecodeInFlight = true;
  const frameBlob = pendingVideoBlob;
  pendingVideoBlob = null;

  const nextUrl = URL.createObjectURL(frameBlob);
  const previousUrl = currentVideoObjectUrl;

  videoImg.onload = () => {
    if (previousUrl) {
      URL.revokeObjectURL(previousUrl);
    }
    syncVideoOverlaySize();
    drawFaceOverlay(latestFaces, latestFaceImageSize.width, latestFaceImageSize.height);
    videoDecodeInFlight = false;
    if (pendingVideoBlob) {
      renderLatestVideoFrame();
    }
  };

  videoImg.onerror = () => {
    URL.revokeObjectURL(nextUrl);
    if (previousUrl) {
      URL.revokeObjectURL(previousUrl);
    }
    videoDecodeInFlight = false;
    if (pendingVideoBlob) {
      renderLatestVideoFrame();
    }
  };

  currentVideoObjectUrl = nextUrl;
  videoImg.src = nextUrl;
};

const fetchJson = async (url, opts) => {
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.message || `HTTP ${res.status}`);
  return data;
};

const startVideo = () => {
  const ws = new WebSocket(`${wsBase}/video_feed`);
  ws.binaryType = 'blob';
  ws.onmessage = (event) => {
    if (event.data instanceof Blob) {
      // Binary JPEG frame — keep only the newest while decode/render is busy.
      pendingVideoBlob = event.data;
      renderLatestVideoFrame();
    } else if (typeof event.data === 'string') {
      // JSON message from the server (face_data, etc.).
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'face_data') handleServerFaceData(msg);
      } catch (_) {}
    }
  };
  ws.onclose = () => {
    latestFaces = [];
    latestFaceImageSize = { width: 0, height: 0 };
    drawFaceOverlay(latestFaces, 0, 0);
    setFaceStatus('Face AI: server not connected');
    pendingVideoBlob = null;
    videoDecodeInFlight = false;
    if (currentVideoObjectUrl) {
      URL.revokeObjectURL(currentVideoObjectUrl);
      currentVideoObjectUrl = null;
    }
  };
};

const startAudio = () => {
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
  nextAudioPlayTime = audioCtx.currentTime;
  audioSocket = new WebSocket(`${wsBase}/audio_feed`);
  audioSocket.binaryType = 'arraybuffer';
  audioSocket.onmessage = async (event) => {
    const sampleCount = Math.floor(event.data.byteLength / 2);
    if (sampleCount <= 0) return;
    const data = new Int16Array(event.data, 0, sampleCount);
    playAudioChunk(data);
  };
  audioSocket.onclose = () => {
    resetVolumeMeter('speaker-volume-bar', 'speaker-volume-text', 'speaker');
  };
};

const rmsToPercent = (rms) => {
  const db = rms > 0 ? 20 * Math.log10(rms) : -100;
  return Math.min(100, Math.max(0, Math.round(((db + 70) / 60) * 100)));
};

const volumeLabel = (percent) => {
  if (percent < 5) return 'Silent';
  if (percent < 25) return 'Low';
  if (percent < 75) return 'Good';
  return 'High';
};

const updateVolumeMeter = (barId, textId, rawPercent, meterType) => {
  const bar = document.getElementById(barId);
  const text = document.getElementById(textId);
  if (!bar || !text) return;

  const clamped = Math.min(100, Math.max(0, rawPercent));
  const prev = meterType === 'speaker' ? speakerMeterSmooth : micMeterSmooth;
  const smooth = Math.round(prev * 0.72 + clamped * 0.28);
  
  if (meterType === 'speaker') {
    speakerMeterSmooth = smooth;
  } else {
    micMeterSmooth = smooth;
  }

  bar.style.width = smooth + '%';
  text.textContent = `${smooth}% · ${volumeLabel(smooth)}`;
};

const resetVolumeMeter = (barId, textId, meterType) => {
  if (meterType === 'speaker') {
    speakerMeterSmooth = 0;
  } else {
    micMeterSmooth = 0;
  }
  updateVolumeMeter(barId, textId, 0, meterType);
};

const playAudioChunk = (int16Array) => {
  const buffer = audioCtx.createBuffer(1, int16Array.length, 48000);
  const channel = buffer.getChannelData(0);
  
  let sumSq = 0;
  for (let i = 0; i < int16Array.length; i += 1) {
    let s = int16Array[i] / 32768;
    if (s > 0.98) s = 0.98;
    if (s < -0.98) s = -0.98;
    channel[i] = s;
    sumSq += s * s;
  }
  
  const rms = Math.sqrt(sumSq / int16Array.length);
  const volumePercent = rmsToPercent(rms);
  updateVolumeMeter('speaker-volume-bar', 'speaker-volume-text', volumePercent, 'speaker');

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(audioCtx.destination);

  const now = audioCtx.currentTime;
  if (nextAudioPlayTime > now + AUDIO_MAX_BUFFER_SECONDS) {
    // Drop stale audio chunks when playback queue drifts too far behind live time.
    nextAudioPlayTime = now + AUDIO_MIN_LEAD_SECONDS;
    droppedAudioChunks += 1;
    if (droppedAudioChunks % 20 === 0) {
      console.debug(`Dropped ${droppedAudioChunks} stale audio chunks`);
    }
    return;
  }

  if (nextAudioPlayTime < now + AUDIO_MIN_LEAD_SECONDS) {
    nextAudioPlayTime = now + AUDIO_MIN_LEAD_SECONDS;
  }

  source.start(nextAudioPlayTime);
  nextAudioPlayTime += buffer.duration;
};

const fillResolutionOptions = (resolutions, selectedWidth, selectedHeight) => {
  resolutionSelect.innerHTML = '';
  if (!Array.isArray(resolutions) || !resolutions.length) return;

  const hasExactSelection = Number.isFinite(selectedWidth)
    && Number.isFinite(selectedHeight)
    && resolutions.some((item) => item.width === selectedWidth && item.height === selectedHeight);

  let closestIndex = 0;
  if (!hasExactSelection && Number.isFinite(selectedWidth) && Number.isFinite(selectedHeight)) {
    let bestScore = Number.POSITIVE_INFINITY;
    for (let i = 0; i < resolutions.length; i += 1) {
      const item = resolutions[i];
      const dw = item.width - selectedWidth;
      const dh = item.height - selectedHeight;
      const score = (dw * dw) + (dh * dh);
      if (score < bestScore) {
        bestScore = score;
        closestIndex = i;
      }
    }
  }

  for (const item of resolutions) {
    const option = document.createElement('option');
    option.value = `${item.width}x${item.height}`;
    option.textContent = `${item.width}x${item.height}`;
    if (item.width === selectedWidth && item.height === selectedHeight) option.selected = true;
    resolutionSelect.appendChild(option);
  }

  if (resolutionSelect.selectedIndex === -1) {
    resolutionSelect.selectedIndex = closestIndex;
  }
};

const getSelectedResolution = () => {
  const [width, height] = String(resolutionSelect.value || '')
    .split('x')
    .map((value) => Number.parseInt(value, 10));
  return { width, height };
};

const updateResolutionForSelectedCamera = (selectedWidth, selectedHeight, fallbackResolutions) => {
  const resolvedWidth = Number.isFinite(selectedWidth) ? selectedWidth : getSelectedResolution().width;
  const resolvedHeight = Number.isFinite(selectedHeight) ? selectedHeight : getSelectedResolution().height;
  const resolutions = cameraResolutionsByDevice.get(cameraDevice.value)
    || fallbackResolutions
    || [];
  fillResolutionOptions(resolutions, resolvedWidth, resolvedHeight);
};

const loadCameraSettings = async () => {
  const data = await fetchJson('/camera_settings');
  cameraResolutionsByDevice.clear();
  cameraDevice.innerHTML = '';
  for (const item of data.available_camera_devices) {
    const option = document.createElement('option');
    option.value = item.path;
    option.textContent = item.name;
    if (item.path === data.selected_camera_device) option.selected = true;
    cameraDevice.appendChild(option);
    cameraResolutionsByDevice.set(item.path, item.supported_resolutions || []);
  }
  updateResolutionForSelectedCamera(data.width, data.height, data.supported_resolutions || data.allowed_resolutions || []);
  fpsInput.value = data.fps;
};

const loadAudioDevices = async () => {
  const data = await fetchJson('/server_audio_devices');
  microphoneSelect.innerHTML = '';
  data.microphones.forEach((mic) => {
    const option = document.createElement('option');
    option.value = mic.id;
    option.textContent = mic.name;
    if (mic.id === data.selected_microphone) option.selected = true;
    microphoneSelect.appendChild(option);
  });

  speakerSelect.innerHTML = '';
  data.speakers.forEach((spk) => {
    const option = document.createElement('option');
    option.value = spk.id;
    option.textContent = spk.name;
    if (spk.id === data.selected_speaker) option.selected = true;
    speakerSelect.appendChild(option);
  });
};

const updateVolumeDisplay = () => {
  const vol = Number(volumeInput.value) || 0;
  const display = document.getElementById('currentVolumeDisplay');
  if (display) {
    display.textContent = `${vol}% (Server Volume)`;
  }
};

const loadVolume = async () => {
  const data = await fetchJson('/speaker_volume');
  volumeInput.value = data.volume;
  updateVolumeDisplay();
};

document.getElementById('saveCamera').onclick = async () => {
  const [width, height] = String(resolutionSelect.value || '').split('x').map((value) => Number.parseInt(value, 10));
  if (!Number.isFinite(width) || !Number.isFinite(height)) return;
  await fetchJson('/camera_settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      camera_device: cameraDevice.value,
      width,
      height,
      fps: Number(fpsInput.value)
    })
  });
};

cameraDevice.onchange = () => {
  updateResolutionForSelectedCamera();
};

document.getElementById('saveAudioDevices').onclick = async () => {
  await fetchJson('/server_audio_devices/select', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      microphone: microphoneSelect.value,
      speaker: speakerSelect.value
    })
  });
};

document.getElementById('saveVolume').onclick = async () => {
  await fetchJson('/speaker_volume', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ volume: Number(volumeInput.value) })
  });
  updateVolumeDisplay();
};

volumeInput.oninput = () => {
  updateVolumeDisplay();
};

let isTalking = false;
let micStream = null;
let micSourceNode = null;
let scriptProcessor = null;

const startTalk = async () => {
  try {
    // Get microphone stream
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: false,
        channelCount: 1,
        sampleRate: 48000
      }
    });

    // Create WebSocket for sending audio
    talkSocket = new WebSocket(`${wsBase}/ws/talk`);
    talkSocket.binaryType = 'arraybuffer';
    
    talkSocket.onerror = (error) => {
      console.error('Talk socket error:', error);
      stopTalk();
    };
    
    talkSocket.onclose = () => {
      console.log('Talk socket closed');
      if (isTalking) stopTalk();
    };

    // Create audio context if not exists
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    }

    // Create media stream source and script processor for raw audio capture
    micSourceNode = audioCtx.createMediaStreamSource(micStream);
    scriptProcessor = audioCtx.createScriptProcessor(4096, 1, 1);

    let sendCount = 0;
    scriptProcessor.onaudioprocess = (event) => {
      if (!isTalking || talkSocket.readyState !== WebSocket.OPEN) return;

      const inputData = event.inputBuffer.getChannelData(0);
      
      // Calculate RMS for microphone meter
      let sumSq = 0;
      for (let i = 0; i < inputData.length; i++) {
        sumSq += inputData[i] * inputData[i];
      }
      const rms = Math.sqrt(sumSq / inputData.length);
      const volumePercent = rmsToPercent(rms);
      updateVolumeMeter('mic-volume-bar', 'mic-volume-text', volumePercent, 'mic');

      // Convert float32 to int16
      const int16 = new Int16Array(inputData.length);
      for (let i = 0; i < inputData.length; i++) {
        let s = Math.max(-1, Math.min(1, inputData[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }

      // Send to server
      try {
        talkSocket.send(int16.buffer);
      } catch (e) {
        console.error('Failed to send audio chunk:', e);
      }

      sendCount++;
      if (sendCount % 10 === 0) {
        console.debug(`Sending microphone: ${volumePercent}%`);
      }
    };

    micSourceNode.connect(scriptProcessor);
    scriptProcessor.connect(audioCtx.destination);

    isTalking = true;
    document.getElementById('talkToggle').textContent = 'Stop Talk';
    console.log('Microphone capture started');
  } catch (error) {
    console.error('Failed to start talkback:', error);
    alert(`Microphone error: ${error.message}`);
    stopTalk();
  }
};

const stopTalk = () => {
  isTalking = false;
  
  // Clean up audio nodes
  if (scriptProcessor) {
    scriptProcessor.onaudioprocess = null;
    scriptProcessor.disconnect();
    scriptProcessor = null;
  }
  
  if (micSourceNode) {
    micSourceNode.disconnect();
    micSourceNode = null;
  }
  
  // Stop all microphone tracks
  if (micStream) {
    micStream.getTracks().forEach(track => track.stop());
    micStream = null;
  }
  
  // Close WebSocket
  if (talkSocket && (talkSocket.readyState === WebSocket.OPEN || talkSocket.readyState === WebSocket.CONNECTING)) {
    talkSocket.close();
    talkSocket = null;
  }
  
  // Reset microphone meter
  resetVolumeMeter('mic-volume-bar', 'mic-volume-text', 'mic');
  document.getElementById('talkToggle').textContent = 'Hold to Talk';
  console.log('Microphone capture stopped');
};

document.getElementById('talkToggle').onclick = async () => {
  if (isTalking) {
    stopTalk();
  } else {
    await startTalk();
  }
};

(async () => {
  startVideo();
  startAudio();
  await loadCameraSettings();
  await loadAudioDevices();
  await loadVolume();
})();

window.addEventListener('resize', () => {
  syncVideoOverlaySize();
  drawFaceOverlay(latestFaces, latestFaceImageSize.width, latestFaceImageSize.height);
});
