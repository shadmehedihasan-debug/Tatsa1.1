/*
 * Real-time object detection and tracking in the browser.
 *
 *  1. Video input:   webcam (getUserMedia) or a local video file
 *  2. Detection:     pre-trained YOLOv8n model (onnxruntime-web)
 *  3. Per frame:     detect objects and get bounding boxes
 *  4. Tracking:      SORT (see sort.js)
 *  5. Display:       boxes, class labels and tracking IDs drawn on a canvas
 */
(() => {
  'use strict';

  const $ = (sel) => document.querySelector(sel);
  const video = $('#video');
  const canvas = $('#canvas');
  const ctx = canvas.getContext('2d'); // transparent overlay: boxes, labels, trails only
  const cap = document.createElement('canvas'); // offscreen frame grab sent to the detector
  const cctx = cap.getContext('2d');
  const emptyState = $('#empty');
  const messageEl = $('#message');
  const btnWebcam = $('#btn-webcam');
  const btnFile = $('#btn-file');
  const btnStop = $('#btn-stop');
  const fileInput = $('#file-input');
  const trackedList = $('#tracked');

  const LOW_CONF = 0.1; // weak detections kept only to rescue existing tracks
  const settings = { conf: 0.4, trails: true };
  const tracker = new Sort({ maxAge: 30, minHits: 2, iouThreshold: 0.3, highThresh: 0.4 });

  let model = null;
  let running = false;
  let runId = 0;
  let stream = null;
  let fileUrl = null;

  /* ---------- Helpers ---------- */
  function setMessage(text, isError = false) {
    messageEl.textContent = text;
    messageEl.classList.toggle('error', isError);
  }

  function nextFrame() {
    return new Promise((resolve) => requestAnimationFrame(resolve));
  }

  function colorFor(id) {
    return `hsl(${Math.round((id * 137.508) % 360)} 80% 45%)`;
  }

  function setSourceButtons(enabled) {
    btnWebcam.disabled = !enabled;
    btnFile.disabled = !enabled;
  }

  /* ---------- Model ---------- */
  async function loadModel() {
    model = null;
    setSourceButtons(false);
    setMessage('Loading detection model…');
    try {
      model = await Yolo.load('yolov8n.onnx');
      $('#stat-backend').textContent = model.backend;
      setMessage(running ? 'Model ready.' : 'Model ready. Choose a video source to start.');
      setSourceButtons(true);
    } catch (err) {
      console.error(err);
      setMessage('The detection model could not be loaded (' + (err && err.message ? err.message : err) + '). Make sure yolov8n.onnx is in the project folder, you are online, and the page is served over http(s) rather than opened as a file, then reload.', true);
    }
  }

  /* ---------- Video sources ---------- */
  function stopSource() {
    running = false;
    runId++;
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.pause();
    video.srcObject = null;
    video.removeAttribute('src');
    video.load();
    if (fileUrl) {
      URL.revokeObjectURL(fileUrl);
      fileUrl = null;
    }
  }

  function showIdle() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.hidden = true;
    emptyState.hidden = false;
    btnStop.disabled = true;
    renderTracked([]);
    $('#stat-fps').textContent = '0';
    $('#stat-count').textContent = '0';
  }

  async function startSource(kind, file) {
    stopSource();
    tracker.reset();
    $('#stat-ids').textContent = '0';

    try {
      if (kind === 'webcam') {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw Object.assign(new Error('unsupported'), { name: 'NotSupportedError' });
        }
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        video.srcObject = stream;
        video.loop = false;
      } else {
        fileUrl = URL.createObjectURL(file);
        video.srcObject = null;
        video.src = fileUrl;
        video.loop = true;
      }
      video.muted = true;
      await video.play();
    } catch (err) {
      console.error(err);
      stopSource();
      showIdle();
      const messages = {
        NotAllowedError: 'Camera access was blocked. Allow the camera in your browser’s address bar, or open a video file instead.',
        NotFoundError: 'No camera was found on this device. Open a video file instead.',
        NotSupportedError: 'Camera access needs a secure (https) page. Open a video file instead.',
      };
      setMessage(messages[err.name] || 'The video could not be played. Try a different file or source.', true);
      return;
    }

    running = true;
    canvas.hidden = false;
    emptyState.hidden = true;
    btnStop.disabled = false;
    setMessage(kind === 'webcam' ? 'Tracking from your webcam.' : `Tracking ${file.name}.`);
    detectLoop(++runId);
  }

  /* ---------- Main loop ---------- */
  async function detectLoop(id) {
    let fps = 0;
    let last = performance.now();
    let lastUi = 0;

    while (running && id === runId) {
      if (!model || video.readyState < 2 || !video.videoWidth) {
        await nextFrame();
        continue;
      }

      // Process this frame
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (cap.width !== w || cap.height !== h) {
        cap.width = canvas.width = w;
        cap.height = canvas.height = h;
      }
      cctx.drawImage(video, 0, 0, w, h);

      let predictions;
      try {
        predictions = await model.detect(cap, LOW_CONF);
      } catch (err) {
        console.error(err);
        if (id === runId) {
          stopSource();
          showIdle();
          setMessage('Detection stopped because of an error. Reload the page and try again.', true);
        }
        return;
      }
      if (id !== runId) return;

      const detections = predictions
        .map((p) => ({ bbox: p.bbox, cls: p.class, score: p.score }));

      tracker.highThresh = settings.conf;
      const tracks = tracker.update(detections);
      drawTracks(tracks);

      // Stats (smoothed FPS) and sidebar list, refreshed a few times per second
      const now = performance.now();
      fps = fps ? fps * 0.9 + (1000 / (now - last)) * 0.1 : 1000 / (now - last);
      last = now;
      if (now - lastUi > 250) {
        lastUi = now;
        $('#stat-fps').textContent = fps.toFixed(1);
        $('#stat-count').textContent = String(tracks.length);
        $('#stat-ids').textContent = String(tracker.idsAssigned);
        renderTracked(tracks);
      }

      await nextFrame();
    }
  }

  /* ---------- Drawing ---------- */
  function drawTracks(tracks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height); // wipe the previous boxes
    const s = Math.max(1, canvas.width / 900); // keep strokes readable on large frames
    ctx.font = `600 ${Math.round(13 * s)}px "Instrument Sans", system-ui, sans-serif`;
    ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round';

    for (const t of tracks) {
      const color = colorFor(t.id);

      if (settings.trails && t.trail.length > 1) {
        ctx.save();
        ctx.globalAlpha = 0.75;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2 * s;
        ctx.beginPath();
        t.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
        ctx.restore();
      }

      const [x1, y1, x2, y2] = t.bbox;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5 * s;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const label = `${t.cls} #${t.id}`;
      const pad = 6 * s;
      const th = 22 * s;
      const tw = ctx.measureText(label).width + pad * 2;
      const ly = y1 - th >= 0 ? y1 - th : y1; // sit above the box, or inside if no room
      ctx.fillStyle = color;
      ctx.fillRect(x1, ly, tw, th);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x1 + pad, ly + th / 2 + 0.5 * s);
    }
  }

  function renderTracked(tracks) {
    trackedList.textContent = '';
    if (!tracks.length) {
      const li = document.createElement('li');
      li.className = 'tracked-empty';
      li.textContent = 'No objects in view yet.';
      trackedList.appendChild(li);
      return;
    }
    [...tracks]
      .sort((a, b) => a.id - b.id)
      .forEach((t) => {
        const li = document.createElement('li');

        const swatch = document.createElement('span');
        swatch.className = 'swatch';
        swatch.style.background = colorFor(t.id);

        const name = document.createElement('span');
        name.className = 'name';
        name.textContent = t.cls;

        const idEl = document.createElement('span');
        idEl.className = 'id';
        idEl.textContent = `#${t.id}`;

        const conf = document.createElement('span');
        conf.className = 'conf';
        conf.textContent = `${Math.round(t.score * 100)}%`;

        li.append(swatch, name, idEl, conf);
        trackedList.appendChild(li);
      });
  }

  /* ---------- Events ---------- */
  btnWebcam.addEventListener('click', () => startSource('webcam'));
  btnFile.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) startSource('file', file);
    fileInput.value = '';
  });
  btnStop.addEventListener('click', () => {
    stopSource();
    showIdle();
    setMessage('Stopped. Choose a video source to start again.');
  });

  $('#conf').addEventListener('input', (e) => {
    settings.conf = Number(e.target.value) / 100;
    $('#conf-out').textContent = `${e.target.value}%`;
  });
  $('#age').addEventListener('input', (e) => {
    tracker.maxAge = Number(e.target.value);
    $('#age-out').textContent = `${e.target.value} frames`;
  });
  $('#trails').addEventListener('change', (e) => {
    settings.trails = e.target.checked;
  });

  /* ---------- Start ---------- */
  showIdle();
  loadModel();
})();
