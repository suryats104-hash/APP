// On-device YOLOv8 inference (caries detection) for DHANT.
// All work happens inside the WebView — no network.
//
// Public API (attached to window.DhantYolo):
//   await DhantYolo.init({ modelUrl }) — loads the ONNX model
//   await DhantYolo.detect(htmlImageOrCanvas) — returns [{x,y,w,h, score, classId, label}, ...]

(function () {
  const NS = (window.DhantYolo = window.DhantYolo || {});

  const INPUT_SIZE = 640;
  const CONF_THRESH = 0.25;
  const IOU_THRESH  = 0.45;
  const CLASS_NAMES = ['caries', 'severe caries']; // 'd' / 'D'

  let session = null;
  let loading = null;

  // Configure ORT runtime — CPU WASM only, multi-threaded if available.
  NS.init = async function ({ modelUrl } = {}) {
    if (session) return session;
    if (loading) return loading;

    if (typeof ort === 'undefined') throw new Error('onnxruntime-web not loaded');

    // WASM file paths (we ship them next to ort.min.js)
    ort.env.wasm.wasmPaths = 'vendor/onnx/';
    ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 2, 4);
    ort.env.wasm.simd = true;

    loading = ort.InferenceSession.create(
      modelUrl || 'models/dhant-caries.onnx',
      {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      },
    ).then((s) => { session = s; return s; });

    return loading;
  };

  // Letterbox-resize the input bitmap into a 640×640 RGB float32 tensor
  // (CHW layout, values 0..1). Returns { tensor, scale, padX, padY }.
  function letterbox(source) {
    const W = source.naturalWidth  || source.videoWidth  || source.width;
    const H = source.naturalHeight || source.videoHeight || source.height;
    if (!W || !H) throw new Error('Image has no dimensions');

    const scale = Math.min(INPUT_SIZE / W, INPUT_SIZE / H);
    const nW = Math.round(W * scale);
    const nH = Math.round(H * scale);
    const padX = Math.floor((INPUT_SIZE - nW) / 2);
    const padY = Math.floor((INPUT_SIZE - nH) / 2);

    const canvas = document.createElement('canvas');
    canvas.width = INPUT_SIZE; canvas.height = INPUT_SIZE;
    const ctx = canvas.getContext('2d');
    // Background = gray 114 (Ultralytics convention)
    ctx.fillStyle = 'rgb(114,114,114)';
    ctx.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    ctx.drawImage(source, padX, padY, nW, nH);

    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

    // RGBA → CHW float32 [0,1]
    const total = INPUT_SIZE * INPUT_SIZE;
    const data = new Float32Array(3 * total);
    for (let i = 0, p = 0; i < imageData.length; i += 4, p++) {
      data[p]             = imageData[i]     / 255;  // R
      data[p + total]     = imageData[i + 1] / 255;  // G
      data[p + 2 * total] = imageData[i + 2] / 255;  // B
    }

    const tensor = new ort.Tensor('float32', data, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    return { tensor, scale, padX, padY, srcW: W, srcH: H };
  }

  // Non-max suppression on [{x1,y1,x2,y2,score,classId}] — returns the surviving subset.
  function nms(boxes, iouThresh) {
    const sorted = boxes.slice().sort((a, b) => b.score - a.score);
    const kept = [];
    while (sorted.length) {
      const top = sorted.shift();
      kept.push(top);
      for (let i = sorted.length - 1; i >= 0; i--) {
        if (iou(top, sorted[i]) > iouThresh) sorted.splice(i, 1);
      }
    }
    return kept;
  }
  function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const w  = Math.max(0, x2 - x1);
    const h  = Math.max(0, y2 - y1);
    const inter = w * h;
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-6);
  }

  // Decode YOLOv8 detect head output: [1, 4+nc, 8400] → per-detection {x,y,w,h, score, classId}.
  // Cells where max class score < threshold are discarded.
  function decode(output, scale, padX, padY, srcW, srcH) {
    const data = output.data;
    const dims = output.dims;            // [1, 4+nc, N]
    const N    = dims[2];
    const C    = dims[1];
    const nc   = C - 4;                  // number of classes

    const boxes = [];
    for (let i = 0; i < N; i++) {
      // Pick best class
      let bestC = 0, bestS = -Infinity;
      for (let c = 0; c < nc; c++) {
        const s = data[(4 + c) * N + i];
        if (s > bestS) { bestS = s; bestC = c; }
      }
      if (bestS < CONF_THRESH) continue;

      // Bbox in 640x640 letterboxed coordinates (cx, cy, w, h)
      const cx = data[0 * N + i];
      const cy = data[1 * N + i];
      const w  = data[2 * N + i];
      const h  = data[3 * N + i];

      // Convert to letterboxed corner coordinates, then unscale to source image
      let x1 = (cx - w / 2 - padX) / scale;
      let y1 = (cy - h / 2 - padY) / scale;
      let x2 = (cx + w / 2 - padX) / scale;
      let y2 = (cy + h / 2 - padY) / scale;

      // Clip to source bounds
      x1 = Math.max(0, Math.min(srcW, x1));
      y1 = Math.max(0, Math.min(srcH, y1));
      x2 = Math.max(0, Math.min(srcW, x2));
      y2 = Math.max(0, Math.min(srcH, y2));
      if (x2 <= x1 || y2 <= y1) continue;

      boxes.push({ x1, y1, x2, y2, score: bestS, classId: bestC });
    }
    return nms(boxes, IOU_THRESH);
  }

  NS.detect = async function (source) {
    if (!session) await NS.init();
    const { tensor, scale, padX, padY, srcW, srcH } = letterbox(source);
    const feeds = {};
    feeds[session.inputNames[0]] = tensor;
    const results = await session.run(feeds);
    const output  = results[session.outputNames[0]];
    const detections = decode(output, scale, padX, padY, srcW, srcH);

    return detections.map((d) => ({
      // Pixel coordinates relative to the source image:
      x: d.x1, y: d.y1, w: d.x2 - d.x1, h: d.y2 - d.y1,
      // Normalized 0..1 coordinates (same convention as MOCK_FINDINGS_BY_SLOT):
      box: { x: d.x1 / srcW, y: d.y1 / srcH, w: (d.x2 - d.x1) / srcW, h: (d.y2 - d.y1) / srcH },
      score: d.score,
      classId: d.classId,
      label: CLASS_NAMES[d.classId] || ('class ' + d.classId),
    }));
  };

  // Helper to load a stored dataURL into an HTMLImageElement
  NS.loadImage = function (dataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload  = () => resolve(img);
      img.onerror = reject;
      img.src = dataUrl;
    });
  };
})();
