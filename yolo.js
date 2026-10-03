/*
 * YOLOv8n detector for the browser (onnxruntime-web).
 *  - Input:  letterboxed 640x640 RGB image, values 0..1
 *  - Output: [1, 84, 8400] = 4 box values (cx, cy, w, h) + 80 class scores per candidate
 *  - Steps:  preprocess -> run model -> decode -> class-aware NMS
 * detect() returns [{ bbox: [x, y, w, h], class, score }] in source-image pixels.
 */
(function (root) {
  'use strict';

  const SIZE = 640;
  const NAMES = 'person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'.split(',');

  const buf = document.createElement('canvas');
  buf.width = buf.height = SIZE;
  const bctx = buf.getContext('2d', { willReadFrequently: true });
  const input = new Float32Array(3 * SIZE * SIZE);

  // Class-aware non-maximum suppression (boxes are [x1, y1, x2, y2]; Sort.iou comes from sort.js)
  function nms(cands, thr) {
    cands.sort((a, b) => b.score - a.score);
    const keep = [];
    for (const d of cands) {
      if (keep.every((k) => k.cls !== d.cls || Sort.iou(k.box, d.box) < thr)) keep.push(d);
      if (keep.length >= 60) break;
    }
    return keep;
  }

  async function load(url) {
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
    const gpu = !!navigator.gpu;
    const session = await ort.InferenceSession.create(url, {
      executionProviders: gpu ? ['webgpu', 'wasm'] : ['wasm'],
    });
    const inName = session.inputNames[0];

    async function detect(src, minScore = 0.25) {
      const w = src.width;
      const h = src.height;
      const s = Math.min(SIZE / w, SIZE / h);
      const nw = Math.round(w * s);
      const nh = Math.round(h * s);
      const dx = (SIZE - nw) / 2;
      const dy = (SIZE - nh) / 2;

      // Letterbox: keep aspect ratio, pad with gray
      bctx.fillStyle = '#727272';
      bctx.fillRect(0, 0, SIZE, SIZE);
      bctx.drawImage(src, dx, dy, nw, nh);
      const px = bctx.getImageData(0, 0, SIZE, SIZE).data;
      const n = SIZE * SIZE;
      for (let i = 0; i < n; i++) {
        input[i] = px[i * 4] / 255;
        input[n + i] = px[i * 4 + 1] / 255;
        input[2 * n + i] = px[i * 4 + 2] / 255;
      }

      const out = await session.run({ [inName]: new ort.Tensor('float32', input, [1, 3, SIZE, SIZE]) });
      const t = out[session.outputNames[0]];
      const data = t.data;
      const N = t.dims[2];
      const nc = t.dims[1] - 4;

      const cands = [];
      for (let i = 0; i < N; i++) {
        let best = 0;
        let c = 0;
        for (let k = 0; k < nc; k++) {
          const sc = data[(4 + k) * N + i];
          if (sc > best) { best = sc; c = k; }
        }
        if (best < minScore) continue;
        const bw = data[2 * N + i] / s;
        const bh = data[3 * N + i] / s;
        const x1 = (data[i] - dx) / s - bw / 2;
        const y1 = (data[N + i] - dy) / s - bh / 2;
        cands.push({ box: [x1, y1, x1 + bw, y1 + bh], cls: NAMES[c], score: best });
      }

      return nms(cands, 0.45).map((d) => ({
        bbox: [d.box[0], d.box[1], d.box[2] - d.box[0], d.box[3] - d.box[1]],
        class: d.cls,
        score: d.score,
      }));
    }

    return { backend: gpu ? 'WebGPU' : 'WASM', detect };
  }

  root.Yolo = { load };
})(typeof self !== 'undefined' ? self : this);
