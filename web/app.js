// DreamCamera v1.4 — deepdream camera. Web app.
// Dream engine ported from the Dreamlings generator (generator.html):
// same MODELS registry shape, PRE pixel math, gradient-ascent dreamStep,
// zoomCrop + path offsets, octave start-dream, engine fallback.
// Square format, 50 frames per run, IndexedDB gallery, MediaRecorder video.
// Camera only. No dreamlings, no game.
"use strict";
const $ = id => document.getElementById(id);
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Mobile GPUs: half-float textures halve the bandwidth per dream step.
// Set before any backend is created; ignored if the flag is unknown.
try { tf.env().set("WEBGL_FORCE_F16_TEXTURES", true); } catch (e) {}

// ===== MODELS =====
// v1/v2/v3/vgg16: Dreamlings exports, live on the DeepdreamlingsFusion site
// (verified 2026-11-09: model.json 200 for all four). mobilenetv2: TF.js
// savedmodel on Google Cloud Storage (verified: model.json 200, shard1
// HEAD 200, 4 shards, 125 weights, 299 nodes). mobilenetv1 0.25_224:
// TF.js layers model, GCS (verified 2026-11-10: loads, real gradients on
// conv_pw_11/12; fixed 224 input like mobilenetv2). Coming soon (NOT shipped as
// real — TFHub now redirects to Kaggle / GCS paths 404, weights unverified):
// VGG19, ResNet50, EfficientNetB0, InceptionResNetV2, DenseNet121.
const LIVE_BASE = "https://qu33pharcadegame.github.io/DeepdreamlingsFusion/models/";
const MODELS = {
  v1: { short: "v1", label: "Inception v1", kind: "graph", live: LIVE_BASE + "inceptionv1/model.json", local: "models/inceptionv1/model.json", pre: "v1",
    layers: ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"],
    pref: ["mixed4c", "mixed4d"], note: "The 2015 classic. Mid layers make eyes and creatures." },
  v2: { short: "v2", label: "Inception v2", kind: "graph", live: LIVE_BASE + "inceptionv2/model.json", local: "models/inceptionv2/model.json", pre: "v2",
    layers: ["InceptionV2/InceptionV2/Mixed_3b/concat", "InceptionV2/InceptionV2/Mixed_3c/concat", "InceptionV2/InceptionV2/Mixed_4a/concat", "InceptionV2/InceptionV2/Mixed_4b/concat", "InceptionV2/InceptionV2/Mixed_4c/concat", "InceptionV2/InceptionV2/Mixed_4d/concat", "InceptionV2/InceptionV2/Mixed_4e/concat", "InceptionV2/InceptionV2/Mixed_5a/concat", "InceptionV2/InceptionV2/Mixed_5b/concat", "InceptionV2/InceptionV2/Mixed_5c/concat"],
    pref: ["InceptionV2/InceptionV2/Mixed_4d/concat", "InceptionV2/InceptionV2/Mixed_4e/concat"], note: "Noisier than v1, wilder creatures." },
  v3: { short: "v3", label: "Inception v3", kind: "layers", live: LIVE_BASE + "inceptionv3/model.json", local: "models/inceptionv3/model.json", pre: "v3",
    layers: ["mixed2", "mixed3", "mixed4", "mixed5", "mixed6", "mixed7"],
    pref: ["mixed6", "mixed7"], note: "High layers make fur and creature faces." },
  vgg16: { short: "vgg16", label: "VGG16", kind: "layers", live: LIVE_BASE + "vgg16/model.json", local: "models/vgg16/model.json", pre: "vgg",
    layers: ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"],
    pref: ["block4_conv3"], note: "Painterly swirls; deep blocks make flowing organic shapes. Slowest model." },
  mobilenetv2: { short: "mobilenetv2", label: "MobileNetV2", kind: "graph", live: "https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json", local: "models/mobilenetv2/model.json", pre: "v2", trustLayers: true, minSize: 224, fixedSize: 224,
    layers: ["module_apply_default/MobilenetV2/expanded_conv_5/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_7/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_10/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_16/project/BatchNorm/FusedBatchNorm"],
    pref: ["module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm"], note: "Small and fast. Good first model on a slow phone." },
  mobilenetv1: { short: "mobilenetv1", label: "MobileNetV1", kind: "layers", live: "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json", local: "models/mobilenetv1/model.json", pre: "v2", minSize: 224, fixedSize: 224,
    layers: ["conv_pw_9_relu", "conv_pw_10_relu", "conv_pw_11_relu", "conv_pw_12_relu", "conv_pw_13_relu"],
    pref: ["conv_pw_11_relu", "conv_pw_12_relu"], note: "Smallest + fastest. Good first model on a slow phone." },
};
const COMING_SOON = ["VGG19", "ResNet50", "EfficientNetB0", "InceptionResNetV2", "DenseNet121"];
// Per-model pixel math (same as the Dreamlings generator).
const PRE = {
  v1: { norm: t => t.sub(117), denorm: t => t.add(117).div(255), lo: -117, hi: 138, noise: [-10, 10] },
  v3: { norm: t => t.div(127.5).sub(1), denorm: t => t.add(1).div(2), lo: -1, hi: 1, noise: [-1, 1] },
  v2: { norm: t => t.div(127.5).sub(1), denorm: t => t.add(1).div(2), lo: -1, hi: 1, noise: [-1, 1] },
  vgg: { norm: t => t.reverse(-1).sub(tf.tensor1d([103.939, 116.779, 123.68])),
         denorm: t => t.add(tf.tensor1d([103.939, 116.779, 123.68])).reverse(-1).div(255),
         lo: -103.939, hi: 151.061, noise: [-10, 10] },
};
const WASM_PATH = "tfjs/";
const QUALITY = { turbo: { w: 192, h: 192 }, fast: { w: 256, h: 256 }, normal: { w: 384, h: 384 }, high: { w: 512, h: 512 } };
const MAX_RUN = 50, TOTAL_MAX = 500;   // 50 frames per run; Continue adds 50 more, up to 500 total
const OCTAVE_SCALE = 1.35;
const LABEL = { webgpu: "WebGPU (GPU)", webgl: "WebGL (GPU)", wasm: "WebAssembly (CPU)", cpu: "JavaScript (CPU)" };
const ENGINE_SHORT = { webgpu: "WebGPU", webgl: "WebGL", wasm: "WASM", cpu: "CPU" };
const SETTINGS_KEY = "dreamcamera_settings_v1", COUNT_KEY = "dreamcamera_count";
const PATH_LABEL = { straight: "Straight", square: "Square", triangle: "Triangle", spiral: "Spiral", wave: "Wave" };

const PRESETS = [
  { m: "v1", name: "Wild dive", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.045, sharpen: 0.35, layers: ["mixed4a", "mixed4b"] } },
  { m: "v1", name: "Eye storm", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["mixed4c", "mixed4d"] } },
  { m: "v1", name: "Hound dreams", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 18, lr: 0.055, sharpen: 0.35, layers: ["mixed4d", "mixed4e"] } },
  { m: "v1", name: "The 2015 classic", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 22, lr: 0.05, sharpen: 0.35, layers: ["mixed4c", "mixed4d", "mixed4e"] } },
  { m: "v1", name: "Spiral melt", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["mixed4b", "mixed4c"] } },
  { m: "v2", name: "Woods noise", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 8, lr: 0.04, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_4d/concat"] } },
  { m: "v2", name: "Wave trip", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_4c/concat", "InceptionV2/InceptionV2/Mixed_4d/concat"] } },
  { m: "v3", name: "Fur storm", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 12, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["mixed6", "mixed7"] } },
  { m: "v3", name: "Silk road", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["mixed2", "mixed3"] } },
  { m: "vgg16", name: "Petal storm", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["block3_conv3"] } },
  { m: "vgg16", name: "Deep kelp", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["block4_conv3"] } },
  { m: "vgg16", name: "Ghost garden", s: { path: "straight", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.04, sharpen: 0.3, layers: ["block5_conv3"] } },
  { m: "mobilenetv2", name: "Quick sketch", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv2", name: "Pocket spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_10/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv1", name: "Tiny dream", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["conv_pw_11_relu", "conv_pw_12_relu"] } },
];
const DEFAULTS = { model: "v1", preset: 0, path: "straight", frames: 50, zoomFactor: 0.97, steps: 4, lr: 0.04, range: 0.08, segFrames: 10, sharpen: 0.35, layers: ["mixed4c", "mixed4d"], quality: "turbo", playFps: 12, playFpsOut: 12, boomerang: true, octaves: 3, jitter: 16, facing: "environment", modelSrc: "live" };

// ===== STATE =====
const vid = $("vid"), artCv = $("artCv");
const seedCv = document.createElement("canvas"), tmpCv = document.createElement("canvas");
let settings = loadSettings();
let MODEL = MODELS[settings.model] || MODELS.v1, P = PRE[MODEL.pre];
let model = null, midModel = null, modelLayers = [], grads = {}, engineOrder = [], engineIdx = -1, ready = false;
let busy = false, zooming = false, playing = false, playTimer = null;
let card = null, cur = null, msPerStep = null, stream = null;
let JITTER = settings.jitter;

// Camera zoom. Display scales the live video; capture crops the same
// centre region, so the dream sees exactly what the viewfinder shows.
let camZoom = 1, pinch = null, pinchEndAt = 0;
const CAM_ZOOM_MAX = 5;
function applyCamZoom() { vid.style.transform = camZoom > 1 ? "scale(" + camZoom.toFixed(3) + ")" : ""; }

// Widget support (Android app only). The app's Shoot & Dream widget
// launches the page with ?autodream=1; once the camera is ready the
// page starts a dream by itself, once per load.
let autoDreamDone = false;
const AUTO_DREAM = /[?&]autodream=1/.test(location.href);
// Hand the latest dream frame to the Android app (if any) so the
// home screen widgets can show it. No-op on the web.
function sendLatestDreamToWidget() {
  try {
    if (window.DreamCameraBridge && window.DreamCameraBridge.setLatestDream && artCv.width) {
      window.DreamCameraBridge.setLatestDream(artCv.toDataURL("image/jpeg", 0.85));
    }
  } catch (e) {}
}

function loadSettings() { let s = {}; try { s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") || {}; } catch (e) {} return Object.assign({}, DEFAULTS, s); }
function saveSettings() { try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch (e) {} updateEstimate(); }
function setStatus(t, err) { const s = $("status"); s.textContent = t; s.className = err ? "err" : ""; }
function setBar(p) { const b = $("bar"); if (p == null) { b.style.visibility = "hidden"; return; } b.style.visibility = "visible"; b.firstChild.style.width = Math.round(Math.max(0, Math.min(1, p)) * 100) + "%"; }
function dims() { return QUALITY[settings.quality] || QUALITY.normal; }
function clampFps(v) { v = parseFloat(v); if (!(v > 0)) v = 12; return Math.max(2, Math.min(30, v)); }
function cardFps() { return clampFps(settings.playFps); }
// The size to feed the model at one octave. A fixed-size model
// (MobileNetV2, MobileNetV1) takes exactly its fixed size at
// every octave — feeding it anything else throws the shape
// error (284x284 into a 224x224 model). Other models take the
// octave-scaled size, floored at minSize.
function dreamHW(h, w, s) {
  if (MODEL.fixedSize) return [MODEL.fixedSize, MODEL.fixedSize];
  return [Math.max(MODEL.minSize || 96, Math.round(h * s)),
          Math.max(MODEL.minSize || 96, Math.round(w * s))];
}
function cardFpsOut() { return clampFps(settings.playFpsOut); }
function cardBoom() { return settings.boomerang !== false; }
function seqLength() { const n = card.frames.length; return (cardBoom() && n > 2) ? 2 * n - 2 : n; }
function seqFrame(k) { const n = card.frames.length; if (!(cardBoom() && n > 2)) return k % n; const p = 2 * n - 2, m = k % p; return m < n ? m : p - m; }
function seqLeg(i) { const n = card.frames.length; if (!(cardBoom() && n > 2)) return 0; return (i % (2 * n - 2)) >= n ? 1 : 0; }

// ===== CAMERA =====
async function startCamera() {
  stopCamera();
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: settings.facing, width: { ideal: 1280 }, height: { ideal: 1280 } }, audio: false });
    vid.srcObject = stream;
    await vid.play();
    $("viewHint").textContent = "tap to shoot";
    setStatus("Camera ready. Tap the picture, or Shoot & Dream.");
    if (AUTO_DREAM && !autoDreamDone) { autoDreamDone = true; setTimeout(() => { if (!busy && !zooming) shootAndDream(); }, 600); }
  } catch (e) {
    setStatus("No camera: " + (e && e.message || e) + " — use Upload a photo instead.", true);
    $("viewHint").textContent = "no camera — upload a photo";
  }
}
function stopCamera() { if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; } vid.srcObject = null; }
function captureSquare() {
  const q = dims(), c = document.createElement("canvas"); c.width = q.w; c.height = q.h;
  const ctx = c.getContext("2d");
  if (vid.readyState >= 2 && vid.videoWidth) {
    const s = Math.min(vid.videoWidth, vid.videoHeight) / camZoom;   // zoomed: crop the centre, same as the scaled viewfinder
    ctx.drawImage(vid, (vid.videoWidth - s) / 2, (vid.videoHeight - s) / 2, s, s, 0, 0, q.w, q.h);
  } else { ctx.fillStyle = "#222"; ctx.fillRect(0, 0, q.w, q.h); }
  return c;
}

// ===== ENGINE SELECTION (same fallback as the Dreamlings generator) =====
let wasmPathSet = false;
async function tryBackend(name) {
  try {
    if (!(name in tf.engine().registryFactory)) return false;
    if (name === "wasm" && !wasmPathSet && tf.wasm && tf.wasm.setWasmPaths) { tf.wasm.setWasmPaths(WASM_PATH); wasmPathSet = true; }
    if (name === "webgpu" && !navigator.gpu) return false;
    const ok = await tf.setBackend(name); if (!ok) return false; await tf.ready();
    return tf.getBackend() === name;
  } catch (e) { return false; }
}
async function benchEngines() {
  engineOrder = []; engineIdx = -1;
  for (const n of ["webgpu", "webgl", "wasm", "cpu"]) if (await tryBackend(n)) engineOrder.push(n);
  return engineOrder.length > 0;
}
function patchBatchNorm(m) {
  for (const l of m.layers) {
    if (l.getClassName() !== "BatchNormalization" || l.scale || l.gamma) continue;
    const c = l.movingMean.shape[l.movingMean.shape.length - 1];
    const g = l.addWeight("gamma", [c], "float32", tf.initializers.ones(), null, false);
    g.read().assign(tf.ones([c]));   // addWeight leaves the value at 0; gamma 0 makes BN output only beta, so v3 dreamed nothing
    l.gamma = g; l.scale = true;
  }
}
function modelUrl() { return settings.modelSrc === "local" ? MODEL.local : MODEL.live; }
async function activate(name) {
  if (model) { model.dispose(); model = null; } midModel = null; grads = {}; if (cur) { cur.dispose(); cur = null; }
  if (!(await tryBackend(name))) throw new Error(name + " unavailable");
  $("engineBtn").textContent = "Engine: " + LABEL[name];
  setStatus("Loading " + MODEL.label + " on " + LABEL[name] + "…"); setBar(0);
  try {
    if (MODEL.kind === "layers") { model = await tf.loadLayersModel(modelUrl(), { onProgress: p => setBar(p) }); patchBatchNorm(model); }
    else { model = await tf.loadGraphModel(modelUrl(), { onProgress: p => setBar(p) }); }
  } catch (e) { throw new Error("loading the model failed: " + (e && e.message || e)); }
  if (MODEL.kind === "layers") {
    modelLayers = (model.outputNames && model.outputNames.length) ? model.outputNames.slice() : model.outputs.map(o => o.sourceLayer.name);
    if (MODEL.layers.some(n => modelLayers.indexOf(n) < 0)) {
      // Single-output export (MobileNetV1): dream on an intermediate model
      // over the named layers, not the final output (its gradient is 0).
      // midModel shares the base model's weights; never dispose it alone.
      midModel = tf.model({ inputs: model.inputs, outputs: MODEL.layers.map(n => model.getLayer(n).output) });
      modelLayers = MODEL.layers.slice();
    }
  } else if (MODEL.trustLayers) {
    modelLayers = MODEL.layers.slice();   // MobileNetV2 export lists only its final output; trust the listed nodes
  } else {
    const have = (model.outputs || []).map(o => o.name);
    modelLayers = MODEL.layers.filter(n => have.indexOf(n) >= 0 || have.indexOf(n + ":0") >= 0);
  }
  setStatus("Warming up the dream engine…"); setBar(null);
  try { const ws = MODEL.fixedSize || MODEL.minSize || 96; const t = tf.randomUniform([1, ws, ws, 3], P.lo, P.hi), s = dreamStep(t, 0.01); await s.data(); t.dispose(); s.dispose(); }
  catch (e) { throw new Error("the test dream failed: " + (e && e.message || e)); }
}
// Tap the Engine row: re-test every backend, then switch to
// the next one that works and reload the model on it.
async function cycleEngine() {
  if (busy || zooming) return;
  setStatus("Testing engines…");
  await benchEngines();
  if (!engineOrder.length) { setStatus("No dream engine on this device.", true); return; }
  const cur = tf.getBackend();
  const i = engineOrder.indexOf(cur);
  const next = engineOrder[(i + 1) % engineOrder.length];
  engineIdx = engineOrder.indexOf(next);
  ready = false;
  try {
    await activate(next);
    setStatus("Engine: " + LABEL[next] + ". Ready.");
  } catch (e) {
    setStatus("Could not start " + LABEL[next] + ": " + (e && e.message || e), true);
  }
  updateUI();
}
async function nextEngine() {
  while (++engineIdx < engineOrder.length) {
    try { await activate(engineOrder[engineIdx]); return true; }
    catch (e) { console.warn(e); }
  }
  return false;
}
async function ensureReady() {
  if (ready && model) return true;
  if (!engineOrder.length && !(await benchEngines())) { setStatus("No dream engine on this device (need WebGL, WebGPU or WASM).", true); return false; }
  engineIdx = -1;
  if (await nextEngine()) { ready = true; return true; }
  setStatus("Dreaming failed on every engine.", true); return false;
}

// ===== DEEPDREAM CORE (ported from the Dreamlings generator) =====
function activeLayers() {
  const want = (settings.layers && settings.layers.length) ? settings.layers : null;
  let use = want ? modelLayers.filter(n => want.indexOf(n) >= 0) : [];
  if (!use.length) { const pref = (MODEL.pref || []).filter(n => modelLayers.indexOf(n) >= 0); use = pref.length ? pref : modelLayers.slice(-2); }
  return use;
}
function getGrad() {
  const use = activeLayers(), key = use.join("+");
  if (grads[key]) return grads[key];
  if (MODEL.kind === "layers") {
    const src = midModel || model;   // intermediate model when the export has one output
    const idx = use.map(n => modelLayers.indexOf(n));
    return grads[key] = tf.grad(img => {
      let outs = src.apply(img, { training: false }); if (!Array.isArray(outs)) outs = [outs];
      let loss = null; for (const i of idx) { const l = tf.mean(outs[i]); loss = loss ? loss.add(l) : l; }
      return loss;
    });
  }
  return grads[key] = tf.grad(img => {
    let outs;
    try { outs = model.execute(img, use); }
    catch (e) { outs = model.execute(img); if (!Array.isArray(outs)) outs = [outs]; }  // fall back to declared outputs
    if (!Array.isArray(outs)) outs = [outs];
    let loss = null; for (const o of outs) { const l = tf.mean(o); loss = loss ? loss.add(l) : l; }
    return loss;
  });
}
function roll(t, shift, axis) { const n = t.shape[axis]; shift = ((shift % n) + n) % n; if (!shift) return t; const p = tf.split(t, [n - shift, shift], axis); return tf.concat([p[1], p[0]], axis); }
function dreamStep(img, lr) {
  return tf.tidy(() => {
    const sy = Math.floor((Math.random() * 2 - 1) * JITTER), sx = Math.floor((Math.random() * 2 - 1) * JITTER);
    let g = getGrad()(roll(roll(img, sy, 1), sx, 2));
    g = roll(roll(g, -sy, 1), -sx, 2);
    const std = tf.moments(g).variance.sqrt();
    return img.add(g.div(std.add(1e-8)).mul(lr)).clipByValue(P.lo, P.hi);
  });
}
function pathOffset(n, w, h, s) {
  const seg = Math.max(1, Math.round(s.segFrames)), sub = (n % seg) / seg;
  const rx = w * s.range / 2, ry = h * s.range / 2;
  if (s.path === "square") {
    const ph = Math.floor((n % (4 * seg)) / seg);
    if (ph === 0) return [-rx + 2 * rx * sub, -ry];
    if (ph === 1) return [rx, -ry + 2 * ry * sub];
    if (ph === 2) return [rx - 2 * rx * sub, ry];
    return [-rx, ry - 2 * ry * sub];
  }
  if (s.path === "triangle") {
    const ph = Math.floor((n % (3 * seg)) / seg);
    const v = [[0, -ry], [rx * Math.sqrt(3) / 2, ry / 2], [-rx * Math.sqrt(3) / 2, ry / 2]], a = v[ph], b = v[(ph + 1) % 3];
    return [a[0] + (b[0] - a[0]) * sub, a[1] + (b[1] - a[1]) * sub];
  }
  if (s.path === "spiral") {   // NEW: slow outward spiral
    const a = n * 0.35, r = Math.min(rx, ry) * Math.min(1, n / (seg * 6));
    return [Math.cos(a) * r, Math.sin(a) * r];
  }
  if (s.path === "wave") {     // NEW: side-to-side sine wave
    return [Math.sin(n / seg * Math.PI) * rx, Math.cos(n / (seg * 2) * Math.PI) * ry * 0.6];
  }
  return [0, 0];
}
function zoomCrop(img, zf, ox, oy, sharpen) {
  return tf.tidy(() => {
    const h = img.shape[1], w = img.shape[2], z = Math.max(0.5, Math.min(1, zf));
    const ch = Math.max(16, Math.round(h * z)), cw = Math.max(16, Math.round(w * z)), dy = Math.round(oy), dx = Math.round(ox);
    if (ch === h && cw === w) {
      if (!dx && !dy) return img.clone();
      const p = Math.min(Math.max(Math.abs(dx), Math.abs(dy)), Math.min(h, w) - 1);
      return tf.mirrorPad(img, [[0, 0], [p, p], [p, p], [0, 0]], "reflect").slice([0, p + Math.max(-p, Math.min(p, dy)), p + Math.max(-p, Math.min(p, dx)), 0], [1, h, w, 3]);
    }
    const y0 = Math.max(0, Math.min(h - ch, Math.round((h - ch) / 2 + oy))), x0 = Math.max(0, Math.min(w - cw, Math.round((w - cw) / 2 + ox)));
    let out = tf.image.resizeBilinear(img.slice([0, y0, x0, 0], [1, ch, cw, 3]), [h, w], false, true);
    if (sharpen > 0) {
      const k = [1, 2, 1, 2, 4, 2, 1, 2, 1].map(v => v / 16), ker = tf.tensor4d([].concat(...k.map(v => [v, v, v])), [3, 3, 3, 1]);
      const blur = tf.depthwiseConv2d(out, ker, 1, "same");
      out = out.add(out.sub(blur).mul(sharpen * 2)).clipByValue(P.lo, P.hi);
    }
    return out;
  });
}
function canvasToTensor(cv) { return tf.tidy(() => P.norm(tf.browser.fromPixels(cv).toFloat()).expandDims(0)); }
async function tensorToCanvas(img, cv) { const t = tf.tidy(() => P.denorm(img.squeeze([0])).clipByValue(0, 1)); await tf.browser.toPixels(t, cv); t.dispose(); }
function noteStepTime(ms) { msPerStep = msPerStep == null ? ms : msPerStep * 0.8 + ms * 0.2; updateEstimate(); }
function updateEstimate() {
  const el = $("estimate"); if (msPerStep == null) { el.textContent = ""; return; }
  const q = dims(), perFrame = msPerStep * settings.steps / 1000, run = perFrame * settings.frames;
  el.textContent = "≈" + (perFrame < 10 ? perFrame.toFixed(1) : Math.round(perFrame)) + "s/frame · run " + (run < 90 ? Math.round(run) + "s" : Math.round(run / 60) + " min") + " · " + q.w + "×" + q.h;
}

// ===== FRAMES (JPEG blobs, decoded on demand) =====
function toBlob(cv, type, q) { return new Promise(r => cv.toBlob(r, type, q)); }
async function decodeBlob(b) {
  if (window.createImageBitmap) { try { return await createImageBitmap(b); } catch (e) {} }
  const u = URL.createObjectURL(b); const im = await new Promise(res => { const i = new Image(); i.onload = () => res(i); i.onerror = () => res(null); i.src = u; }); URL.revokeObjectURL(u); return im;
}
async function pushFrame() { const b = await toBlob(artCv, "image/jpeg", 0.92); card.frames.push(b); }
function disposeCard() { if (cur) { cur.dispose(); cur = null; } card = null; }
function nextNumber() { let n = parseInt(localStorage.getItem(COUNT_KEY) || "0", 10) + 1; try { localStorage.setItem(COUNT_KEY, String(n)); } catch (e) {} return n; }

// ===== SHOOT + DREAM =====
async function shootAndDream() {
  if (busy || zooming) return;
  const shot = captureSquare();
  if (!await ensureReady()) return;
  disposeCard(); stopPlaying();
  const q = dims();
  card = { id: "dc" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6), no: nextNumber(), name: "Dream " + new Date().toLocaleDateString(), model: MODEL.short, preset: $("presetSel").selectedOptions[0] ? $("presetSel").selectedOptions[0].textContent : "", w: q.w, h: q.h, created: Date.now(), frames: [], pathFrame: 0, settings: Object.assign({}, settings) };
  seedCv.width = q.w; seedCv.height = q.h; seedCv.getContext("2d").drawImage(shot, 0, 0);
  artCv.width = q.w; artCv.height = q.h; artCv.getContext("2d").drawImage(shot, 0, 0);
  $("viewBox").classList.add("dreaming"); $("viewHint").textContent = "";
  await startDream();
  if (card && card.frames.length) { await zoomRun(); }
  updateUI();
}
async function startDream() {
  if (!card || busy || !ready) return;
  busy = true; updateUI();
  const octaves = Math.max(1, Math.round(settings.octaves)), total = octaves;
  for (;;) {
    try {
      let img = tf.tidy(() => canvasToTensor(seedCv));
      for (let o = octaves - 1; o >= 0; o--) {
        const s = Math.pow(OCTAVE_SCALE, -o), hw = dreamHW(card.h, card.w, s), h = hw[0], w = hw[1];
        const r = tf.image.resizeBilinear(img, [h, w]); img.dispose(); img = r;
        // One gentle step per octave: sharp start, like the Dreamlings generator (startSteps=0).
        const t0 = performance.now(); const n = dreamStep(img, Math.max(0.01, settings.lr)); img.dispose(); img = n;
        if (o === 0) noteStepTime(performance.now() - t0);
        setStatus("Dreaming… detail layer " + (octaves - o) + " of " + octaves); setBar((octaves - o) / total);
        if (o === 0) await tensorToCanvas(img, artCv); else await tf.nextFrame();
      }
      cur = img; break;
    } catch (e) {
      console.error(e);
      if (!(await nextEngine())) { setStatus("Dreaming failed on every engine: " + (e && e.message || e), true); setBar(null); busy = false; updateUI(); return; }
      setStatus("Switched to " + LABEL[engineOrder[engineIdx]] + " — trying again…");
    }
  }
  await tensorToCanvas(cur, artCv); await pushFrame();
  busy = false; setBar(null);
  setStatus("Dreaming. Falling in — " + settings.frames + " frames.");
  updateUI();
}
async function ensureCur() {
  if (cur) return;
  const src = await decodeBlob(card.frames[card.frames.length - 1]);
  tmpCv.width = card.w; tmpCv.height = card.h; const t = tmpCv.getContext("2d"); t.clearRect(0, 0, card.w, card.h); t.drawImage(src, 0, 0, card.w, card.h);
  if (src && src.close) src.close();
  cur = canvasToTensor(tmpCv);
}
async function zoomRun() {
  if (!card || busy || zooming || !card.frames.length) return;
  zooming = true; let failed = false, made = 0; updateUI();
  const want = Math.max(1, Math.min(MAX_RUN, Math.round(settings.frames)));
  while (zooming && made < want && card.frames.length < TOTAL_MAX) {
    try {
      await ensureCur();
      card.pathFrame = (card.pathFrame || 0) + 1;
      const off = pathOffset(card.pathFrame, card.w, card.h, settings);
      const z = zoomCrop(cur, settings.zoomFactor, off[0], off[1], settings.sharpen || 0); cur.dispose(); cur = z;
      if (MODEL.fixedSize && (cur.shape[1] !== MODEL.fixedSize || cur.shape[2] !== MODEL.fixedSize)) {
        const r = tf.image.resizeBilinear(cur, [MODEL.fixedSize, MODEL.fixedSize]); cur.dispose(); cur = r;
      }
      const t0 = performance.now();
      for (let i = 0; i < settings.steps; i++) { const n = dreamStep(cur, settings.lr); cur.dispose(); cur = n; }
      await tensorToCanvas(cur, artCv); noteStepTime((performance.now() - t0) / Math.max(1, settings.steps));
      await pushFrame(); made++;
      setStatus("Zooming… frame " + made + " of " + want + " (" + card.frames.length + " total)"); setBar(made / want);
      await tf.nextFrame();
    } catch (e) {
      console.error(e); if (cur) { cur.dispose(); cur = null; }
      if (!(await nextEngine())) { setStatus("Zoom failed: " + (e && e.message || e), true); failed = true; break; }
      setStatus("Switched to " + LABEL[engineOrder[engineIdx]] + " — carrying on…");
    }
  }
  zooming = false; setBar(null);
  if (!failed) {
    if (card.frames.length >= TOTAL_MAX) setStatus("Full (" + TOTAL_MAX + " frames). Save it as a video!");
    else if (made >= want) setStatus("Done: " + card.frames.length + " frames. Continue for 50 more, or save.");
    else setStatus(card.frames.length + " frames. Stopped.");
  }
  sendLatestDreamToWidget();
  startPlaying();
  updateUI();
}

// ===== PLAYBACK =====
function startPlaying() {
  stopPlaying(); if (!card || card.frames.length < 2) return;
  playing = true; let i = 0, last = performance.now();
  const tick = async () => {
    if (!playing || !card) return;
    const now = performance.now(), leg = seqLeg(i), fps = leg ? cardFpsOut() : cardFps();
    if (now - last >= 1000 / fps) {
      last = now; const fi = seqFrame(i); i++;
      const im = await decodeBlob(card.frames[fi]);
      if (im && card) { const c = artCv.getContext("2d"); c.clearRect(0, 0, card.w, card.h); c.drawImage(im, 0, 0, card.w, card.h); if (im.close) im.close(); }
    }
    playTimer = requestAnimationFrame(tick);
  };
  playTimer = requestAnimationFrame(tick);
}
function stopPlaying() { playing = false; if (playTimer) { cancelAnimationFrame(playTimer); playTimer = null; } }

// ===== EXPORT: VIDEO + STILL =====
function videoMime() { if (!window.MediaRecorder) return null; const m = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]; for (const t of m) if (MediaRecorder.isTypeSupported(t)) return t; return ""; }
async function exportVideo() {
  if (!card || busy || zooming || !card.frames.length) return;
  busy = true; updateUI();
  try {
    const mime = videoMime(), L = seqLength(), fpsIn = cardFps(), fpsOut = cardFpsOut();
    const stream2 = artCv.captureStream(30);
    const rec = new MediaRecorder(stream2, mime ? { mimeType: mime, videoBitsPerSecond: 5000000 } : { videoBitsPerSecond: 5000000 });
    const chunks = []; rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    const done = new Promise(r => rec.onstop = r);
    rec.start(200);
    for (let i = 0; i < L; i++) {
      const fi = seqFrame(i), leg = seqLeg(i), fps = leg ? fpsOut : fpsIn;
      const im = await decodeBlob(card.frames[fi]);
      if (im) { const c = artCv.getContext("2d"); c.clearRect(0, 0, card.w, card.h); c.drawImage(im, 0, 0, card.w, card.h); if (im.close) im.close(); }
      setStatus("Recording video… " + (i + 1) + " / " + L); setBar((i + 1) / L);
      await sleep(1000 / fps);
    }
    await sleep(300); rec.stop(); await done;
    const blob = new Blob(chunks, { type: mime || "video/webm" });
    const ext = (mime || "").includes("mp4") ? "mp4" : "webm";
    downloadBlob(blob, "dreamcamera-" + String(card.no).padStart(4, "0") + "." + ext);
    setStatus("Video saved (" + (blob.size / 1048576).toFixed(1) + " MB). Also keep it in the gallery to watch later.");
  } catch (e) { console.error(e); setStatus("Video export failed: " + (e && e.message || e), true); }
  setBar(null); busy = false; updateUI();
}
async function exportStill() {
  if (!card || !card.frames.length) return;
  const im = await decodeBlob(card.frames[card.frames.length - 1]);
  const c = document.createElement("canvas"); c.width = card.w; c.height = card.h;
  c.getContext("2d").drawImage(im, 0, 0, card.w, card.h); if (im && im.close) im.close();
  const b = await toBlob(c, "image/png", 1);
  downloadBlob(b, "dreamcamera-" + String(card.no).padStart(4, "0") + ".png");
  setStatus("Still saved.");
}
function downloadBlob(blob, name) {
  const u = URL.createObjectURL(blob), a = document.createElement("a");
  a.href = u; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(u), 5000);
}

// ===== GALLERY (IndexedDB) =====
let dbp = null;
function idb() {
  if (dbp) return dbp;
  dbp = new Promise((res, rej) => {
    const rq = indexedDB.open("dreamcamera", 1);
    rq.onupgradeneeded = () => { rq.result.createObjectStore("dreams", { keyPath: "id" }); };
    rq.onsuccess = () => res(rq.result); rq.onerror = () => rej(rq.error);
  });
  return dbp;
}
async function idbDo(mode, fn) { const db = await idb(); return new Promise((res, rej) => { const tx = db.transaction("dreams", mode), st = tx.objectStore("dreams"); const out = fn(st); tx.oncomplete = () => res(out && out.result !== undefined ? out.result : out); tx.onerror = () => rej(tx.error); }); }
async function keepInGallery() {
  if (!card || !card.frames.length) return;
  const thumb = card.frames[0];
  const rec = { id: card.id, created: card.created, no: card.no, name: card.name, model: card.model, preset: card.preset, w: card.w, h: card.h, frames: card.frames.slice(), thumb: thumb, settings: card.settings };
  try { await idbDo("readwrite", st => st.put(rec)); setStatus("Kept in gallery."); renderGallery(); sendLatestDreamToWidget(); }
  catch (e) { setStatus("Couldn't keep it: " + (e && e.message || e), true); }
}
async function renderGallery() {
  const box = $("tiles"); box.innerHTML = "";
  let recs = []; try { recs = await idbDo("readonly", st => st.getAll()) || []; } catch (e) { box.innerHTML = '<div class="empty">Gallery unavailable on this browser.</div>'; return; }
  recs.sort((a, b) => b.created - a.created);
  if (!recs.length) { box.innerHTML = '<div class="empty">Nothing yet. Shoot a photo and dream it.</div>'; return; }
  for (const r of recs) {
    const tile = document.createElement("div"); tile.className = "tile";
    const im = document.createElement("img"); im.alt = r.name;
    const meta = document.createElement("div"); meta.className = "meta";
    meta.innerHTML = "<button class='del' aria-label='Delete'>✕</button>" + r.name + "<small>" + (r.model || "") + " · " + r.frames.length + " frames · " + new Date(r.created).toLocaleDateString() + "</small>";
    tile.appendChild(im); tile.appendChild(meta); box.appendChild(tile);
    if (r.thumb) { const u = URL.createObjectURL(r.thumb); im.src = u; }
    tile.onclick = async e => {
      if (e.target.classList.contains("del")) {
        if (!confirm("Delete " + r.name + "?")) return;
        await idbDo("readwrite", st => st.delete(r.id)); renderGallery(); return;
      }
      if (busy || zooming) return;
      disposeCard(); stopPlaying();
      card = { id: r.id, no: r.no, name: r.name, model: r.model, preset: r.preset, w: r.w, h: r.h, created: r.created, frames: r.frames.slice(), pathFrame: r.frames.length - 1, settings: r.settings || settings };
      artCv.width = r.w; artCv.height = r.h;
      $("viewBox").classList.add("dreaming"); $("viewHint").textContent = "";
      setStatus(r.name + " · " + r.frames.length + " frames. Continue to dream further.");
      startPlaying(); updateUI();
    };
  }
}

// ===== UI =====
// Reset: stop everything, drop the dream, back to the live camera
// with the default settings.
async function resetCamera() {
  zooming = false; busy = false; stopPlaying();
  disposeCard();
  $("viewBox").classList.remove("dreaming");
  $("viewHint").textContent = "tap to shoot";
  setBar(null);
  camZoom = 1; applyCamZoom();
  settings = Object.assign({}, DEFAULTS);
  JITTER = settings.jitter; msPerStep = null;
  MODEL = MODELS[settings.model]; P = PRE[MODEL.pre];
  ready = false; model = null; midModel = null; grads = {};
  saveSettings();
  buildModelSelect(); buildLayers(); buildPresets();
  if (PRESETS[settings.preset] && PRESETS[settings.preset].m === MODEL.short) applyPreset(settings.preset);
  syncControls();
  updateUI();
  await startCamera();
}
function updateUI() {
  $("shootBtn").disabled = busy || zooming;
  $("stopBtn").disabled = !(busy || zooming);
  $("continueBtn").disabled = !card || !card.frames.length || busy || zooming || card.frames.length >= TOTAL_MAX;
  $("videoBtn").disabled = !card || !card.frames.length || busy || zooming;
  $("stillBtn").disabled = !card || !card.frames.length || busy || zooming;
  $("keepBtn").disabled = !card || !card.frames.length || busy || zooming;
  $("modelNote").textContent = MODEL.note || "";
}
function bindRange(id, key, fmt, after) {
  const el = $(id);
  el.value = settings[key];
  const show = () => { $(id + "Val") && ($(id + "Val").textContent = fmt(settings[key])); };
  show();
  el.addEventListener("input", () => { settings[key] = parseFloat(el.value); show(); saveSettings(); if (after) after(); });
}
function onModelChange() {
  const sel = $("modelSel");
  settings.model = sel.value; MODEL = MODELS[settings.model]; P = PRE[MODEL.pre];
  settings.layers = MODEL.pref.slice(); ready = false; model = null; midModel = null; grads = {};
  buildLayers(); buildPresets(); saveSettings(); updateUI();
  setStatus(MODEL.label + " selected. It loads when you shoot.");
}
function buildModelSelect() {
  const sel = $("modelSel"); sel.innerHTML = "";
  for (const k of Object.keys(MODELS)) { const o = document.createElement("option"); o.value = k; o.textContent = MODELS[k].label; sel.appendChild(o); }
  for (const n of COMING_SOON) { const o = document.createElement("option"); o.disabled = true; o.textContent = n + " — coming soon"; sel.appendChild(o); }
  sel.value = settings.model;
  sel.removeEventListener("change", onModelChange);
  sel.addEventListener("change", onModelChange);
}
function buildLayers() {
  const box = $("layerBox"); box.innerHTML = "";
  const want = settings.layers || [];
  for (const n of MODEL.layers) {
    const lab = document.createElement("label");
    const short = n.split("/").pop().replace(":0", "");
    lab.textContent = short; lab.title = n;
    if (want.indexOf(n) >= 0) lab.classList.add("on");
    lab.onclick = () => {
      const i = settings.layers.indexOf(n);
      if (i >= 0) settings.layers.splice(i, 1); else settings.layers.push(n);
      lab.classList.toggle("on"); saveSettings();
    };
    box.appendChild(lab);
  }
}
function buildPresets() {
  const sel = $("presetSel"); sel.innerHTML = "";
  const list = PRESETS.map((p, i) => ({ p, i })).filter(o => o.p.m === MODEL.short);
  list.forEach((o, j) => { const op = document.createElement("option"); op.value = o.i; op.textContent = o.p.name; sel.appendChild(op); });
  const cur = list.findIndex(o => o.i === settings.preset);
  sel.value = cur >= 0 ? list[cur].i : (list[0] ? list[0].i : "");
  sel.onchange = () => { applyPreset(parseInt(sel.value, 10)); };
}
function applyPreset(i) {
  const p = PRESETS[i]; if (!p) return;
  settings.preset = i;
  Object.assign(settings, p.s);
  if (p.s.layers) settings.layers = p.s.layers.slice();
  syncControls(); buildLayers(); saveSettings();
}
function syncControls() {
  const map = { steps: "steps", lr: "lr", zoomFactor: "zoomFactor", range: "range", segFrames: "segFrames", frames: "frames", playFps: "playFps", playFpsOut: "playFpsOut", sharpen: "sharpen", octaves: "octaves", jitter: "jitter" };
  for (const id of Object.keys(map)) { const el = $(id); el.value = settings[map[id]]; const v = $(id + "Val"); if (v) v.textContent = fmtVal(id, settings[map[id]]); }
  $("path").value = settings.path; $("quality").value = settings.quality;
  $("boomerang").value = settings.boomerang ? "1" : "0";
  $("facing").value = settings.facing; $("modelSrc").value = settings.modelSrc;
}
function fmtVal(id, v) {
  if (id === "lr") return (+v).toFixed(3);
  if (id === "zoomFactor") return (+v).toFixed(3);
  if (id === "range") return (+v).toFixed(2);
  if (id === "sharpen") return (+v).toFixed(2);
  return String(v);
}
function initControls() {
  bindRange("steps", "steps", v => String(v));
  bindRange("lr", "lr", v => (+v).toFixed(3));
  bindRange("zoomFactor", "zoomFactor", v => (+v).toFixed(3));
  bindRange("range", "range", v => (+v).toFixed(2));
  bindRange("segFrames", "segFrames", v => String(v));
  bindRange("frames", "frames", v => String(v));
  bindRange("playFps", "playFps", v => String(v));
  bindRange("playFpsOut", "playFpsOut", v => String(v));
  bindRange("sharpen", "sharpen", v => (+v).toFixed(2));
  bindRange("octaves", "octaves", v => String(v));
  bindRange("jitter", "jitter", v => String(v), () => { JITTER = settings.jitter; });
  $("path").value = settings.path;
  $("path").addEventListener("change", () => { settings.path = $("path").value; saveSettings(); });
  $("quality").value = settings.quality;
  $("quality").addEventListener("change", () => { settings.quality = $("quality").value; saveSettings(); });
  $("boomerang").value = settings.boomerang ? "1" : "0";
  $("boomerang").addEventListener("change", () => { settings.boomerang = $("boomerang").value === "1"; saveSettings(); });
  $("facing").value = settings.facing;
  $("facing").addEventListener("change", () => { settings.facing = $("facing").value; saveSettings(); startCamera(); });
  $("modelSrc").value = settings.modelSrc;
  $("engineBtn").addEventListener("click", () => { cycleEngine(); });
  $("modelSrc").addEventListener("change", () => { settings.modelSrc = $("modelSrc").value; ready = false; model = null; midModel = null; saveSettings(); setStatus("Model source: " + (settings.modelSrc === "local" ? "local models/ folder" : "live site") + ". Loads on next shoot."); });
  $("shootBtn").onclick = shootAndDream;
  $("viewBox").onclick = () => { if (Date.now() - pinchEndAt < 400) return; if (!busy && !zooming && card && card.frames.length && $("viewBox").classList.contains("dreaming")) return; shootAndDream(); };
  const vb = $("viewBox");
  vb.addEventListener("touchstart", e => {
    if (e.touches.length === 2 && !vb.classList.contains("dreaming")) {
      const dx = e.touches[0].clientX - e.touches[1].clientX, dy = e.touches[0].clientY - e.touches[1].clientY;
      pinch = { d: Math.hypot(dx, dy), z: camZoom };
    }
  }, { passive: true });
  vb.addEventListener("touchmove", e => {
    if (e.touches.length === 2 && pinch && !vb.classList.contains("dreaming")) {
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX, dy = e.touches[0].clientY - e.touches[1].clientY;
      camZoom = Math.max(1, Math.min(CAM_ZOOM_MAX, pinch.z * Math.hypot(dx, dy) / pinch.d));
      applyCamZoom();
    }
  }, { passive: false });
  const endPinch = e => { if (pinch && e.touches.length < 2) { pinch = null; pinchEndAt = Date.now(); } };
  vb.addEventListener("touchend", endPinch);
  vb.addEventListener("touchcancel", endPinch);
  $("stopBtn").onclick = () => { zooming = false; busy = false; setStatus("Stopped."); updateUI(); };
  $("resetBtn").onclick = resetCamera;
  $("continueBtn").onclick = async () => { if (card && card.frames.length) await zoomRun(); };
  $("videoBtn").onclick = exportVideo;
  $("stillBtn").onclick = exportStill;
  $("keepBtn").onclick = keepInGallery;
  $("uploadBtn").onclick = () => $("fileIn").click();
  $("fileIn").addEventListener("change", async () => {
    const f = $("fileIn").files[0]; if (!f) return;
    const url = URL.createObjectURL(f), im = new Image();
    im.onload = async () => {
      if (!await ensureReady()) { URL.revokeObjectURL(url); return; }
      disposeCard(); stopPlaying();
      const q = dims();
      card = { id: "dc" + Date.now().toString(36), no: nextNumber(), name: f.name.replace(/\.[^.]+$/, "") || "Upload", model: MODEL.short, preset: "", w: q.w, h: q.h, created: Date.now(), frames: [], pathFrame: 0, settings: Object.assign({}, settings) };
      const c = document.createElement("canvas"); c.width = q.w; c.height = q.h;
      const s = Math.min(im.naturalWidth, im.naturalHeight);
      c.getContext("2d").drawImage(im, (im.naturalWidth - s) / 2, (im.naturalHeight - s) / 2, s, s, 0, 0, q.w, q.h);
      seedCv.width = q.w; seedCv.height = q.h; seedCv.getContext("2d").drawImage(c, 0, 0);
      artCv.width = q.w; artCv.height = q.h; artCv.getContext("2d").drawImage(c, 0, 0);
      $("viewBox").classList.add("dreaming"); $("viewHint").textContent = "";
      URL.revokeObjectURL(url);
      await startDream(); if (card && card.frames.length) await zoomRun(); updateUI();
    };
    im.src = url;
  });
}

// ===== BOOT =====
buildModelSelect();
buildLayers();
buildPresets();
if (settings.preset != null && PRESETS[settings.preset] && PRESETS[settings.preset].m === MODEL.short) applyPreset(settings.preset);
syncControls();
initControls();
updateUI();
renderGallery();
startCamera();
