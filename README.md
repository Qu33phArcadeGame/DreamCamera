# DreamCamera v1.4

Deepdream camera. Nothing else. No dreamlings, no game, no coins.

- **Web app** (`web/`): two stacked squares. Top = plain live viewfinder. Bottom = settings. Shoot → it dreams the photo. 50 frames per run, then Continue or Save.
- **DSi** (`dsi/`): top screen = plain camera viewfinder. Bottom screen = settings. Same 50-frame runs, frames saved to SD.
- **Android** (`android/`): WebView wrapper of the web app. Camera permission, offline models bundled.

## Web app

Open `web/index.html` on any HTTPS host (camera needs HTTPS or localhost), or serve it: `python3 -m http.server` in `web/`.

- Top square: live camera, square crop, no markings. Tap it (or Shoot & Dream) to capture.
- Bottom square: model, preset, layers, steps, learning rate, zoom, path, frames (1–50), size, fps in/out, zoomerang, sharpen, octaves, jitter, camera front/back, model source.
- At 50 frames: **Continue +50** or **Save video** (WebM/MP4) / **Save still** / **Keep in gallery**.
- Gallery (below): IndexedDB. Tap to replay, ✕ to delete.
- No camera? **Upload a photo instead**.

## Models

| Model | Status | Source |
|---|---|---|
| Inception v1 | Real | Live: DeepdreamlingsFusion site |
| Inception v2 | Real | Live: DeepdreamlingsFusion site |
| Inception v3 | Real | Live: DeepdreamlingsFusion site |
| VGG16 | Real | Live: DeepdreamlingsFusion site |
| MobileNetV2 | Real | Google Cloud Storage (tfjs-models) |
| MobileNetV1 (0.25) | Real | Google Cloud Storage (tfjs-models) |
| VGG19, ResNet50, EfficientNetB0, InceptionResNetV2, DenseNet121 | Coming soon | Not verified — TFHub now redirects to Kaggle, GCS paths 404 |

Models load on demand from the live site by default. For offline use: `web/tools/download_models.sh` fetches all six into `web/models/`, then set Model source → Local.

Verified 2026-11-09: Inception v1 and MobileNetV2 loaded in TF.js 4.22.0 and produced real gradients (v1 mixed4c at 96×96, MobileNetV2 expanded_conv_13 at 224×224 — it needs exactly 224×224 input, the app clamps octaves for it). v2/VGG16: model.json + weight shards reachable, same exports already shipping in the Dreamlings Fusion app.

Verified 2026-11-10 (v1.2, pushed): MobileNetV1 0.25_224 loads in TF.js 4.22.0, 1.9 MB, real gradients on conv_pw_11/12 at 224×224 (fixed 224 input, like MobileNetV2). Same check found Inception v3 was silently dreaming nothing: its 70 BatchNorm layers have no gamma, and the patch that adds one left it at 0, so every BatchNorm output was just its beta — v3 is fixed in this build by setting the patched gamma to 1 (verified: all six mixed outputs now vary with the input, mixed6 gradient real).

v1.3: neon redesign, Reset button (stops the dream, back to the live camera with default settings), Turbo 192 size (default), half-float WebGL textures for speed.

v1.4: home screen widgets (Android app only) — DreamCamera 2x2 shows your latest dream, tap to open; DreamCamera Shoot 3x3 shows it big with a Shoot & Dream button that opens the app and starts a dream right away. The app hands each finished dream to the widgets through a JS bridge.

## DSi

Push → GitHub Actions builds `dreamcamera.nds` (BlocksDS, DSi mode, `dsi_arm9.specs`).

- Top screen: DSi camera preview, 256×192, plain. Inner/outer switch with X.
- Bottom screen: Dream strength, zoom, path, frames, octaves. A shoots, B back, Y gallery.
- The dream on DSi is a **tiny dream**: fixed 3×3 edge-enhance kernels + zoom + octaves. Not the neural models.

Why: a real Inception/VGG model is 12–29 MB of weights and needs far more than the DSi's 16 MB RAM at 133 MHz. The full neural dream is the web app.

Camera: libnds DSi camera API — `cameraInit()`, `cameraSelect(CAMERA_INNER/OUTER)`, `cameraStartTransfer(buf, MCUREG_APT_SEQ_CMD_PREVIEW, ndma)` (preview 256×192 RGB). If the camera doesn't answer (DS mode, emulator), the viewfinder shows a labelled test pattern and the dream still works on it. Frames save as raw RGB565 to `/dreamcamera/dcNNNN/fNNN.bin` + `info.txt`. No video encode on DSi — play frames back in Gallery, or dream the same shot in the web app for video.

## Android APK

`android/build_apk.sh` builds `com.dreamcamera.app` from `web/` + downloaded models. Needs Camera permission. See `android/README.md`.

## Test first

1. Phone: open the web app, back camera, Inception v1, Shoot & Dream. Should reach 50 frames, then offer Continue / Save video.
2. Same shot on MobileNetV2 — fastest model.
3. DSi: run `dreamcamera.nds`, check the top screen shows the real camera, shoot, check `/dreamcamera/` on the SD card.
