# DreamCamera v1.23

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

v1.5: three new models, all gradient-verified real — VGG19 (80 MB), ResNet50 (103 MB, fixed 224), EfficientNetB0 (21 MB) — all bundled in the APK (~300 MB total). Upload a photo now works in the APK (the WebView opens a photo picker). The footer text is gone. The camera video can no longer pop out into a floating window. InceptionResNetV2 + DenseNet121 still have no hosted TF.js copy — Keras conversion in progress.

v1.6: the last two models, converted from Keras weights and gradient-verified — DenseNet121 (32 MB, new torch preprocessing) and InceptionResNetV2 (214 MB, graph model, fixed 299 — its layers version cannot load in TF.js). All 11 models real and bundled in the APK (~535 MB total).

v1.7: long runs stay clean — the zoom now caps at 3x total magnification and the dream strength eases off as the run goes on, so it no longer turns into a grid of boxes past 50 frames. Higher resolutions (Ultra 768, Max 1024), photo ratios (1:1, 9:16, 16:9), progress bar moved below the Shoot button, subtle holographic buttons, 46 presets (4+ per model), and the model notes cleaned up.

v1.23: 8K crash fixed + zentangledream goes sharp at 4K/8K + way more trippy knobs. 8K was crashing because the tiled dream's normalize step held three 8K ImageData objects (~400MB) at once on top of the canvases — the weights already sum to exactly 1, so the normalize and the weight canvas are gone; the peak is now just the two canvases, and a still frees its accumulation canvas right away so the PNG encode does not share memory with it. Your phone does support 8K — the hardware is fine, the app was just holding too much at once. Fixed-size models (zentangledream, MobileNet, ResNet, DenseNet, InceptionResNetV2) now tile at their native size (224 / 299) instead of 512, so nothing is downscaled and back — zentangledream stays sharp at 4K/8K. That is slower (an 8K zentangledream still is ~1300 tiles, a few minutes), and 4K is the sweet spot. New trippy knobs, Camera + Video: Color spin (hue degrees per frame — the colours cycle as it dreams), Color pop (saturation up to 2.5x), Pulse (the dream strength breathes in and out, one full breath every ~25 frames). Video also gets a Zoom speed slider and Weave now goes to 0.9 for longer smears.
v1.22: true 4K/8K deepdreams — real resolution, not an upscale. The 4K export used to be a small dream (192–256px by default) stretched 15–20x, which is why it looked pixelated. A whole 4K frame through the model would run the phone out of memory, so the high-res dream is now done in overlapping tiles: each 512px tile is dreamed on its own (bounded memory), and the tiles are blended with a feathered weight that sums to exactly 1 everywhere (verified numerically on 19 sizes — no seams, no dimming). The result is real gradient ascent at full resolution. Video tab: a Dream size selector (256–1440, default 512, separate from the Camera quality) and a Detail pass (4K) that dreams each frame again at 3840px tile by tile — true 4K video, recorded at 30 Mbps. It is slow (~20s a frame, so a 15s clip can take over an hour — short clips are quick). The presets now set the dream size too (Acid Weave / Hyper Weave / Deep Vortex / Meltdown dream at 1024). Stills: Save still at 4K or 8K now dreams the still at that size tile by tile (true 4K/8K, ~20s / ~2 min), instead of stretching it. Camera quality adds 2K · 1440, and Export size adds 8K (stills only — video stays at 4K max, since a true 8K video would need the tiled dream on every frame). Use Inception v1 for the sharpest high-res detail: it dreams each tile at full size, while fixed-size models (MobileNet, ResNet, DenseNet, zentangledream) dream each tile at their fixed 224/299 inside the tile.
v1.21: Dreamweave zooms way deeper into the photo, with more knobs and presets to play with. Three new Dreamweave-only controls: Style zoom (0.90–0.995, default 0.970 — was a fixed 0.985, so the default now dives to ~16% of the photo by the end of a 15s clip instead of ~40%), Style dreams (how often the style photo dreams: every frame / 2nd / 3rd / 6th), and Style steps (1–5 dream steps per style dream). The knobs are read live, so you can turn the zoom or the morph speed up and down while the video is dreaming. Three new presets: Deep Vortex (max weave, style morphs every frame), Meltdown (the style photo almost takes over, 5 style steps), Fractal Dive (deepest style zoom, 0.94). 12 video presets total. The fastest-morph presets dream more per frame, so they take longer per frame — shorter clips or 8 fps keep them quick.
v1.20: video fixes + trippier Dreamweave + cleaner buttons + status-bar fix. Videos sometimes never started: a seek that never landed hung the whole run with no message, and the Dream button could stay wedged if the recorder failed to start. Now every seek has an 8s timeout with a plain message, the video is primed (first frame readable) before the loop, every early exit says why, and the recorder can't wedge the button. Dreamweave is trippier: the style photo now dreams every 3rd frame (was every 6th) with 3 steps and a faster zoom, so the style visibly morphs as the video plays; the default preset is stronger (steps 3, mix 0.70, weave 0.50); new Acid Weave preset (steps 5, mix 0.95, weave 0.75) is the maximum trip; Weave slider now goes to 0.75, Steps to 5. Export glitch fixed: Save video was fighting the playback loop over the same viewfinder — the export now stops playback first and resumes it after, so what you watch while it records is the export itself, smooth and in order. The Video tab also shows a live preview now: each dreamed frame appears in the square as it is made, and the finished video takes over at the end. Buttons cleaned up: the four post-dream actions share one row equally, Upload fills its row, and the Engine button reads "Auto · tap to switch" instead of a bare dash before the model loads. The header now sits below the status bar (safe-area inset) instead of under it.
v1.19: smooth past 50, for real this time — the jump came back because Continue started a run with the playback loop still live, so old frames and new frames fought over the same canvas. zoomRun now stops playback first. Dream strength (lr + sharpen) now eases with the zoom, so a long dream settles into a gentle drift instead of churning in place, and the playback tick only advances once the previous frame has landed, so frames can't overtake each other. Verified in TF.js with a 100-frame run (Inception v1): no NaN, depth 11.22x at frame 100, and frames 51-100 change 0.038 per frame vs 0.065 before (59%), falling to 0.020 by frames 76-100.

v1.18: long dreams stay smooth — the zoom used to reset its magnification every 50 frames while the picture was already ~6x deep, so the true depth compounded 6x per run (35x by frame 100, 200x by frame 150) and the dream collapsed into glitch past 50. The working image and its depth now live across Continue runs, the zoom eases to a stop at 16x, the centre wanders smoothly instead of jumping, and the dream strength never fades out. Verified in TF.js with a 100-frame run (Inception v1): no NaN, and the second 50 frames change LESS per frame than the first (0.065 vs 0.084) — it settles instead of spazzing.

v1.17: zentangledream — Coby's own model, a MobileNetV2 fine-tuned in Colab on his Instagram art (deepdream / zentangle / graphics / dreamlings, 352 images). The Colab converter couldn't install, so the .h5 was converted here and gradient-verified in TF.js 4.22.0 CPU — block_10/13/16 all 100% nonzero gradients, forward 224 gives 4 classes. All 12 models real and bundled in the APK (542 MB total), with 3 new presets (Zentangle drift / weave / bloom).

v1.16: the zoom is capped at 50 frames per run again — it stops after 50, you switch the preset or model (or anything else), then Continue for 50 more, up to 500 total. The new model loads when you Continue, so the next 50 dream with it. Export size is new: saved videos and stills can now go up to 2K or 4K (720p / 1080p / 2K / 4K in settings). The dream itself still tops out at Max 1024 — a 2K/4K image through the model would take minutes a frame and run the phone out of memory — so 2K/4K is a true 2K/4K file, upscaled with high-quality filtering; sharpest if the dream was made at Max 1024.

v1.15: Deep Zoom is now the main way — merged into Shoot & Dream. There is one zoom now, the deep one: free-form, no grid, it upscales when it runs out of pixels. The separate Deep Zoom button is gone, and Shoot & Dream, Continue, and photo upload all use it. The old capped path-based zoom is removed.

v1.14: new Deep Zoom mode — a "🔍 Deep Zoom" button next to Shoot & Dream. It zooms in and keeps dreaming, free-form, with no grid: the zoom centre wanders a little every frame so the pattern never locks onto a grid, it dreams new layers in at every zoom level, and when it has magnified so far that it is running out of pixels it upscales the working image (up to 512px, or the card size if bigger) so the dream always has fresh room to grow. It runs until you hit Stop or 500 frames, and it works from a fresh photo or continues the dream you already have. Save it as a video when you are done.

v1.13: rebuilt the video export the way you asked — it now records the dream area on screen as it plays (zoom in, then back out) and spits it out as an MP4. The export renders to a dedicated canvas at up to 720p (the card itself is small, so recording it 1:1 made a tiny blurry file) at 10 Mbps, and the de-grid blur is gentler now (10% → 5% per frame) so the frames stay sharper through the zoom instead of washing out soft. The viewfinder shows the dream playing while it records.

v1.12: fixed glitchy saved videos — the recorder was capturing in real time while each frame takes seconds to dream, so it recorded dozens of duplicate frames per dreamed frame and the video came out minutes long, choppy and glitchy. It now captures exactly one frame per dreamed frame (captureStream(0) + requestFrame), so saved videos are the right length and smooth — this covers both the Video tab and the still Save video button. Way more presets: 82 camera presets now (was 46) — v1 12, v2 10, v3 8, VGG16 8, MobileNetV2 7, MobileNetV1 7, VGG19 6, ResNet50 6, EfficientNetB0 6, DenseNet121 6, InceptionResNetV2 6 — every layer reference verified against the model; plus 4 new video presets (Slow Burn, Weave Light, Transfer Deep, Hyper Weave) for 8 total.

v1.11: the video Save button now works in the app — the WebView ignores the download attribute, so the page hands the file to the app, which saves it into Downloads (this also fixes the still photo/video Save buttons, which had the same bug). Videos are now capped at 15s (was 30s). Dreamweave goes deeper: the style photo now zooms in and dreams along with the video (extra inception), instead of staying a static overlay. More Video tab options: a Preset picker (Gentle / Dreamweave / Deep Inception / Photo Transfer), a Weave slider for how much of the last frame carries over, and a "Zoom in while dreaming" toggle for the video itself.

v1.10: fixed the video picker only showing photos — the app's file chooser hardcoded image/*, so tapping "Choose a video" opened a photo-only picker. It now uses the MIME type the page asked for, so the video picker shows videos and the photo pickers still show photos. The zooming frame number and time estimate now sit right under the progress bar at the top of the settings, instead of buried at the bottom.

v1.9: fixed the bottom settings panel not scrolling at all — the v1.8 tab wrapper broke the flex layout, so the settings grew past the screen and the bottom options (Engine, Continue, Upload, gallery) could not be swiped to. The tab panes are now proper flex columns, so the settings area gets a real height and scrolls under the pinned viewfinder, in both the Camera and Video tabs.

v1.8: the grid is washed out further — default jitter 16 → 24 (slider now to 48) and a light de-grid blur mix (10%) after every frame's dream steps. New logo: the Dreamlings pixel-D eye set into the camera lens (dream eye camera), now the app icon + web favicon/header. Preset "Silk road" renamed "Velvet fur". New Video tab: deepdream a whole video (up to 30s) in three modes — Frame Dream (dream every frame, stitch it back), Photo Transfer (a photo becomes the dream overlay), and Dreamweave (the style photo is dreamed live while the video plays, so the style itself grows, and each frame is woven out of the last so it flows without flicker). Fixed the start-of-dream zoom/crop jump: fixed-size models now letterbox the photo into their square and un-letterbox it back, so the dream starts exactly where the photo is — same aspect ratio, same framing, no squeeze, no crop.

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
