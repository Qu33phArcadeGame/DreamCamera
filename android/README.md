# DreamCamera Android app

WebView wrapper of the web app (`web/`). Package `com.dreamcamera.app`.

## Build

```bash
bash android/build_apk.sh
```

Makes `dreamcamera-v<VERSION>.apk` in the repo root (staging dir) — the
version sits next to the name, before `.apk`, so the file installs as an APK.
Needs `web/models/` downloaded first for offline models:
`bash web/tools/download_models.sh`.

## What it does

- Loads `file:///android_asset/index.html` with file-URL access on, so the
  page can fetch the bundled `models/` folder.
- Asks for the Camera permission at runtime and grants the page's
  `getUserMedia` video request.
- Models: all eleven real models bundled (~535 MB), so it dreams fully offline.
  Set Model source to Live site in the app to use online models instead.
- Home screen widgets: DreamCamera 2x2 (latest dream, tap to open) and
  DreamCamera Shoot 3x3 (latest dream + Shoot & Dream button). The page
  hands each finished dream to the app through the DreamCameraBridge, which
  saves it as latest_dream.jpg for the widgets to show.
- Nothing is uploaded anywhere. No game, no dreamlings.

## Keystore

Own keystore: `android/dreamcamera-upload.keystore`, password in
`android/.kspass`. Never reuse the QU33PH or Dreamlings keystores.
Back both files up — losing them means a new app listing.
