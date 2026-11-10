# DreamCamera Android app

WebView wrapper of the web app (`web/`). Package `com.dreamcamera.app`.

## Build

```bash
bash android/build_apk.sh
```

Makes `dreamcamera-v1.apk` in the repo root (staging dir). Needs
`web/models/` downloaded first for offline models:
`bash web/tools/download_models.sh`.

## What it does

- Loads `file:///android_asset/index.html` with file-URL access on, so the
  page can fetch the bundled `models/` folder.
- Asks for the Camera permission at runtime and grants the page's
  `getUserMedia` video request.
- Models: all five real models bundled (~93 MB), so it dreams fully offline.
  Set Model source to Live site in the app to use online models instead.
- Nothing is uploaded anywhere. No game, no dreamlings.

## Keystore

Own keystore: `android/dreamcamera-upload.keystore`, password in
`android/.kspass`. Never reuse the QU33PH or Dreamlings keystores.
Back both files up — losing them means a new app listing.
