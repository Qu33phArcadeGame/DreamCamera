#!/bin/bash
# Download the 5 real DreamCamera models into web/models/ for offline use
# (web app "Local" source, or the Android APK build).
# v1/v2/v3/vgg16 come from the live DeepdreamlingsFusion site;
# mobilenetv2 comes from Google Cloud Storage (tfjs-models).
# Run from the repo root: bash web/tools/download_models.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p models/inceptionv1 models/inceptionv2 models/inceptionv3 models/vgg16 models/mobilenetv2
BASE="https://qu33pharcadegame.github.io/DeepdreamlingsFusion/models"
for m in inceptionv1 inceptionv2 inceptionv3 vgg16; do
  echo "== $m =="
  curl -sL "$BASE/$m/model.json" -o "models/$m/model.json"
done
echo "== mobilenetv2 =="
curl -sL "https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json" -o "models/mobilenetv2/model.json"
echo "Now fetching weight shards (run the python fetcher):"
python3 - << 'EOF'
import json, urllib.request, os
jobs = [
    ("inceptionv1", "https://qu33pharcadegame.github.io/DeepdreamlingsFusion/models/inceptionv1/"),
    ("inceptionv2", "https://qu33pharcadegame.github.io/DeepdreamlingsFusion/models/inceptionv2/"),
    ("inceptionv3", "https://qu33pharcadegame.github.io/DeepdreamlingsFusion/models/inceptionv3/"),
    ("vgg16", "https://qu33pharcadegame.github.io/DeepdreamlingsFusion/models/vgg16/"),
    ("mobilenetv2", "https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/"),
]
for m, base in jobs:
    d = json.load(open(f"models/{m}/model.json"))
    for g in d["weightsManifest"]:
        for p in g["paths"]:
            dst = f"models/{m}/{p}"
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                print(f"skip {dst}"); continue
            print(f"get {dst}")
            urllib.request.urlretrieve(base + p, dst)
    print(m, "done")
EOF
du -sh models/*
echo "Done. In the web app, set Model source to Local."
