#!/bin/bash
# build_apk.sh — DreamCamera Android wrapper (com.dreamcamera.app).
# Usage: bash android/build_apk.sh
# Stages web/ (+ downloaded models in web/models/) into apk-stage/,
# builds a signed APK with aapt2 + d8 + zipalign + apksigner.
# Keystore: android/dreamcamera-upload.keystore, password in android/.kspass
# (same pattern as the Dreamlings Fusion app — its own keystore, never shared).
set -e
export JAVA_HOME=/home/hatch/jdk-17
export PATH=$JAVA_HOME/bin:$PATH
BT=/home/hatch/android-sdk/build-tools/35.0.1
AJAR=/home/hatch/android-sdk/platforms/android-35/android.jar
WB=/home/hatch/workspace/dreamcamera
APP=$WB/android/app/src/main
STAGE=$WB/apk-stage
BB=$WB/apk-tmp
VCODE=1
VNAME="1.0"
MINSDK=24
TGTSDK=35
OUT=$WB/dreamcamera-v1.apk

# keystore (create once, keep the password file with it)
if [ ! -f "$WB/android/dreamcamera-upload.keystore" ]; then
  KSPASS=$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 24)
  printf '%s' "$KSPASS" > "$WB/android/.kspass"
  keytool -genkeypair -keystore "$WB/android/dreamcamera-upload.keystore" -storepass "$KSPASS" -keypass "$KSPASS" \
    -alias dreamcamera -keyalg RSA -keysize 2048 -validity 10000 \
    -dname "CN=DreamCamera, O=Coby" 2>/dev/null
  echo "keystore created"
fi

rm -rf "$STAGE" "$BB" && mkdir -p "$STAGE" "$BB"
cp -r "$WB/web/"* "$STAGE/"
if [ ! -d "$STAGE/models/inceptionv1" ]; then
  echo "WARNING: web/models/ is empty — run web/tools/download_models.sh first for offline models."
fi
cd "$APP"
sed 's|<manifest xmlns:android="http://schemas.android.com/apk/res/android">|<manifest xmlns:android="http://schemas.android.com/apk/res/android"\n    package="com.dreamcamera.app">|' AndroidManifest.xml > $BB/AndroidManifest.xml
$BT/aapt2 compile --dir res -o $BB/compiled_res.zip
$BT/aapt2 link -o $BB/unsigned.apk --manifest $BB/AndroidManifest.xml -I $AJAR -A $STAGE $BB/compiled_res.zip --java $BB/gen --min-sdk-version $MINSDK --target-sdk-version $TGTSDK --version-code $VCODE --version-name "$VNAME" --no-compress-regex '\.(json|wasm|bin)$'
mkdir -p $BB/classes
javac -encoding UTF-8 -source 8 -target 8 -bootclasspath $AJAR -d $BB/classes $BB/gen/com/dreamcamera/app/R.java $(find $APP/java -name "*.java")
mkdir -p $BB/dexout
$BT/d8 --lib $AJAR --min-api $MINSDK --output $BB/dexout $BB/classes/com/dreamcamera/app/*.class
(cd $BB/dexout && zip -q -j $BB/unsigned.apk classes.dex)
$BT/zipalign -f 4 $BB/unsigned.apk $OUT
export KSPASS=$(cat $WB/android/.kspass)
$BT/apksigner sign --ks $WB/android/dreamcamera-upload.keystore --ks-pass env:KSPASS --key-pass env:KSPASS $OUT
$BT/apksigner verify $OUT
echo "=== APK DONE: $OUT ==="
ls -lh $OUT | awk '{print $9, $5}'
$BT/aapt2 dump badging $OUT | head -n 5
