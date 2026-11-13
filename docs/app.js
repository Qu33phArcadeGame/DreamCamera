// DreamCamera v1.8 — deepdream camera. Web app.
// Gradient-ascent dream engine: MODELS registry, PRE pixel math,
// dreamStep, zoomCrop + path offsets, octave start-dream, engine fallback.
// Photo ratios 1:1, 9:16, 16:9. 50 frames per run, IndexedDB gallery, MediaRecorder video.
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
// conv_pw_11/12; fixed 224 input like mobilenetv2). New in v1.5 (verified
// 2026-11-10, real gradients in TF.js 4.22.0 CPU): VGG19, ResNet50,
// EfficientNetB0 — all three bundled in the APK. New in v1.6 (converted
// from Keras weights 2026-11-10, real gradients verified): DenseNet121,
// InceptionResNetV2 (graph model — its layers version cannot load, the
// residual blocks use a custom layer TF.js rejects). New in v1.17 (converted
// 2026-11-11, real gradients verified in TF.js 4.22.0 CPU): zentangledream —
// MobileNetV2, fine-tuned in Colab on Instagram art
// (deepdream / Zentangle / graphics / dreamlings), fixed 224, v2 preprocessing.
// All 12 models real.
const LIVE_BASE = "https://qu33pharcadegame.github.io/DeepdreamlingsFusion/models/";
const MODELS = {
  v1: { short: "v1", label: "Inception v1", kind: "graph", live: LIVE_BASE + "inceptionv1/model.json", local: "models/inceptionv1/model.json", pre: "v1",
    layers: ["mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"],
    pref: ["mixed4c", "mixed4d"], note: "Mid layers make eyes and creatures." },
  v2: { short: "v2", label: "Inception v2", kind: "graph", live: LIVE_BASE + "inceptionv2/model.json", local: "models/inceptionv2/model.json", pre: "v2",
    layers: ["InceptionV2/InceptionV2/Mixed_3b/concat", "InceptionV2/InceptionV2/Mixed_3c/concat", "InceptionV2/InceptionV2/Mixed_4a/concat", "InceptionV2/InceptionV2/Mixed_4b/concat", "InceptionV2/InceptionV2/Mixed_4c/concat", "InceptionV2/InceptionV2/Mixed_4d/concat", "InceptionV2/InceptionV2/Mixed_4e/concat", "InceptionV2/InceptionV2/Mixed_5a/concat", "InceptionV2/InceptionV2/Mixed_5b/concat", "InceptionV2/InceptionV2/Mixed_5c/concat"],
    pref: ["InceptionV2/InceptionV2/Mixed_4d/concat", "InceptionV2/InceptionV2/Mixed_4e/concat"], note: "Wilder creatures, noisier patterns." },
  v3: { short: "v3", label: "Inception v3", kind: "layers", live: LIVE_BASE + "inceptionv3/model.json", local: "models/inceptionv3/model.json", pre: "v3",
    layers: ["mixed2", "mixed3", "mixed4", "mixed5", "mixed6", "mixed7"],
    pref: ["mixed6", "mixed7"], note: "High layers make fur and creature faces." },
  vgg16: { short: "vgg16", label: "VGG16", kind: "layers", live: LIVE_BASE + "vgg16/model.json", local: "models/vgg16/model.json", pre: "vgg",
    layers: ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"],
    pref: ["block4_conv3"], note: "Painterly swirls; deep blocks make flowing organic shapes." },
  mobilenetv2: { short: "mobilenetv2", label: "MobileNetV2", kind: "graph", live: "https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json", local: "models/mobilenetv2/model.json", pre: "v2", trustLayers: true, minSize: 224, fixedSize: 224,
    layers: ["module_apply_default/MobilenetV2/expanded_conv_5/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_7/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_10/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_16/project/BatchNorm/FusedBatchNorm"],
    pref: ["module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm"], note: "Small and fast, clean lines." },
  mobilenetv1: { short: "mobilenetv1", label: "MobileNetV1", kind: "layers", live: "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json", local: "models/mobilenetv1/model.json", pre: "v2", minSize: 224, fixedSize: 224,
    layers: ["conv_pw_9_relu", "conv_pw_10_relu", "conv_pw_11_relu", "conv_pw_12_relu", "conv_pw_13_relu"],
    pref: ["conv_pw_11_relu", "conv_pw_12_relu"], note: "Tiny and quick, soft sketchy dreams." },
  vgg19: { short: "vgg19", label: "VGG19", kind: "layers", live: "https://raw.githubusercontent.com/paulsp94/tfjs_vgg19_imagenet/master/model/model.json", local: "models/vgg19/model.json", pre: "vgg",
    layers: ["block3_conv4", "block4_conv4", "block5_conv4"],
    pref: ["block5_conv4", "block4_conv4"], note: "Painterly swirls, deep and flowing." },
  resnet50: { short: "resnet50", label: "ResNet50", kind: "layers", live: "https://raw.githubusercontent.com/paulsp94/tfjs_resnet_imagenet/master/ResNet50/model.json", local: "models/resnet50/model.json", pre: "vgg", fixedSize: 224,
    layers: ["activation_21", "activation_39"],
    pref: ["activation_39", "activation_21"], note: "Deep residual layers, rich texture." },
  efficientnetb0: { short: "efficientnetb0", label: "EfficientNet B0", kind: "graph", live: "https://raw.githubusercontent.com/ntedgi/efficientnet-tensorflowjs-binaries/main/models/B0/model.json", local: "models/efficientnetb0/model.json", pre: "eff", trustLayers: true,
    layers: ["StatefulPartitionedCall/model/stack_2/block_1/add/add", "StatefulPartitionedCall/model/stack_4/block_2/add/add", "StatefulPartitionedCall/model/stack_6/block_0/project_bn/FusedBatchNormV3", "StatefulPartitionedCall/model/top_activation/mul"],
    pref: ["StatefulPartitionedCall/model/stack_4/block_2/add/add", "StatefulPartitionedCall/model/stack_6/block_0/project_bn/FusedBatchNormV3"], note: "Efficient and sharp." },
  densenet121: { short: "densenet121", label: "DenseNet121", kind: "layers", live: "https://raw.githubusercontent.com/Qu33phArcadeGame/DreamCamera/main/web/models/densenet121/model.json", local: "models/densenet121/model.json", pre: "torch", fixedSize: 224,
    layers: ["conv3_block12_concat", "conv4_block24_concat", "conv5_block16_concat"],
    pref: ["conv4_block24_concat", "conv5_block16_concat"], note: "Dense connections. Sharp, detailed dreams." },
  inceptionresnetv2: { short: "inceptionresnetv2", label: "InceptionResNetV2", kind: "graph", live: "https://raw.githubusercontent.com/Qu33phArcadeGame/DreamCamera/main/web/models/inceptionresnetv2/model.json", local: "models/inceptionresnetv2/model.json", pre: "v2", trustLayers: true, fixedSize: 299,
    layers: ["StatefulPartitionedCall/inception_resnet_v2/block35_10_ac/Relu", "StatefulPartitionedCall/inception_resnet_v2/block17_10_ac/Relu", "StatefulPartitionedCall/inception_resnet_v2/block17_18_ac/Relu"],
    pref: ["StatefulPartitionedCall/inception_resnet_v2/block35_10_ac/Relu", "StatefulPartitionedCall/inception_resnet_v2/block17_10_ac/Relu"], note: "Inception + residual. Rich, layered dreams." },
  zentangledream: { short: "zentangledream", label: "zentangledream", kind: "layers", live: "https://raw.githubusercontent.com/Qu33phArcadeGame/DreamCamera/main/web/models/zentangledream/model.json", local: "models/zentangledream/model.json", pre: "v2", fixedSize: 224,
    layers: ["block_10_project_BN", "block_13_project_BN", "block_16_project_BN"],
    pref: ["block_13_project_BN"], note: "Zentangle art — MobileNetV2 fine-tuned on Instagram art." },
};
const COMING_SOON = [];
// Per-model pixel math (same as the Dreamlings generator).
const PRE = {
  v1: { norm: t => t.sub(117), denorm: t => t.add(117).div(255), lo: -117, hi: 138, noise: [-10, 10] },
  v3: { norm: t => t.div(127.5).sub(1), denorm: t => t.add(1).div(2), lo: -1, hi: 1, noise: [-1, 1] },
  v2: { norm: t => t.div(127.5).sub(1), denorm: t => t.add(1).div(2), lo: -1, hi: 1, noise: [-1, 1] },
  vgg: { norm: t => t.reverse(-1).sub(tf.tensor1d([103.939, 116.779, 123.68])),
         denorm: t => t.add(tf.tensor1d([103.939, 116.779, 123.68])).reverse(-1).div(255),
         lo: -103.939, hi: 151.061, noise: [-10, 10] },
  eff: { norm: t => t.div(255), denorm: t => t, lo: 0, hi: 1, noise: [-0.05, 0.05] },   // EfficientNet: scale to [0,1]
  torch: { norm: t => t.div(255).sub(tf.tensor1d([0.485, 0.456, 0.406])).div(tf.tensor1d([0.229, 0.224, 0.225])),
           denorm: t => t.mul(tf.tensor1d([0.229, 0.224, 0.225])).add(tf.tensor1d([0.485, 0.456, 0.406])),
           lo: -2.118, hi: 2.64, noise: [-0.05, 0.05] },   // DenseNet: x/255, then (x-mean)/std per channel (RGB)
};
const WASM_PATH = "tfjs/";
const QUALITY = { turbo: { w: 192, h: 192 }, fast: { w: 256, h: 256 }, normal: { w: 384, h: 384 }, high: { w: 512, h: 512 }, ultra: { w: 768, h: 768 }, max: { w: 1024, h: 1024 }, twoK: { w: 1440, h: 1440 } };
// Export size is the long side of the saved video/still. A still saved
// bigger than the dream is dreamed for real at that size, tile by tile
// (dreamTiled) — so 4K/8K stills are actual 4K/8K deepdreams, not an
// upscale. Video stays at 4K max: a true 8K video would need the tiled
// dream on every frame of the card, so 8K is stills only.
const EXPORT_SIZE = { "720p": 1280, "1080p": 1920, "2k": 2560, "4k": 3840, "8k": 7680 };
const EXPORT_BITRATE = { "720p": 8000000, "1080p": 12000000, "2k": 18000000, "4k": 30000000, "8k": 60000000 };
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
  { m: "v1", name: "First dream", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 22, lr: 0.05, sharpen: 0.35, layers: ["mixed4c", "mixed4d", "mixed4e"] } },
  { m: "v1", name: "Spiral melt", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["mixed4b", "mixed4c"] } },
  { m: "v2", name: "Woods noise", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 8, lr: 0.04, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_4d/concat"] } },
  { m: "v2", name: "Wave trip", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_4c/concat", "InceptionV2/InceptionV2/Mixed_4d/concat"] } },
  { m: "v3", name: "Fur storm", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 12, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["mixed6", "mixed7"] } },
  { m: "v3", name: "Velvet fur", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["mixed2", "mixed3"] } },
  { m: "vgg16", name: "Petal storm", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["block3_conv3"] } },
  { m: "vgg16", name: "Deep kelp", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["block4_conv3"] } },
  { m: "vgg16", name: "Ghost garden", s: { path: "straight", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.04, sharpen: 0.3, layers: ["block5_conv3"] } },
  { m: "mobilenetv2", name: "Quick sketch", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv2", name: "Pocket spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_10/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv1", name: "Tiny dream", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["conv_pw_11_relu", "conv_pw_12_relu"] } },
  { m: "vgg19", name: "Deep petals", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["block4_conv4"] } },
  { m: "vgg19", name: "VGG19 garden", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["block5_conv4", "block4_conv4"] } },
  { m: "resnet50", name: "Residual bloom", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["activation_39"] } },
  { m: "resnet50", name: "Deep residual", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["activation_39", "activation_21"] } },
  { m: "efficientnetb0", name: "Efficient swirl", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["StatefulPartitionedCall/model/stack_4/block_2/add/add"] } },
  { m: "efficientnetb0", name: "B0 bloom", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["StatefulPartitionedCall/model/stack_4/block_2/add/add", "StatefulPartitionedCall/model/stack_6/block_0/project_bn/FusedBatchNormV3"] } },
  { m: "densenet121", name: "Dense bloom", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["conv4_block24_concat"] } },
  { m: "densenet121", name: "Dense garden", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["conv5_block16_concat", "conv4_block24_concat"] } },
  { m: "inceptionresnetv2", name: "Residual storm", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["StatefulPartitionedCall/inception_resnet_v2/block17_10_ac/Relu"] } },
  { m: "inceptionresnetv2", name: "IRv2 spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["StatefulPartitionedCall/inception_resnet_v2/block35_10_ac/Relu", "StatefulPartitionedCall/inception_resnet_v2/block17_10_ac/Relu"] } },
  { m: "v1", name: "Classic wave", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["mixed4a", "mixed4c"] } },
  { m: "v2", name: "Woods swirl", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_4e/concat"] } },
  { m: "v2", name: "Deep woods", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 12, lr: 0.045, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_5a/concat", "InceptionV2/InceptionV2/Mixed_5b/concat"] } },
  { m: "v3", name: "Fur drift", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["mixed4", "mixed5"] } },
  { m: "v3", name: "Silk spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["mixed7"] } },
  { m: "vgg16", name: "Kelp spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["block5_conv3"] } },
  { m: "mobilenetv2", name: "Pocket wave", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_7/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv2", name: "Quick bloom", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_16/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv1", name: "Tiny swirl", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["conv_pw_10_relu", "conv_pw_11_relu"] } },
  { m: "mobilenetv1", name: "Tiny wave", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["conv_pw_9_relu"] } },
  { m: "mobilenetv1", name: "Pocket garden", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["conv_pw_13_relu"] } },
  { m: "vgg19", name: "Deep swirl", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["block3_conv4"] } },
  { m: "vgg19", name: "VGG19 bloom", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 12, lr: 0.05, sharpen: 0.35, layers: ["block5_conv4"] } },
  { m: "resnet50", name: "Residual wave", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["activation_21"] } },
  { m: "resnet50", name: "Residual spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["activation_39"] } },
  { m: "efficientnetb0", name: "B0 wave", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["StatefulPartitionedCall/model/stack_2/block_1/add/add"] } },
  { m: "efficientnetb0", name: "B0 spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["StatefulPartitionedCall/model/top_activation/mul"] } },
  { m: "densenet121", name: "Dense wave", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["conv3_block12_concat"] } },
  { m: "densenet121", name: "Dense spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["conv5_block16_concat"] } },
  { m: "inceptionresnetv2", name: "IRv2 wave", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["StatefulPartitionedCall/inception_resnet_v2/block17_18_ac/Relu"] } },
  { m: "inceptionresnetv2", name: "IRv2 bloom", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 12, lr: 0.05, sharpen: 0.35, layers: ["StatefulPartitionedCall/inception_resnet_v2/block35_10_ac/Relu"] } },
  { m: "v1", name: "Tiny eyes", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["mixed3a", "mixed3b"] } },
  { m: "v1", name: "Deep dive", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 14, lr: 0.05, sharpen: 0.35, layers: ["mixed5a", "mixed5b"] } },
  { m: "v1", name: "Creature soup", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 12, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["mixed4a", "mixed4e"] } },
  { m: "v1", name: "Neon garden", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["mixed3b", "mixed4d"] } },
  { m: "v1", name: "Slow spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["mixed5a"] } },
  { m: "v1", name: "Eye candy", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 16, lr: 0.055, sharpen: 0.35, layers: ["mixed4b", "mixed4e"] } },
  { m: "v2", name: "Early woods", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 8, lr: 0.04, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_3b/concat", "InceptionV2/InceptionV2/Mixed_3c/concat"] } },
  { m: "v2", name: "Mid woods", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_4a/concat", "InceptionV2/InceptionV2/Mixed_4b/concat"] } },
  { m: "v2", name: "Deep canopy", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 12, lr: 0.045, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_5c/concat"] } },
  { m: "v2", name: "Woods drift", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_4b/concat", "InceptionV2/InceptionV2/Mixed_5a/concat"] } },
  { m: "v2", name: "Canopy spiral", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.045, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_5b/concat"] } },
  { m: "v2", name: "Forest floor", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.04, sharpen: 0.35, layers: ["InceptionV2/InceptionV2/Mixed_3c/concat", "InceptionV2/InceptionV2/Mixed_4a/concat"] } },
  { m: "v3", name: "Soft fur", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["mixed2", "mixed4"] } },
  { m: "v3", name: "Wild fur", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 12, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["mixed3", "mixed6"] } },
  { m: "v3", name: "Fur wave", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["mixed5", "mixed7"] } },
  { m: "v3", name: "Deep fur", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 14, lr: 0.05, sharpen: 0.35, layers: ["mixed4", "mixed6", "mixed7"] } },
  { m: "vgg16", name: "Early paint", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["block1_conv2"] } },
  { m: "vgg16", name: "Soft paint", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 8, lr: 0.045, sharpen: 0.35, layers: ["block2_conv2"] } },
  { m: "vgg16", name: "Paint drift", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["block1_conv2", "block3_conv3"] } },
  { m: "vgg16", name: "Deep paint", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 12, lr: 0.045, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["block2_conv2", "block4_conv3", "block5_conv3"] } },
  { m: "mobilenetv2", name: "Pocket early", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_5/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv2", name: "Pocket mid", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_5/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_10/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv2", name: "Pocket deep", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["module_apply_default/MobilenetV2/expanded_conv_16/project/BatchNorm/FusedBatchNorm", "module_apply_default/MobilenetV2/expanded_conv_13/project/BatchNorm/FusedBatchNorm"] } },
  { m: "mobilenetv1", name: "Tiny early", s: { path: "straight", frames: 50, zoomFactor: 0.96, steps: 8, lr: 0.05, sharpen: 0.35, layers: ["conv_pw_9_relu", "conv_pw_10_relu"] } },
  { m: "mobilenetv1", name: "Tiny deep", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["conv_pw_12_relu", "conv_pw_13_relu"] } },
  { m: "mobilenetv1", name: "Tiny mix", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 8, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["conv_pw_9_relu", "conv_pw_12_relu"] } },
  { m: "vgg19", name: "VGG19 early", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["block3_conv4", "block4_conv4"] } },
  { m: "vgg19", name: "VGG19 full", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 12, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["block3_conv4", "block4_conv4", "block5_conv4"] } },
  { m: "resnet50", name: "Residual drift", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 12, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["activation_21"] } },
  { m: "resnet50", name: "Residual melt", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 14, lr: 0.055, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["activation_39"] } },
  { m: "efficientnetb0", name: "B0 early", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["StatefulPartitionedCall/model/stack_2/block_1/add/add", "StatefulPartitionedCall/model/stack_4/block_2/add/add"] } },
  { m: "efficientnetb0", name: "B0 full", s: { path: "square", frames: 50, zoomFactor: 0.97, steps: 12, lr: 0.05, range: 0.08, segFrames: 8, sharpen: 0.35, layers: ["StatefulPartitionedCall/model/stack_2/block_1/add/add", "StatefulPartitionedCall/model/stack_4/block_2/add/add", "StatefulPartitionedCall/model/stack_6/block_0/project_bn/FusedBatchNormV3", "StatefulPartitionedCall/model/top_activation/mul"] } },
  { m: "densenet121", name: "Dense full", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 12, lr: 0.05, sharpen: 0.35, layers: ["conv3_block12_concat", "conv4_block24_concat", "conv5_block16_concat"] } },
  { m: "densenet121", name: "Dense early", s: { path: "wave", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.10, segFrames: 8, sharpen: 0.35, layers: ["conv3_block12_concat", "conv4_block24_concat"] } },
  { m: "inceptionresnetv2", name: "IRv2 full", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 12, lr: 0.05, sharpen: 0.35, layers: ["StatefulPartitionedCall/inception_resnet_v2/block35_10_ac/Relu", "StatefulPartitionedCall/inception_resnet_v2/block17_10_ac/Relu", "StatefulPartitionedCall/inception_resnet_v2/block17_18_ac/Relu"] } },
  { m: "inceptionresnetv2", name: "IRv2 deep", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["StatefulPartitionedCall/inception_resnet_v2/block17_18_ac/Relu", "StatefulPartitionedCall/inception_resnet_v2/block35_10_ac/Relu"] } },
  { m: "zentangledream", name: "Zentangle drift", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["block_13_project_BN"] } },
  { m: "zentangledream", name: "Zentangle weave", s: { path: "spiral", frames: 50, zoomFactor: 0.97, steps: 10, lr: 0.05, range: 0.12, segFrames: 10, sharpen: 0.35, layers: ["block_10_project_BN", "block_13_project_BN"] } },
  { m: "zentangledream", name: "Zentangle bloom", s: { path: "straight", frames: 50, zoomFactor: 0.965, steps: 10, lr: 0.05, sharpen: 0.35, layers: ["block_16_project_BN"] } },
];
const DEFAULTS = { model: "v1", preset: 0, path: "straight", frames: 50, zoomFactor: 0.97, steps: 4, lr: 0.04, range: 0.08, segFrames: 10, sharpen: 0.35, layers: ["mixed4c", "mixed4d"], quality: "turbo", ratio: "1:1", playFps: 12, playFpsOut: 12, boomerang: true, octaves: 3, jitter: 24, facing: "environment", modelSrc: "live", exportSize: "1080p" };

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
function dims() {
  const q = QUALITY[settings.quality] || QUALITY.normal;
  const r = settings.ratio || "1:1";   // quality is the short side; the long side follows the ratio
  if (r === "9:16") return { w: q.w, h: Math.round(q.w * 16 / 9) };
  if (r === "16:9") return { w: Math.round(q.h * 16 / 9), h: q.h };
  return { w: q.w, h: q.h };
}
// Size the viewfinder to the photo ratio and fit it on screen.
function applyRatio() {
  const box = $("viewBox");
  const r = settings.ratio || "1:1";
  let ar = 1;
  if (r === "9:16") ar = 9 / 16; else if (r === "16:9") ar = 16 / 9;
  const availW = box.parentElement.clientWidth - 20;
  const availH = Math.round(window.innerHeight * 0.58);
  let w = availW, h = Math.round(w / ar);
  if (h > availH) { h = availH; w = Math.round(h * ar); }
  box.style.width = w + "px"; box.style.height = h + "px";
  box.style.aspectRatio = "auto"; box.style.margin = "0 auto";
}
function clampFps(v) { v = parseFloat(v); if (!(v > 0)) v = 12; return Math.max(2, Math.min(30, v)); }
function cardFps() { return clampFps(settings.playFps); }
// The size to feed the model at one octave. Aspect is always preserved —
// the octave just scales the photo down. A fixed-size model takes its
// square inside dreamSteps (letterboxed), never by squeezing the photo.
function dreamHW(h, w, s) {
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
function capturePhoto() {
  const q = dims(), c = document.createElement("canvas"); c.width = q.w; c.height = q.h;
  const ctx = c.getContext("2d");
  if (vid.readyState >= 2 && vid.videoWidth) {
    const vw = vid.videoWidth, vh = vid.videoHeight, target = q.w / q.h;
    let sw = vw, sh = vh;   // crop the video to the photo ratio, centred
    if (vw / vh > target) sw = vh * target; else sh = vw / target;
    sw /= camZoom; sh /= camZoom;   // pinch zoom: crop tighter, same as the scaled viewfinder
    ctx.drawImage(vid, (vw - sw) / 2, (vh - sh) / 2, sw, sh, 0, 0, q.w, q.h);
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
// De-grid: the grid is a high-frequency checkerboard the network builds
// up frame after frame. Blending a little of a blurred copy back in each
// frame keeps it from compounding, without softening the dream.
function degrid(img, amt) {
  return tf.tidy(() => {
    const k = [1, 2, 1, 2, 4, 2, 1, 2, 1].map(v => v / 16), ker = tf.tensor4d([].concat(...k.map(v => [v, v, v])), [3, 3, 3, 1]);
    const blur = tf.depthwiseConv2d(img, ker, 1, "same");
    return img.mul(1 - amt).add(blur.mul(amt)).clipByValue(P.lo, P.hi);
  });
}
// Dream `steps` steps on img and return the new tensor (img is disposed).
// The aspect ratio and framing never change: for a fixed-size model the
// image is letterboxed into the model's square, dreamed there, and
// un-letterboxed back — never squeezed into a square and never
// cover-cropped, so the dream starts exactly where the photo is.
function dreamSteps(img, lr, steps, doDegrid) {
  if (!MODEL.fixedSize) {
    let cur = img;
    for (let i = 0; i < steps; i++) { const n = dreamStep(cur, lr); cur.dispose(); cur = n; }
    if (doDegrid) { const d = degrid(cur, 0.05); cur.dispose(); cur = d; }
    return cur;
  }
  const S = MODEL.fixedSize, h = img.shape[1], w = img.shape[2];
  const scale = S / Math.max(h, w);
  const nh = Math.max(1, Math.round(h * scale)), nw = Math.max(1, Math.round(w * scale));
  const top = Math.floor((S - nh) / 2), left = Math.floor((S - nw) / 2);
  let sq = tf.tidy(() => {
    const r = tf.image.resizeBilinear(img, [nh, nw]);
    return tf.pad(r, [[0, 0], [top, S - nh - top], [left, S - nw - left], [0, 0]], P.lo);
  });
  img.dispose();
  for (let i = 0; i < steps; i++) { const n = dreamStep(sq, lr); sq.dispose(); sq = n; }
  if (doDegrid) { const d = degrid(sq, 0.05); sq.dispose(); sq = d; }
  const out = tf.tidy(() => {
    const c = sq.slice([0, top, left, 0], [1, nh, nw, 3]);
    return tf.image.resizeBilinear(c, [h, w]);
  });
  sq.dispose();
  return out;
}
function dreamStep(img, lr) {
  return tf.tidy(() => {
    const sy = Math.floor((Math.random() * 2 - 1) * JITTER), sx = Math.floor((Math.random() * 2 - 1) * JITTER);
    let g = getGrad()(roll(roll(img, sy, 1), sx, 2));
    g = roll(roll(g, -sy, 1), -sx, 2);
    const std = tf.moments(g).variance.sqrt();
    return img.add(g.div(std.add(1e-8)).mul(lr)).clipByValue(P.lo, P.hi);
  });
}
// ===== TILED HIGH-RES DREAM (true 4K / 8K) =====
// A 4K image through the model would take minutes a frame and run the
// phone out of memory — that is why 4K export used to be an upscale of a
// small dream. The real fix is to dream in overlapping tiles: each tile is
// dreamed on its own (bounded memory, one tile at a time), and the tiles
// are blended with a feathered weight that sums to 1 everywhere, so there
// are no seams. The result is real gradient ascent at full resolution,
// not an upscale. One tile is ~0.3s, so a 4K frame is ~50 tiles (~20s)
// and an 8K frame ~220 tiles (~2 min).
const TILE_SIZE = 512, TILE_OVERLAP = 128;
function getTileMask(oLeft, oRight, oTop, oBottom) {
  // Feather weight for a tile: 1 in the middle, ramping to 0 over the
  // overlap at each edge. Each overlap is the tile's ACTUAL overlap with
  // its neighbour on that side (even spacing makes them nearly equal, but
  // rounding makes them differ by a pixel or two — the ramp must span
  // that side's overlap exactly, or the blend sums to more than 1 in the
  // middle). A border edge (null) stays 1, so the image edge is not
  // dimmed. The weight is the product of the horizontal and vertical
  // ramps, so neighbouring tiles' weights sum to exactly 1 in every
  // overlap, corners included.
  const key = [oLeft, oRight, oTop, oBottom].join(",");
  if (getTileMask.cache[key]) return getTileMask.cache[key];
  const c = document.createElement("canvas"); c.width = TILE_SIZE; c.height = TILE_SIZE;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(TILE_SIZE, TILE_SIZE);
  const d = img.data;
  for (let y = 0; y < TILE_SIZE; y++) {
    // Each ramp is clamped to 1 on its own: a pixel far from both edges
    // of an axis has weight 1 on that axis, not the raw (large) ratio —
    // without the clamp the product overshoots and the blend sums to ~2.
    const wy = Math.max(0, Math.min(1, Math.min(oTop == null ? 1 : (y + 0.5) / oTop, oBottom == null ? 1 : (TILE_SIZE - y - 0.5) / oBottom)));
    for (let x = 0; x < TILE_SIZE; x++) {
      const wx = Math.max(0, Math.min(1, Math.min(oLeft == null ? 1 : (x + 0.5) / oLeft, oRight == null ? 1 : (TILE_SIZE - x - 0.5) / oRight)));
      const w = wx * wy;
      const i = (y * TILE_SIZE + x) * 4;
      d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; d[i + 3] = Math.round(w * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
  getTileMask.cache[key] = c;
  return c;
}
getTileMask.cache = {};
function tileAxis(W) {
  // Evenly spaced tile positions covering [0, W - TILE_SIZE]. The stride
  // is kept between 256 and 448 (overlap 64–256), so no tile is ever in
  // two overlaps at once. Target stride is 384 (overlap 128).
  const Q = W - TILE_SIZE;
  if (Q <= 0) return [0];
  const mLo = Math.ceil(Q / 448), mHi = Math.floor(Q / 256);
  let m = Math.round(Q / (TILE_SIZE - TILE_OVERLAP));
  if (mLo <= mHi) m = Math.max(mLo, Math.min(mHi, m));
  if (m < 1) m = 1;
  const pos = [];
  for (let i = 0; i <= m; i++) pos.push(Math.round(i * Q / m));
  return pos;
}
async function dreamTiled(cv, lr, steps, onTile, scratch) {
  // cv: a 2D canvas at the target resolution. Dreamed in place, tile by
  // tile. onTile(done, total) reports progress. scratch (optional) holds
  // the accumulation canvases so a video can reuse them across frames
  // instead of allocating a 4K canvas per frame.
  const W = cv.width, H = cv.height;
  if (W <= 0 || H <= 0) return;
  // A small image is just dreamed whole — no tiling needed.
  if (W <= TILE_SIZE && H <= TILE_SIZE) {
    let t = canvasToTensor(cv);
    t = dreamSteps(t, lr, steps, true);
    await tensorToCanvas(t, cv);
    t.dispose();
    if (onTile) onTile(1, 1);
    return;
  }
  scratch = scratch || {};
  if (!scratch.resCv || scratch.resCv.width !== W || scratch.resCv.height !== H) {
    scratch.resCv = document.createElement("canvas"); scratch.resCv.width = W; scratch.resCv.height = H;
    scratch.wCv = document.createElement("canvas"); scratch.wCv.width = W; scratch.wCv.height = H;
    scratch.tileCv = document.createElement("canvas"); scratch.tileCv.width = TILE_SIZE; scratch.tileCv.height = TILE_SIZE;
    scratch.tmpCv = document.createElement("canvas"); scratch.tmpCv.width = TILE_SIZE; scratch.tmpCv.height = TILE_SIZE;
  }
  const resCtx = scratch.resCv.getContext("2d"), wCtx = scratch.wCv.getContext("2d");
  const tileCtx = scratch.tileCv.getContext("2d"), tmpCtx = scratch.tmpCv.getContext("2d");
  resCtx.clearRect(0, 0, W, H); wCtx.clearRect(0, 0, W, H);
  // Tile grid: evenly spaced positions, so every overlap is about the
  // same width. A stride grid with a clamped last tile makes one overlap
  // up to 400px wide, and then a tile sits in two overlaps at once and
  // the feather weights sum to ~1.1 instead of 1 (a bright band). Even
  // spacing keeps every overlap between 64 and 256px, so no tile is ever
  // in two overlaps at once and the weights always sum to 1.
  const xs = tileAxis(W), ys = tileAxis(H);
  const total = xs.length * ys.length;
  let done = 0;
  for (let yi = 0; yi < ys.length; yi++) for (let xi = 0; xi < xs.length; xi++) {
    const tx = xs[xi], ty = ys[yi];
    // Each edge's ramp spans that side's actual overlap with its
    // neighbour (a border edge has none). The overlaps come from the grid
    // positions, so the two tiles sharing an overlap always agree on it.
    const oLeft = xi === 0 ? null : (xs[xi - 1] + TILE_SIZE) - tx;
    const oRight = xi === xs.length - 1 ? null : (tx + TILE_SIZE) - xs[xi + 1];
    const oTop = yi === 0 ? null : (ys[yi - 1] + TILE_SIZE) - ty;
    const oBottom = yi === ys.length - 1 ? null : (ty + TILE_SIZE) - ys[yi + 1];
    // Extract + dream the tile.
    tileCtx.clearRect(0, 0, TILE_SIZE, TILE_SIZE);
    tileCtx.drawImage(cv, tx, ty, TILE_SIZE, TILE_SIZE, 0, 0, TILE_SIZE, TILE_SIZE);
    let t = canvasToTensor(scratch.tileCv);
    t = dreamSteps(t, lr, steps, true);
    await tensorToCanvas(t, scratch.tileCv);
    t.dispose();
    // Feather mask for this tile (cached by its four overlaps).
    const mask = getTileMask(oLeft, oRight, oTop, oBottom);
    // tmp = dreamed tile with alpha = weight.
    tmpCtx.clearRect(0, 0, TILE_SIZE, TILE_SIZE);
    tmpCtx.drawImage(scratch.tileCv, 0, 0);
    tmpCtx.globalCompositeOperation = "destination-in";
    tmpCtx.drawImage(mask, 0, 0);
    tmpCtx.globalCompositeOperation = "source-over";
    // Accumulate: result += tile*weight, weight += weight. 'lighter' adds
    // source*alpha, and the weights sum to 1, so nothing clips.
    resCtx.globalCompositeOperation = "lighter";
    resCtx.drawImage(scratch.tmpCv, tx, ty);
    resCtx.globalCompositeOperation = "source-over";
    wCtx.globalCompositeOperation = "lighter";
    wCtx.drawImage(mask, tx, ty);
    wCtx.globalCompositeOperation = "source-over";
    done++;
    if (onTile) onTile(done, total);
    await tf.nextFrame();
  }
  // Normalize: out = result / weight. The weights sum to 1, so this only
  // corrects 8-bit rounding — but it also guarantees the edges are right.
  const resImg = resCtx.getImageData(0, 0, W, H);
  const wImg = wCtx.getImageData(0, 0, W, H);
  const outCtx = cv.getContext("2d");
  const outImg = outCtx.getImageData(0, 0, W, H);
  const rd = resImg.data, wd = wImg.data, od = outImg.data;
  for (let i = 0; i < od.length; i += 4) {
    const wsum = wd[i] / 255;   // weight canvas R channel = 255 * weight sum
    const inv = wsum > 0.01 ? 1 / wsum : 1;
    od[i] = Math.max(0, Math.min(255, rd[i] * inv));
    od[i + 1] = Math.max(0, Math.min(255, rd[i + 1] * inv));
    od[i + 2] = Math.max(0, Math.min(255, rd[i + 2] * inv));
    od[i + 3] = 255;
  }
  outCtx.putImageData(outImg, 0, 0);
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
const drawCv = document.createElement("canvas");   // reused offscreen canvas for tensorToCanvas
async function tensorToCanvas(img, cv) {
  const t = tf.tidy(() => P.denorm(img.squeeze([0])).clipByValue(0, 1));
  // A square dream on a non-square canvas must cover, not stretch:
  // draw the tensor to an offscreen canvas, then centre-crop it onto cv.
  drawCv.width = t.shape[1]; drawCv.height = t.shape[0];
  await tf.browser.toPixels(t, drawCv); t.dispose();
  const ctx = cv.getContext("2d");
  const tw = cv.width, th = cv.height, sw = drawCv.width, sh = drawCv.height;
  const scale = Math.max(tw / sw, th / sh), dw = sw * scale, dh = sh * scale;
  ctx.clearRect(0, 0, tw, th);
  ctx.drawImage(drawCv, (tw - dw) / 2, (th - dh) / 2, dw, dh);
}
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
function disposeCard() { if (cur) { cur.dispose(); cur = null; } resetDeep(); card = null; }
function nextNumber() { let n = parseInt(localStorage.getItem(COUNT_KEY) || "0", 10) + 1; try { localStorage.setItem(COUNT_KEY, String(n)); } catch (e) {} return n; }

// ===== SHOOT + DREAM =====
async function shootAndDream() {
  if (busy || zooming) return;
  const shot = capturePhoto();
  if (!await ensureReady()) return;
  disposeCard(); stopPlaying();
  const q = dims();
  card = { id: "dc" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6), no: nextNumber(), name: "Dream " + new Date().toLocaleDateString(), model: MODEL.short, preset: $("presetSel").selectedOptions[0] ? $("presetSel").selectedOptions[0].textContent : "", w: q.w, h: q.h, created: Date.now(), frames: [], pathFrame: 0, zoomTotal: 1, settings: Object.assign({}, settings) };
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
        const t0 = performance.now(); img = dreamSteps(img, Math.max(0.01, settings.lr), 1, false);
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
// The zoom is the deep zoom: free-form, no grid. It keeps zooming in and
// dreaming, and when it has magnified so far that it is running out of
// pixels it upscales the working image (up to 512px, or the card size if
// bigger) so the dream always has fresh room to grow new layers. The zoom
// centre wanders a little every frame, so the pattern never locks onto a grid.
//
// The working image and its magnification live across Continue runs. They
// used to reset every 50 frames: the tracker restarted at 1x while the
// picture was already ~6x deep, so the true depth compounded 6x per run
// (35x by frame 100, 200x by frame 150) and the network ended up dreaming
// on a speck of its own hallucination — that was the past-50 glitch. The
// true depth now eases to a stop at 16x, so a long dream settles into a
// smooth drift instead of collapsing.
// v1.19: the past-50 jump was still there — Continue started a run with
// the playback loop still live, so old frames and new frames fought over
// the canvas. zoomRun now stops playback first. Dream strength also eases
// with the zoom (lr + sharpen scale with the ease), and the playback tick
// only advances once the previous frame has landed, so frames can't
// overtake each other.
const ZOOM_CAP = 3;   // kept for loading old saves (zoomTotal); the zoom itself no longer caps
const ZOOM_MAX = 16;  // total magnification vs the seed: the zoom eases to a stop here
let deep = null, deepZoom = 1, viewDepth = 1, driftX = 0, driftY = 0, lastViewDepth = 1;
function resetDeep() { if (deep) { deep.dispose(); deep = null; } deepZoom = 1; viewDepth = 1; driftX = 0; driftY = 0; lastViewDepth = 1; }
async function zoomRun() {
  if (!card || busy || zooming || !card.frames.length) return;
  zooming = true; stopPlaying();   // the playback loop must not keep drawing old frames into the canvas while the dream runs — that was the past-50 jump: Continue started a run with playback still live
  let failed = false, made = 0; updateUI();
  if (!await ensureReady()) { zooming = false; updateUI(); return; }   // loads the current model (in case it was switched between runs)
  const want = Math.max(1, Math.min(MAX_RUN, Math.round(settings.frames)));   // 50 frames per run; switch preset/model, Continue for 50 more, up to 500 total
  const DEEP_MAX = Math.max(512, Math.max(card.w, card.h));   // longest side the working image grows to
  try {
    if (!deep) {
      await ensureCur();   // cur = last frame (card size); only seeds when there is no live working image
      deep = cur; cur = null;
      // Magnification already baked into the seed: tracked on the card when
      // present, else estimated from how many frames it took to get there.
      viewDepth = (card.viewDepth && card.viewDepth > 1) ? card.viewDepth : Math.min(ZOOM_MAX, 1 + card.frames.length * 0.12);
      lastViewDepth = viewDepth; deepZoom = 1; driftX = 0; driftY = 0;
    }
    while (zooming && made < want && card.frames.length < TOTAL_MAX) {
      try {
        const h = deep.shape[1], w = deep.shape[2];
        // Ease the zoom to a stop as the true depth approaches the cap.
        const ease = Math.max(0, 1 - viewDepth / ZOOM_MAX);
        const zf = 1 - (1 - settings.zoomFactor) * ease;
        // Dream strength settles with the zoom: as the view stops
        // magnifying, the dreaming eases to a gentle drift instead of
        // churning in place (the boil that read as jumpy past frame 50).
        const dreamEase = 0.25 + 0.75 * ease;
        // Smooth wander: the crop centre takes a small step each frame
        // instead of jumping somewhere random (the old random jump each
        // frame read as jitter once the view got deep).
        driftX = Math.max(-1, Math.min(1, driftX + (Math.random() * 2 - 1) * 0.2));
        driftY = Math.max(-1, Math.min(1, driftY + (Math.random() * 2 - 1) * 0.2));
        const cw = Math.max(16, Math.round(w * zf)), ch = Math.max(16, Math.round(h * zf));
        const ox = driftX * (w - cw) / 2, oy = driftY * (h - ch) / 2;
        const z = zoomCrop(deep, zf, ox, oy, (settings.sharpen || 0) * (0.4 + 0.6 / (1 + card.frames.length * 0.01)) * dreamEase);
        deep.dispose(); deep = z;
        viewDepth = viewDepth / zf;   // true depth: upscaling never reduces it, only zooming grows it
        deepZoom = deepZoom / zf;
        const t0 = performance.now();
        // Dream strength eases down with total frames but never fades out:
        // a long dream keeps dreaming, just gentler.
        const lrEff = settings.lr * (0.35 + 0.65 / (1 + card.frames.length * 0.012)) * dreamEase;
        deep = dreamSteps(deep, lrEff, settings.steps, true);   // dream new layers in at this zoom
        // Ran out of pixels? Upscale so the dream has fresh room to grow.
        const longSide = Math.max(deep.shape[1], deep.shape[2]);
        if (deepZoom >= 2 && longSide < DEEP_MAX) {
          const ns = Math.min(DEEP_MAX, longSide * 2);
          const nh = Math.round(deep.shape[1] * ns / longSide), nw = Math.round(deep.shape[2] * ns / longSide);
          const up = tf.image.resizeBilinear(deep, [nh, nw]);
          deep.dispose(); deep = up;
          deepZoom = deepZoom / (ns / longSide);   // honest accounting: the ratio it really grew by
        }
        lastViewDepth = viewDepth; card.viewDepth = viewDepth;
        await tensorToCanvas(deep, artCv); noteStepTime((performance.now() - t0) / Math.max(1, settings.steps));
        await pushFrame(); made++;
        setStatus("Zooming… frame " + made + " of " + want + " · " + Math.max(deep.shape[1], deep.shape[2]) + "px · " + viewDepth.toFixed(1) + "× · " + card.frames.length + " total");
        setBar(made / want);
        await tf.nextFrame();
      } catch (e) {
        console.error(e); if (deep) { deep.dispose(); deep = null; }
        if (!(await nextEngine())) { setStatus("Zoom failed: " + (e && e.message || e), true); failed = true; break; }
        setStatus("Switched to " + LABEL[engineOrder[engineIdx]] + " — carrying on…");
        await ensureCur(); deep = cur; cur = null; viewDepth = lastViewDepth; deepZoom = 1; driftX = 0; driftY = 0;
      }
    }
  } catch (e) {
    console.error(e); setStatus("Zoom failed: " + (e && e.message || e), true); failed = true;
  }
  // deep is NOT disposed here — it lives on as the seed for Continue, with
  // its magnification, so the next run picks up where this one left off.
  zooming = false; setBar(null);
  if (!failed) {
    if (card.frames.length >= TOTAL_MAX) setStatus("Full (" + TOTAL_MAX + " frames). Save it as a video!");
    else if (made >= want) setStatus("Done: " + card.frames.length + " frames. Switch the preset or model, then Continue for 50 more, or save.");
    else setStatus(card.frames.length + " frames. Stopped.");
  }
  sendLatestDreamToWidget();
  startPlaying();
  updateUI();
}

// ===== PLAYBACK =====
function startPlaying() {
  stopPlaying(); if (!card || card.frames.length < 2) return;
  playing = true; let i = 0, last = performance.now(), drawing = false;
  // One decode at a time, in order: the counter only advances once the
  // previous frame has actually landed, so a slow decode delays the next
  // frame instead of letting frames overtake each other (the old code
  // stamped `last` and advanced `i` before the decode, so frames could
  // draw out of order and playback jumped 2-3 frames at a time).
  const tick = async () => {
    if (!playing || !card) return;
    const now = performance.now(), leg = seqLeg(i), fps = leg ? cardFpsOut() : cardFps();
    if (!drawing && now - last >= 1000 / fps) {
      drawing = true; const fi = seqFrame(i); i++;
      const im = await decodeBlob(card.frames[fi]);
      if (im && card && playing) { const c = artCv.getContext("2d"); c.clearRect(0, 0, card.w, card.h); c.drawImage(im, 0, 0, card.w, card.h); if (im.close) im.close(); }
      last = performance.now(); drawing = false;
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
  // Stop the playback loop first: it draws to the same viewfinder the
  // export draws to, so leaving it running makes the picture flicker
  // between the two — the glitchy export. It resumes at the end, so what
  // you watch while it records is the export itself, frame by frame.
  stopPlaying();
  try {
    const mime = videoMime(), L = seqLength(), fpsIn = cardFps(), fpsOut = cardFpsOut();
    // Record the dream area as it plays (zoom in, then back out) and spit
    // it out as an MP4, at the chosen export size (up to 4K). The frames are
    // drawn to a dedicated export canvas at that size (the card itself is
    // small, e.g. Turbo 192, so recording it 1:1 makes a tiny blurry file)
    // and to the viewfinder too, so you watch it record. One frame per draw,
    // via requestFrame, so the video is the right length and smooth — never
    // a live capture, which records duplicates while frames are drawn slowly
    // and comes out glitchy.
    const exLong = EXPORT_SIZE[settings.exportSize] || 1920;
    // Video stays at 4K max: a true 8K video would need the tiled dream on
    // every frame of the card. 8K is for stills (exportStill dreams it).
    const exLongCapped = Math.min(exLong, 3840);
    const exScale = Math.max(1, exLongCapped / Math.max(card.w, card.h));
    const exCv = document.createElement("canvas");
    exCv.width = Math.round(card.w * exScale); exCv.height = Math.round(card.h * exScale);
    const exCtx = exCv.getContext("2d"); exCtx.imageSmoothingEnabled = true; exCtx.imageSmoothingQuality = "high";
    const probeStream = exCv.captureStream(0);
    const vtrack = probeStream.getVideoTracks()[0];
    const manualFrames = !!(vtrack && typeof vtrack.requestFrame === "function");
    const stream2 = manualFrames ? probeStream : exCv.captureStream(30);
    const bitrate = EXPORT_BITRATE[settings.exportSize === "8k" ? "4k" : settings.exportSize] || 12000000;
    const rec = new MediaRecorder(stream2, mime ? { mimeType: mime, videoBitsPerSecond: bitrate } : { videoBitsPerSecond: bitrate });
    const chunks = []; rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    const done = new Promise(r => rec.onstop = r);
    rec.start(200);
    for (let i = 0; i < L; i++) {
      const fi = seqFrame(i), leg = seqLeg(i), fps = leg ? fpsOut : fpsIn;
      const im = await decodeBlob(card.frames[fi]);
      if (im) {
        const c = artCv.getContext("2d"); c.clearRect(0, 0, card.w, card.h); c.drawImage(im, 0, 0, card.w, card.h);
        exCtx.clearRect(0, 0, exCv.width, exCv.height); exCtx.drawImage(im, 0, 0, exCv.width, exCv.height);
        if (im.close) im.close();
      }
      setStatus("Recording video… " + (i + 1) + " / " + L); setBar((i + 1) / L);
      if (manualFrames) { vtrack.requestFrame(); await sleep(60); }
      else await sleep(1000 / fps);
    }
    await sleep(300); rec.stop(); await done;
    const blob = new Blob(chunks, { type: mime || "video/webm" });
    const ext = (mime || "").includes("mp4") ? "mp4" : "webm";
    downloadBlob(blob, "dreamcamera-" + String(card.no).padStart(4, "0") + "." + ext);
    setStatus("Video saved (" + (blob.size / 1048576).toFixed(1) + " MB" + (exLong > 3840 ? ", 4K — 8K is stills only" : "") + "). Also keep it in the gallery to watch later.");
  } catch (e) { console.error(e); setStatus("Video export failed: " + (e && e.message || e), true); }
  setBar(null); busy = false; startPlaying(); updateUI();
}
async function exportStill() {
  if (!card || !card.frames.length) return;
  const im = await decodeBlob(card.frames[card.frames.length - 1]);
  const exLong = EXPORT_SIZE[settings.exportSize] || 1920;
  const exScale = Math.max(1, exLong / Math.max(card.w, card.h));
  const c = document.createElement("canvas"); c.width = Math.round(card.w * exScale); c.height = Math.round(card.h * exScale);
  const ctx = c.getContext("2d"); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
  ctx.drawImage(im, 0, 0, c.width, c.height); if (im && im.close) im.close();
  // A still saved bigger than the dream is not just stretched: the extra
  // pixels are dreamed for real, tile by tile, at the export size. That
  // is what makes a 4K/8K still an actual 4K/8K deepdream.
  const trueHiRes = exLong > Math.max(card.w, card.h);
  if (trueHiRes) {
    if (!await ensureReady()) { setStatus("Could not load the model for the high-res still.", true); return; }
    await dreamTiled(c, settings.lr, 2, (d, t) => { setStatus("Dreaming the still at " + c.width + "×" + c.height + "… tile " + d + "/" + t); setBar(d / t); });
    setBar(null);
  }
  const b = await toBlob(c, "image/png", 1);
  downloadBlob(b, "dreamcamera-" + String(card.no).padStart(4, "0") + ".png");
  setStatus("Still saved (" + c.width + "×" + c.height + (trueHiRes ? ", dreamed at full size" : "") + ").");
}
function downloadBlob(blob, name) {
  // In the app the WebView ignores the download attribute, so hand the
  // file to the app to save into Downloads. In a browser, download it.
  if (window.DreamCameraBridge && window.DreamCameraBridge.saveFile) {
    const r = new FileReader();
    r.onload = () => { try { window.DreamCameraBridge.saveFile(r.result, name, blob.type || "application/octet-stream"); } catch (e) { downloadBlobWeb(blob, name); } };
    r.onerror = () => downloadBlobWeb(blob, name);
    r.readAsDataURL(blob);
    return;
  }
  downloadBlobWeb(blob, name);
}
function downloadBlobWeb(blob, name) {
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
  const rec = { id: card.id, created: card.created, no: card.no, name: card.name, model: card.model, preset: card.preset, w: card.w, h: card.h, frames: card.frames.slice(), thumb: thumb, zoomTotal: card.zoomTotal, viewDepth: card.viewDepth, settings: card.settings };
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
      card = { id: r.id, no: r.no, name: r.name, model: r.model, preset: r.preset, w: r.w, h: r.h, created: r.created, frames: r.frames.slice(), pathFrame: r.frames.length - 1, zoomTotal: (r.zoomTotal != null ? r.zoomTotal : ZOOM_CAP), viewDepth: r.viewDepth, settings: r.settings || settings };
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
  $("path").value = settings.path; $("quality").value = settings.quality; $("ratio").value = settings.ratio || "1:1";
  $("exportSize").value = settings.exportSize || "1080p";
  $("boomerang").value = settings.boomerang ? "1" : "0";
  $("facing").value = settings.facing; $("modelSrc").value = settings.modelSrc;
  applyRatio();
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
  $("exportSize").value = settings.exportSize || "1080p";
  $("exportSize").addEventListener("change", () => { settings.exportSize = $("exportSize").value; saveSettings(); });
  $("ratio").value = settings.ratio || "1:1";
  $("ratio").addEventListener("change", () => { settings.ratio = $("ratio").value; saveSettings(); applyRatio(); });
  window.addEventListener("resize", applyRatio);
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
      card = { id: "dc" + Date.now().toString(36), no: nextNumber(), name: f.name.replace(/\.[^.]+$/, "") || "Upload", model: MODEL.short, preset: "", w: q.w, h: q.h, created: Date.now(), frames: [], pathFrame: 0, zoomTotal: 1, settings: Object.assign({}, settings) };
      const c = document.createElement("canvas"); c.width = q.w; c.height = q.h;
      const target = q.w / q.h;
      let sw = im.naturalWidth, sh = im.naturalHeight;   // crop the photo to the ratio, centred
      if (sw / sh > target) sw = sh * target; else sh = sw / target;
      c.getContext("2d").drawImage(im, (im.naturalWidth - sw) / 2, (im.naturalHeight - sh) / 2, sw, sh, 0, 0, q.w, q.h);
      seedCv.width = q.w; seedCv.height = q.h; seedCv.getContext("2d").drawImage(c, 0, 0);
      artCv.width = q.w; artCv.height = q.h; artCv.getContext("2d").drawImage(c, 0, 0);
      $("viewBox").classList.add("dreaming"); $("viewHint").textContent = "";
      URL.revokeObjectURL(url);
      await startDream(); if (card && card.frames.length) await zoomRun(); updateUI();
    };
    im.src = url;
  });
}

// ===== VIDEO TAB =====
// Deepdream a whole video (up to 15s). Three modes:
//  Frame Dream — dream every frame, stitch it back together.
//  Photo Transfer — a photo becomes the dream overlay on the video.
//  Dreamweave — the style photo is dreamed live while the video plays,
//  so the style itself comes alive, and each frame is woven out of the
//  one before it (carryover), so it flows instead of flickering.
let videoSrc = null, videoMeta = null, styleImg = null;
let videoRunning = false, videoOutBlob = null, videoOutUrl = null;
function setVideoStatus(t, err) { const el = $("videoStatus"); el.textContent = t; el.style.color = err ? "#ff7b7b" : "#7df9ff"; }
function setVideoBar(p) { const b = $("videoBar"); if (p == null) { b.style.visibility = "hidden"; return; } b.style.visibility = "visible"; b.firstChild.style.width = Math.round(Math.max(0, Math.min(1, p)) * 100) + "%"; }
function videoDims(vw, vh) {
  const short = parseInt(($("videoSize") && $("videoSize").value) || "512", 10) || 512;   // the video dream size — separate from the Camera quality, so a video can be dreamed big while the Camera stays fast
  if (vw >= vh) return { w: Math.round(short * vw / vh), h: short };
  return { w: short, h: Math.round(short * vh / vw) };
}
function detailDims(vw, vh) {
  const long = 3840;   // 4K detail pass: long side 3840, aspect preserved
  if (vw >= vh) return { w: long, h: Math.round(long * vh / vw) };
  return { w: Math.round(long * vw / vh), h: long };
}
function switchTab(name) {
  const cam = name === "camera";
  $("cameraTab").hidden = !cam;
  $("videoTab").hidden = cam;
  $("tabCamera").classList.toggle("on", cam);
  $("tabVideo").classList.toggle("on", !cam);
  if (cam) applyRatio();
}
// Video presets: one tap sets mode, steps, mix, weave and zoom together.
const VIDEO_PRESETS = {
  gentle: { mode: "frame", steps: 1, mix: 0, weave: 0.15, zoom: false, size: 256 },
  dreamweave: { mode: "dreamweave", steps: 3, mix: 0.7, weave: 0.5, zoom: true, styleZoom: 0.97, styleRate: 3, styleSteps: 3, size: 512 },
  inception: { mode: "dreamweave", steps: 5, mix: 0.8, weave: 0.6, zoom: true, styleZoom: 0.96, styleRate: 2, styleSteps: 4, size: 768 },
  transfer: { mode: "transfer", steps: 2, mix: 0.6, weave: 0.15, zoom: false, size: 512 },
  slowburn: { mode: "frame", steps: 3, mix: 0, weave: 0.2, zoom: true, size: 768 },
  weavelight: { mode: "dreamweave", steps: 1, mix: 0.3, weave: 0.2, zoom: false, styleZoom: 0.99, styleRate: 6, styleSteps: 1, size: 384 },
  transferdeep: { mode: "transfer", steps: 4, mix: 0.8, weave: 0.15, zoom: true, size: 768 },
  hyperweave: { mode: "dreamweave", steps: 5, mix: 0.9, weave: 0.7, zoom: true, styleZoom: 0.95, styleRate: 2, styleSteps: 4, size: 1024 },
  acidweave: { mode: "dreamweave", steps: 5, mix: 0.95, weave: 0.75, zoom: true, styleZoom: 0.95, styleRate: 1, styleSteps: 3, size: 1024 },
  deepvortex: { mode: "dreamweave", steps: 5, mix: 0.85, weave: 0.75, zoom: true, styleZoom: 0.96, styleRate: 1, styleSteps: 3, size: 1024 },
  meltdown: { mode: "dreamweave", steps: 5, mix: 0.95, weave: 0.45, zoom: true, styleZoom: 0.97, styleRate: 2, styleSteps: 5, size: 1024 },
  fractaldive: { mode: "dreamweave", steps: 4, mix: 0.75, weave: 0.6, zoom: true, styleZoom: 0.94, styleRate: 3, styleSteps: 3, size: 768 }
};
function applyVideoPreset(name) {
  const p = VIDEO_PRESETS[name]; if (!p) return;
  $("videoMode").value = p.mode;
  $("videoSteps").value = p.steps; $("videoStepsVal").textContent = p.steps;
  $("videoMix").value = p.mix; $("videoMixVal").textContent = p.mix.toFixed(2);
  $("videoWeave").value = p.weave; $("videoWeaveVal").textContent = p.weave.toFixed(2);
  $("videoZoom").value = p.zoom ? "1" : "0";
  if (p.size != null) { $("videoSize").value = String(p.size); }
  if (p.styleZoom != null) { $("videoStyleZoom").value = p.styleZoom; $("videoStyleZoomVal").textContent = p.styleZoom.toFixed(3); }
  if (p.styleRate != null) { $("videoStyleRate").value = String(p.styleRate); }
  if (p.styleSteps != null) { $("videoStyleSteps").value = p.styleSteps; $("videoStyleStepsVal").textContent = p.styleSteps; }
  videoModeNote();
}
function videoModeNote() {
  const m = $("videoMode").value;
  const notes = {
    frame: "Every frame is dreamed on its own, then stitched back into a video.",
    transfer: "Your photo is dreamed once, then laid over every frame like a living texture.",
    dreamweave: "The style photo is dreamed live as the video plays, so the style itself grows — and each frame is woven out of the last, so it flows without flicker."
  };
  $("videoModeNote").textContent = notes[m] || "";
  const needStyle = m !== "frame";
  $("styleRow").style.display = needStyle ? "" : "none";
  $("mixRow").style.display = needStyle ? "" : "none";
  $("weaveRow").style.display = m === "dreamweave" ? "" : "none";
  $("styleZoomRow").style.display = m === "dreamweave" ? "" : "none";
  $("styleRateRow").style.display = m === "dreamweave" ? "" : "none";
  $("styleStepsRow").style.display = m === "dreamweave" ? "" : "none";
}
function updateVideoUI() {
  $("videoDreamBtn").disabled = videoRunning || !videoMeta;
  $("videoStopBtn").disabled = !videoRunning;
  $("videoSaveBtn").disabled = !videoOutBlob || videoRunning;
}
// Dream the style photo into a texture tensor at (w,h). The aspect is
// preserved — dreamSteps letterboxes a fixed-size model, never squeezes.
async function makeStyleTexture(w, h) {
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  const target = w / h;
  let sw = styleImg.naturalWidth, sh = styleImg.naturalHeight;
  if (sw / sh > target) sw = sh * target; else sh = sw / target;
  c.getContext("2d").drawImage(styleImg, (styleImg.naturalWidth - sw) / 2, (styleImg.naturalHeight - sh) / 2, sw, sh, 0, 0, w, h);
  let t = canvasToTensor(c);
  t = dreamSteps(t, settings.lr, 8, false);
  return t;
}
// Seek with a timeout: a seek that never lands (thin buffer, odd codec,
// WebView quirk) used to hang the whole run with no message at all.
function seekTo(v, t) {
  return new Promise(res => {
    let done = false;
    const finish = ok => { if (!done) { done = true; res(ok); } };
    v.onseeked = () => finish(true);
    v.onerror = () => finish(false);
    try { v.currentTime = t; } catch (e) { finish(false); return; }
    setTimeout(() => finish(false), 8000);
  });
}
async function processVideo() {
  if (videoRunning) return;
  if (!videoMeta) { setVideoStatus("Pick a video first — it takes a second to read.", true); return; }
  const mode = $("videoMode").value;
  if (mode !== "frame" && !styleImg) { setVideoStatus("Pick a style photo first.", true); return; }
  if (!await ensureReady()) return;
  videoRunning = true; updateVideoUI();
  const src = document.createElement("video");
  src.muted = true; src.playsInline = true; src.preload = "auto"; src.src = videoSrc;
  try { await new Promise((res, rej) => { src.onloadedmetadata = res; src.onerror = () => rej(new Error("could not read that video")); }); }
  catch (e) { setVideoStatus("Could not read that video.", true); videoRunning = false; updateVideoUI(); return; }
  // Prime the video so seeks have data to land on: wait for the first
  // frame to be readable (bounded — a video that never primes is reported,
  // not hung on). Without this the first seek could stall forever.
  await new Promise(res => {
    let done = false;
    const finish = () => { if (!done) { done = true; res(); } };
    src.oncanplay = finish; src.onloadeddata = finish; src.onerror = finish;
    try { src.load(); } catch (e) { finish(); }
    setTimeout(finish, 8000);
  });
  const dur = Math.min(15, src.duration || 0);
  const fps = parseInt($("videoFps").value, 10) || 12;
  const total = Math.max(1, Math.floor(dur * fps));
  const vd = videoDims(src.videoWidth, src.videoHeight);
  // 4K detail pass: each dreamed frame is upscaled to 3840 on the long
  // side and dreamed again at that size, tile by tile — real 4K detail,
  // not an upscale. It is slow (~20-30s a frame), so it is opt-in.
  const detailOn = $("videoDetail") && $("videoDetail").value === "4k";
  const hiD = detailOn ? detailDims(src.videoWidth, src.videoHeight) : null;
  const inCv = document.createElement("canvas"); inCv.width = vd.w; inCv.height = vd.h;
  const outCv = document.createElement("canvas"); outCv.width = vd.w; outCv.height = vd.h;
  const hiCv = detailOn ? document.createElement("canvas") : null;
  if (hiCv) { hiCv.width = hiD.w; hiCv.height = hiD.h; }
  const dispCv = detailOn ? hiCv : outCv;   // what the recorder + preview show
  const tileScratch = detailOn ? {} : null;   // reused across frames, so the 4K canvases are not re-allocated every frame
  let styleT = null;
  if (mode !== "frame") { setVideoStatus("Dreaming the style photo…"); styleT = await makeStyleTexture(vd.w, vd.h); }
  // Record one frame per dreamed frame, not in real time: each frame takes
  // seconds to dream, so a live captureStream(fps) records dozens of
  // duplicate frames per dreamed frame and the saved video comes out
  // minutes long, choppy and glitchy. captureStream(0) + requestFrame
  // captures exactly one frame per draw, so the video is the right length
  // and smooth. (Falls back to the live stream if requestFrame is missing.)
  const probeStream = dispCv.captureStream(0);
  const vtrack = probeStream.getVideoTracks()[0];
  const manualFrames = !!(vtrack && typeof vtrack.requestFrame === "function");
  const stream = manualFrames ? probeStream : dispCv.captureStream(fps);
  const mime = videoMime();
  const vbitrate = detailOn ? 30000000 : 8000000;   // 4K needs a real bitrate or it comes out blocky
  // The recorder must not wedge the run: if it cannot start, say so and
  // give the button back instead of leaving Dream disabled forever.
  let rec, done;
  const chunks = [];
  try {
    rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: vbitrate } : { videoBitsPerSecond: vbitrate });
    rec.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
    done = new Promise(r => rec.onstop = r);
    rec.start(200);
  } catch (e) {
    setVideoStatus("Could not start the video recorder on this device: " + (e && e.message || e), true);
    videoRunning = false; updateVideoUI(); return;
  }
  // Live preview: show each dreamed frame in the square as it is made, so
  // you watch the video build smoothly while it records. The finished
  // video takes over the square at the end.
  const pv = $("vidPrev"); pv.width = vd.w; pv.height = vd.h; pv.style.display = "block";
  $("vidOut").style.display = "none";
  $("videoHint").textContent = "Dreaming… watch it build here.";
  let prev = null;
  try {
    for (let i = 0; i < total && videoRunning; i++) {
      const seeked = await seekTo(src, Math.min(i / fps, Math.max(0, dur - 0.001)));
      if (!seeked) { setVideoStatus("That video stopped seeking at frame " + (i + 1) + " — try a different file.", true); break; }
      inCv.getContext("2d").drawImage(src, 0, 0, vd.w, vd.h);
      let cur = canvasToTensor(inCv);
      if (mode === "dreamweave" && prev) {
        const weave = parseFloat($("videoWeave").value) || 0.3;
        const w = tf.tidy(() => cur.mul(1 - weave).add(prev.mul(weave)));   // weave: carry the last frame forward
        cur.dispose(); cur = w;
      }
      if (mode !== "frame" && styleT) {
        // The style knobs are read live, so you can turn the zoom or the
        // morph speed up and down while the video is dreaming.
        const styleRate = parseInt($("videoStyleRate").value, 10) || 3;
        if (mode === "dreamweave" && i % styleRate === 0) {   // the style itself dreams — it visibly morphs and dives into the photo as the video plays
          // The style zooms in and dreams along with the video — extra
          // inception: the overlay is a living dream, not a static picture.
          const sz = parseFloat($("videoStyleZoom").value) || 0.97;
          const ss = parseInt($("videoStyleSteps").value, 10) || 3;
          const zs = zoomCrop(styleT, sz, 0, 0, 0); styleT.dispose(); styleT = zs;
          styleT = dreamSteps(styleT, settings.lr, ss, false);
        }
        const mix = parseFloat($("videoMix").value) || 0.5;
        const b = tf.tidy(() => cur.mul(1 - mix).add(styleT.mul(mix)));
        cur.dispose(); cur = b;
      }
      if ($("videoZoom").value === "1") {   // zoom in a touch every frame, like the still dream
        const z = zoomCrop(cur, 0.998, 0, 0, 0); cur.dispose(); cur = z;
      }
      const steps = parseInt($("videoSteps").value, 10) || 2;
      cur = dreamSteps(cur, settings.lr, steps, true);
      await tensorToCanvas(cur, outCv);
      if (detailOn) {
        // Upscale to 4K, then dream the 4K frame for real, tile by tile.
        const hx = hiCv.getContext("2d");
        hx.clearRect(0, 0, hiD.w, hiD.h);
        hx.drawImage(outCv, 0, 0, hiD.w, hiD.h);
        await dreamTiled(hiCv, settings.lr, 2, (d, t) => {
          setVideoStatus("Dreaming video… frame " + (i + 1) + " of " + total + " · 4K detail tile " + d + "/" + t);
        }, tileScratch);
        pv.getContext("2d").drawImage(hiCv, 0, 0, pv.width, pv.height);
      } else {
        pv.getContext("2d").drawImage(outCv, 0, 0);   // live preview: the new frame, in order
      }
      if (prev) prev.dispose();
      prev = cur;
      setVideoStatus("Dreaming video… frame " + (i + 1) + " of " + total);
      setVideoBar((i + 1) / total);
      if (manualFrames) { vtrack.requestFrame(); await sleep(60); }   // let the recorder grab this frame before the next draw
      else await sleep(1000 / fps);
      await tf.nextFrame();
    }
  } catch (e) {
    console.error(e);
    setVideoStatus("Video dream failed: " + (e && e.message || e), true);
  }
  try { rec.stop(); await done; } catch (e) { console.error(e); }
  if (prev) { prev.dispose(); prev = null; }
  if (styleT) { styleT.dispose(); styleT = null; }
  videoRunning = false;
  videoOutBlob = new Blob(chunks, { type: mime || "video/webm" });
  pv.style.display = "none"; $("vidOut").style.display = "";
  if (videoOutBlob.size) {
    if (videoOutUrl) URL.revokeObjectURL(videoOutUrl);
    videoOutUrl = URL.createObjectURL(videoOutBlob);
    $("vidOut").src = videoOutUrl;
    $("videoHint").textContent = "";
    setVideoStatus("Done — " + total + " frames · " + (detailOn ? hiD.w + "×" + hiD.h + " true 4K" : vd.w + "×" + vd.h) + ". Watch it above, or save it.");
  } else {
    videoOutBlob = null;
    $("videoHint").textContent = "Dreamed video plays here";
  }
  setVideoBar(null);
  updateVideoUI();
}
function initVideoTab() {
  $("tabCamera").onclick = () => switchTab("camera");
  $("tabVideo").onclick = () => switchTab("video");
  $("videoMode").addEventListener("change", videoModeNote);
  videoModeNote();
  $("videoPreset").addEventListener("change", e => applyVideoPreset(e.target.value));
  $("videoSteps").addEventListener("input", () => { $("videoStepsVal").textContent = $("videoSteps").value; });
  $("videoMix").addEventListener("input", () => { $("videoMixVal").textContent = (+$("videoMix").value).toFixed(2); });
  $("videoWeave").addEventListener("input", () => { $("videoWeaveVal").textContent = (+$("videoWeave").value).toFixed(2); });
  $("videoStyleZoom").addEventListener("input", () => { $("videoStyleZoomVal").textContent = (+$("videoStyleZoom").value).toFixed(3); });
  $("videoStyleSteps").addEventListener("input", () => { $("videoStyleStepsVal").textContent = $("videoStyleSteps").value; });
  $("videoPickBtn").onclick = () => $("videoIn").click();
  $("videoIn").addEventListener("change", () => {
    const f = $("videoIn").files[0]; if (!f) return;
    if (videoSrc) URL.revokeObjectURL(videoSrc);
    videoSrc = URL.createObjectURL(f);
    const v = document.createElement("video");
    v.preload = "metadata"; v.src = videoSrc;
    v.onloadedmetadata = () => {
      const d = v.duration || 0;
      videoMeta = { duration: d, width: v.videoWidth, height: v.videoHeight };
      $("videoInfo").textContent = f.name + " · " + d.toFixed(1) + "s" + (d > 15 ? " (first 15s used)" : "") + " · " + v.videoWidth + "×" + v.videoHeight;
      setVideoStatus(d > 15 ? "That video is over 15s — only the first 15s will be dreamed." : "Video ready. Pick a mode and dream it.");
      updateVideoUI();
    };
    v.onerror = () => { setVideoStatus("Could not read that video.", true); videoMeta = null; updateVideoUI(); };
  });
  $("stylePickBtn").onclick = () => $("styleIn").click();
  $("styleIn").addEventListener("change", () => {
    const f = $("styleIn").files[0]; if (!f) return;
    const url = URL.createObjectURL(f), im = new Image();
    im.onload = () => { styleImg = im; $("styleInfo").textContent = f.name + " · " + im.naturalWidth + "×" + im.naturalHeight; URL.revokeObjectURL(url); };
    im.onerror = () => { setVideoStatus("Could not read that photo.", true); };
    im.src = url;
  });
  $("videoDreamBtn").onclick = processVideo;
  $("videoStopBtn").onclick = () => { videoRunning = false; setVideoStatus("Stopping…"); };
  $("videoSaveBtn").onclick = () => {
    if (!videoOutBlob) return;
    const ext = (videoOutBlob.type || "").includes("mp4") ? "mp4" : "webm";
    downloadBlob(videoOutBlob, "dreamcamera-video-" + Date.now().toString(36) + "." + ext);
    setVideoStatus("Video saved.");
  };
  updateVideoUI();
}

// ===== BOOT =====
buildModelSelect();
buildLayers();
buildPresets();
if (settings.preset != null && PRESETS[settings.preset] && PRESETS[settings.preset].m === MODEL.short) applyPreset(settings.preset);
syncControls();
initControls();
initVideoTab();
updateUI();
renderGallery();
startCamera();
