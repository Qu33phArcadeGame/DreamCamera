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
const QUALITY = { turbo: { w: 192, h: 192 }, fast: { w: 256, h: 256 }, normal: { w: 384, h: 384 }, high: { w: 512, h: 512 }, ultra: { w: 768, h: 768 }, max: { w: 1024, h: 1024 } };
// No 2K (1440) live quality: the Camera dreams the whole image at once,
// and at 1440 the model's activations run the phone out of memory and the
// app crashes. Max 1024 is the live ceiling; true 4K/8K comes from the
// Save still, which dreams tile by tile (bounded memory).
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
const DEFAULTS = { model: "v1", preset: 0, path: "random", frames: 50, zoomFactor: 0.97, steps: 4, lr: 0.04, range: 0.08, segFrames: 10, sharpen: 0.35, layers: ["mixed4c", "mixed4d"], quality: "turbo", ratio: "1:1", playFps: 12, playFpsOut: 12, boomerang: true, octaves: 3, jitter: 24, facing: "environment", modelSrc: "live", exportSize: "1080p", hueSpin: 0, saturation: 1, pulse: 0, glow: 0, soft: 0, vignette: 0, fade: 0, rgbSplit: 0, staticAmt: 0, invertAmt: 0, kaleido: 0, swirl: 0, echo: 0, scanlines: 0 };

// ===== STATE =====
const vid = $("vid"), artCv = $("artCv");
const seedCv = document.createElement("canvas"), tmpCv = document.createElement("canvas");
let seedCardId = null;   // id of the card seedCv was drawn from; null when it belongs to no current card (e.g. a gallery-loaded card, whose seed was never saved)
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

function loadSettings() { let s = {}; try { s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}") || {}; } catch (e) {} if (s.quality === "twoK") s.quality = "max";   // 2K live quality was removed (it crashed the live dream) — fall back to Max
  if (s.path === "drawn") s.path = "joystick"; delete s.drawnPath;   // Drawn path was replaced by the joystick
  return Object.assign({}, DEFAULTS, s); }
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
  try { if (model) model.dispose(); } catch (e) {} model = null; midModel = null; grads = {}; try { if (cur) cur.dispose(); } catch (e) {} cur = null;   // disposals must not throw: on a lost device, disposing the old model must not block switching to a fallback engine
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
function tileSizeForModel() {
  // A fixed-size model dreams at its native size — a 512 tile would be
  // downscaled to 224 and back, softening the detail. So tile at the
  // model's own size instead (224 for zentangledream / MobileNet / ResNet
  // / DenseNet, 299 for InceptionResNetV2). Free-size models use 512.
  return MODEL.fixedSize || TILE_SIZE;
}
function tileOverlapForModel(ts) { return ts === TILE_SIZE ? TILE_OVERLAP : 64; }
function getTileMask(oLeft, oRight, oTop, oBottom, ts) {
  // Feather weight for a tile: 1 in the middle, ramping to 0 over the
  // overlap at each edge. Each overlap is the tile's ACTUAL overlap with
  // its neighbour on that side (even spacing makes them nearly equal, but
  // rounding makes them differ by a pixel or two — the ramp must span
  // that side's overlap exactly, or the blend sums to more than 1 in the
  // middle). A border edge (null) stays 1, so the image edge is not
  // dimmed. The weight is the product of the horizontal and vertical
  // ramps, so neighbouring tiles' weights sum to exactly 1 in every
  // overlap, corners included.
  const key = [oLeft, oRight, oTop, oBottom, ts].join(",");
  if (getTileMask.cache[key]) return getTileMask.cache[key];
  const c = document.createElement("canvas"); c.width = ts; c.height = ts;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(ts, ts);
  const d = img.data;
  for (let y = 0; y < ts; y++) {
    // Each ramp is clamped to 1 on its own: a pixel far from both edges
    // of an axis has weight 1 on that axis, not the raw (large) ratio —
    // without the clamp the product overshoots and the blend sums to ~2.
    const wy = Math.max(0, Math.min(1, Math.min(oTop == null ? 1 : (y + 0.5) / oTop, oBottom == null ? 1 : (ts - y - 0.5) / oBottom)));
    for (let x = 0; x < ts; x++) {
      const wx = Math.max(0, Math.min(1, Math.min(oLeft == null ? 1 : (x + 0.5) / oLeft, oRight == null ? 1 : (ts - x - 0.5) / oRight)));
      const w = wx * wy;
      const i = (y * ts + x) * 4;
      d[i] = 255; d[i + 1] = 255; d[i + 2] = 255; d[i + 3] = Math.round(w * 255);
    }
  }
  ctx.putImageData(img, 0, 0);
  getTileMask.cache[key] = c;
  return c;
}
getTileMask.cache = {};
function tileAxis(W, ts, ov) {
  // Evenly spaced tile positions covering [0, W - ts]. The stride is kept
  // between half the tile and (tile - 32) (overlap 32 to half the tile),
  // so no tile is ever in two overlaps at once. Target stride is ts - ov.
  const Q = W - ts;
  if (Q <= 0) return [0];
  const mLo = Math.ceil(Q / (ts - 32)), mHi = Math.floor(Q / Math.ceil(ts / 2));
  let m = Math.round(Q / (ts - ov));
  if (mLo <= mHi) m = Math.max(mLo, Math.min(mHi, m));
  if (m < 1) m = 1;
  const pos = [];
  for (let i = 0; i <= m; i++) pos.push(Math.round(i * Q / m));
  return pos;
}
async function dreamTiled(cv, lr, steps, onTile, scratch, tsOpt, ovOpt) {
  // cv: a 2D canvas at the target resolution. Dreamed in place, tile by
  // tile. onTile(done, total) reports progress. scratch (optional) holds
  // the accumulation canvases so a video can reuse them across frames
  // instead of allocating a 4K canvas per frame. tsOpt/ovOpt (optional)
  // override the tile size/overlap — the video path passes 768/192; a
  // still passes nothing and uses the model's own size (512 free-size).
  // Returns the tile grid {nx, ny} (1x1 when dreamed whole).
  const W = cv.width, H = cv.height;
  if (W <= 0 || H <= 0) return null;
  // The tile size follows the model: a fixed-size model (zentangledream,
  // MobileNet, ResNet, DenseNet, InceptionResNetV2) tiles at its native
  // size so nothing is downscaled and back; free-size models use 512 —
  // unless the caller passes a size (video: 768, the largest size
  // proven on the phone).
  const ts = tsOpt || tileSizeForModel(), ov = ovOpt || tileOverlapForModel(ts);
  // A small image is just dreamed whole — no tiling needed.
  if (W <= ts && H <= ts) {
    let t = canvasToTensor(cv);
    t = dreamSteps(t, lr, steps, true);
    await tensorToCanvas(t, cv);
    t.dispose();
    if (onTile) onTile(1, 1);
    return { nx: 1, ny: 1 };
  }
  const scratchPassed = scratch != null;   // a video passes its own scratch in and reuses it across frames; a still makes its own and frees it after
  scratch = scratch || {};
  if (!scratch.resCv || scratch.resCv.width !== W || scratch.resCv.height !== H) {
    scratch.resCv = document.createElement("canvas"); scratch.resCv.width = W; scratch.resCv.height = H;
    scratch.tileCv = document.createElement("canvas"); scratch.tileCv.width = ts; scratch.tileCv.height = ts;
    scratch.tmpCv = document.createElement("canvas"); scratch.tmpCv.width = ts; scratch.tmpCv.height = ts;
  }
  const resCtx = scratch.resCv.getContext("2d");
  const tileCtx = scratch.tileCv.getContext("2d"), tmpCtx = scratch.tmpCv.getContext("2d");
  resCtx.clearRect(0, 0, W, H);
  // Tile grid: evenly spaced positions, so every overlap is about the
  // same width. A stride grid with a clamped last tile makes one overlap
  // up to 400px wide, and then a tile sits in two overlaps at once and
  // the feather weights sum to ~1.1 instead of 1 (a bright band). Even
  // spacing keeps every overlap between 32px and half the tile, so no tile
  // is ever in two overlaps at once and the weights always sum to 1.
  const xs = tileAxis(W, ts, ov), ys = tileAxis(H, ts, ov);
  const total = xs.length * ys.length;
  let done = 0;
  for (let yi = 0; yi < ys.length; yi++) for (let xi = 0; xi < xs.length; xi++) {
    const tx = xs[xi], ty = ys[yi];
    // Each edge's ramp spans that side's actual overlap with its
    // neighbour (a border edge has none). The overlaps come from the grid
    // positions, so the two tiles sharing an overlap always agree on it.
    const oLeft = xi === 0 ? null : (xs[xi - 1] + ts) - tx;
    const oRight = xi === xs.length - 1 ? null : (tx + ts) - xs[xi + 1];
    const oTop = yi === 0 ? null : (ys[yi - 1] + ts) - ty;
    const oBottom = yi === ys.length - 1 ? null : (ty + ts) - ys[yi + 1];
    // Extract + dream the tile.
    tileCtx.clearRect(0, 0, ts, ts);
    tileCtx.drawImage(cv, tx, ty, ts, ts, 0, 0, ts, ts);
    let t = canvasToTensor(scratch.tileCv);
    t = dreamSteps(t, lr, steps, true);
    await tensorToCanvas(t, scratch.tileCv);
    t.dispose();
    // Feather mask for this tile (cached by its four overlaps + size).
    const mask = getTileMask(oLeft, oRight, oTop, oBottom, ts);
    // tmp = dreamed tile with alpha = weight.
    tmpCtx.clearRect(0, 0, ts, ts);
    tmpCtx.drawImage(scratch.tileCv, 0, 0);
    tmpCtx.globalCompositeOperation = "destination-in";
    tmpCtx.drawImage(mask, 0, 0);
    tmpCtx.globalCompositeOperation = "source-over";
    // Accumulate: result += tile*weight. 'lighter' adds source*alpha,
    // and the weights sum to 1, so nothing clips and the accumulation IS
    // the final image. There is no weight canvas and no normalize pass:
    // the old normalize held three 8K ImageData objects (~400MB) at once,
    // which is what crashed the app on 8K.
    resCtx.globalCompositeOperation = "lighter";
    resCtx.drawImage(scratch.tmpCv, tx, ty);
    resCtx.globalCompositeOperation = "source-over";
    done++;
    if (onTile) onTile(done, total);
    await tf.nextFrame();
  }
  // The accumulation is the final image (weights sum to 1) — just draw
  // it back onto the canvas. No ImageData pass: that was the 8K crash.
  const outCtx = cv.getContext("2d");
  outCtx.clearRect(0, 0, W, H);
  outCtx.drawImage(scratch.resCv, 0, 0);
  // Free the accumulation canvas right away when this call made it, so
  // the PNG encode that follows does not have to share memory with it.
  // (A video passes its own scratch in and reuses it, so leave that one.)
  if (!scratchPassed) { scratch.resCv.width = 0; scratch.resCv.height = 0; scratch.tileCv.width = 0; scratch.tileCv.height = 0; scratch.tmpCv.width = 0; scratch.tmpCv.height = 0; }
  return { nx: xs.length, ny: ys.length };
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
// ===== EFFECTS (dreamy + glitch) =====
// One pass that applies every enabled effect to a finished frame's canvas,
// in place. It bakes into the saved video/still. The filter effects (hue,
// saturation, invert, soft blur, fade) combine into ONE canvas filter +
// one draw, so they cost almost nothing. The composite effects (RGB split,
// glow, vignette, static, kaleidoscope, swirl, echo, scanlines) each take
// a few draws. If nothing is on, it returns immediately (no overhead). A
// missing canvas filter is a silent no-op, so this never crashes an old
// WebView.
const fxCv = document.createElement("canvas");   // temp 1
const fxCv2 = document.createElement("canvas");  // temp 2 (RGB split channels / kaleidoscope)
let noiseCv = null;   // pre-generated static noise, tiled
function getNoiseCv() {
  if (noiseCv) return noiseCv;
  noiseCv = document.createElement("canvas"); noiseCv.width = 512; noiseCv.height = 512;
  const ctx = noiseCv.getContext("2d");
  const img = ctx.createImageData(512, 512);
  for (let i = 0; i < img.data.length; i += 4) {
    const v = (Math.random() * 255) | 0;
    img.data[i] = v; img.data[i + 1] = v; img.data[i + 2] = v; img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return noiseCv;
}
let scanCv = null;   // pre-generated scanline pattern (1px dark line every 4px), tiled
function getScanCv() {
  if (scanCv) return scanCv;
  scanCv = document.createElement("canvas"); scanCv.width = 4; scanCv.height = 4;
  const ctx = scanCv.getContext("2d");
  ctx.clearRect(0, 0, 4, 4);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, 4, 1);
  return scanCv;
}
function applyFx(cv, fx) {
  if (!fx) return;
  const W = cv.width, H = cv.height;
  if (W <= 0 || H <= 0) return;
  const ctx = cv.getContext("2d");
  const on = (fx.hue || (fx.sat && fx.sat !== 1) || fx.invert || fx.soft || fx.fade || fx.glow || fx.vignette || fx.rgb || fx.static || fx.kaleido || fx.swirl || fx.echo || fx.scanlines);
  if (!on) return;
  // 1. Kaleidoscope: mirror one quadrant into all four (a symmetric mandala).
  // Blends with the original by the slider, so 0.5 is half mirrored.
  if (fx.kaleido > 0) {
    if (fxCv.width !== W || fxCv.height !== H) { fxCv.width = W; fxCv.height = H; }
    if (fxCv2.width !== W || fxCv2.height !== H) { fxCv2.width = W; fxCv2.height = H; }
    const fctx = fxCv.getContext("2d"), gctx = fxCv2.getContext("2d");
    fctx.clearRect(0, 0, W, H); fctx.drawImage(cv, 0, 0);   // fxCv = source
    const hw = W / 2, hh = H / 2;
    gctx.clearRect(0, 0, W, H);
    gctx.drawImage(fxCv, 0, 0, hw, hh, 0, 0, hw, hh);   // top-left, as-is
    gctx.save(); gctx.translate(W, 0); gctx.scale(-1, 1); gctx.drawImage(fxCv, 0, 0, hw, hh, 0, 0, hw, hh); gctx.restore();   // top-right, mirrored
    gctx.save(); gctx.translate(0, H); gctx.scale(1, -1); gctx.drawImage(fxCv, 0, 0, hw, hh, 0, 0, hw, hh); gctx.restore();   // bottom-left, mirrored
    gctx.save(); gctx.translate(W, H); gctx.scale(-1, -1); gctx.drawImage(fxCv, 0, 0, hw, hh, 0, 0, hw, hh); gctx.restore();   // bottom-right, mirrored both
    ctx.globalAlpha = fx.kaleido;
    ctx.drawImage(fxCv2, 0, 0);
    ctx.globalAlpha = 1;
  }
  // 2. Swirl: twist the image around the centre like a vortex. The outer
  // rings twist most, the centre least. The vortex lives in a disc
  // (0.75x the short side) so the corners keep the original image — a
  // full-canvas disc would need a huge zoom to cover a tall frame.
  if (fx.swirl > 0) {
    if (fxCv.width !== W || fxCv.height !== H) { fxCv.width = W; fxCv.height = H; }
    const fctx = fxCv.getContext("2d");
    fctx.clearRect(0, 0, W, H); fctx.drawImage(cv, 0, 0);   // fxCv = source (cv keeps the base at the edges)
    const cx = W / 2, cy = H / 2, maxR = Math.min(W, H) * 0.75;
    const rings = 6, S = 1 + 0.4 * fx.swirl;
    for (let i = 0; i < rings; i++) {
      const rOut = maxR * (1 - i / rings);
      const ang = fx.swirl * (1 - i / rings) * 0.6;   // outer rings twist most
      ctx.save();
      ctx.beginPath(); ctx.arc(cx, cy, rOut, 0, Math.PI * 2); ctx.clip();
      ctx.translate(cx, cy); ctx.rotate(ang); ctx.scale(S, S); ctx.translate(-cx, -cy);
      ctx.drawImage(fxCv, 0, 0);
      ctx.restore();
    }
  }
  // 3. Echo: the image trails into itself, a tunnel of echoes.
  if (fx.echo > 0) {
    if (fxCv.width !== W || fxCv.height !== H) { fxCv.width = W; fxCv.height = H; }
    const fctx = fxCv.getContext("2d");
    fctx.clearRect(0, 0, W, H); fctx.drawImage(cv, 0, 0);   // fxCv = source
    for (let i = 1; i <= 3; i++) {
      const s = 1 + i * 0.05 * fx.echo;
      const a = 0.22 * fx.echo / i;
      const dw = W * s, dh = H * s;
      ctx.globalAlpha = a;
      ctx.drawImage(fxCv, (W - dw) / 2, (H - dh) / 2, dw, dh);
    }
    ctx.globalAlpha = 1;
  }
  // 4. Filter effects in one pass: hue, saturation, invert, soft blur, fade.
  const parts = [];
  if (fx.hue) parts.push("hue-rotate(" + fx.hue + "deg)");
  if (fx.sat && fx.sat !== 1) parts.push("saturate(" + fx.sat + ")");
  if (fx.invert) parts.push("invert(" + fx.invert + ")");
  if (fx.soft) parts.push("blur(" + fx.soft + "px)");
  if (fx.fade) { parts.push("brightness(" + (1 + 0.12 * fx.fade) + ")"); parts.push("contrast(" + (1 - 0.35 * fx.fade) + ")"); }
  if (parts.length) {
    if (fxCv.width !== W || fxCv.height !== H) { fxCv.width = W; fxCv.height = H; }
    const fctx = fxCv.getContext("2d");
    fctx.filter = parts.join(" ");
    fctx.drawImage(cv, 0, 0);
    fctx.filter = "none";
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(fxCv, 0, 0);
  }
  // 5. RGB split: the red channel shifts one way, blue the other.
  if (fx.rgb > 0) {
    const k = Math.round(fx.rgb);
    if (fxCv.width !== W || fxCv.height !== H) { fxCv.width = W; fxCv.height = H; }
    if (fxCv2.width !== W || fxCv2.height !== H) { fxCv2.width = W; fxCv2.height = H; }
    const fctx = fxCv.getContext("2d"), gctx = fxCv2.getContext("2d");
    fctx.clearRect(0, 0, W, H); fctx.drawImage(cv, 0, 0);   // fxCv = source
    ctx.clearRect(0, 0, W, H);
    ctx.globalCompositeOperation = "lighter";
    const channels = [[255, 0, 0, k], [0, 255, 0, 0], [0, 0, 255, -k]];
    for (const ch of channels) {
      gctx.clearRect(0, 0, W, H);
      gctx.drawImage(fxCv, 0, 0);
      gctx.globalCompositeOperation = "multiply";   // isolate one channel
      gctx.fillStyle = "rgb(" + ch[0] + "," + ch[1] + "," + ch[2] + ")";
      gctx.fillRect(0, 0, W, H);
      gctx.globalCompositeOperation = "source-over";
      ctx.drawImage(fxCv2, ch[3], 0);
    }
    ctx.globalCompositeOperation = "source-over";
  }
  // 6. Glow: a blurred bright copy screened over the top (bloom / haze).
  if (fx.glow > 0) {
    if (fxCv.width !== W || fxCv.height !== H) { fxCv.width = W; fxCv.height = H; }
    const fctx = fxCv.getContext("2d");
    fctx.filter = "blur(" + Math.max(2, Math.round(Math.min(W, H) * 0.02)) + "px) brightness(1.3)";
    fctx.drawImage(cv, 0, 0);
    fctx.filter = "none";
    ctx.globalCompositeOperation = "screen";
    ctx.globalAlpha = fx.glow * 0.7;
    ctx.drawImage(fxCv, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }
  // 7. Vignette: darken the edges. Only ever drawn when the slider is
  // above 0 — at 0 it is skipped entirely, so it cannot appear on its own.
  if (fx.vignette > 0) {
    const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.35, W / 2, H / 2, Math.max(W, H) * 0.75);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(1, "rgba(0,0,0," + (fx.vignette * 0.8) + ")");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, W, H);
  }
  // 8. Scanlines: CRT horizontal lines (glitch).
  if (fx.scanlines > 0) {
    const sc = getScanCv();
    ctx.save();
    ctx.globalAlpha = fx.scanlines * 0.5;
    ctx.fillStyle = ctx.createPattern(sc, "repeat");
    ctx.fillRect(0, 0, W, H);
    ctx.restore();
  }
  // 9. Static: film grain (a tiled noise canvas at a random offset).
  if (fx.static > 0) {
    const nc = getNoiseCv();
    const ox = Math.floor(Math.random() * 512), oy = Math.floor(Math.random() * 512);
    ctx.save();
    ctx.globalAlpha = fx.static * 0.45;
    ctx.globalCompositeOperation = "overlay";
    for (let y = -oy; y < H; y += 512) for (let x = -ox; x < W; x += 512) ctx.drawImage(nc, x, y);
    ctx.restore();
  }
}
// Build the fx object for the Camera (hue is the cumulative Color spin).
function cameraFx(hue) {
  return {
    hue: hue || 0, sat: settings.saturation || 1,
    glow: settings.glow || 0, soft: settings.soft || 0,
    vignette: settings.vignette || 0, fade: settings.fade || 0,
    rgb: settings.rgbSplit || 0,
    static: settings.staticAmt || 0,
    invert: settings.invertAmt || 0,
    kaleido: settings.kaleido || 0, swirl: settings.swirl || 0,
    echo: settings.echo || 0, scanlines: settings.scanlines || 0
  };
}
// Build the fx object for the Video (hue is the video's cumulative spin).
function videoFx(hue) {
  const v = id => parseFloat($(id).value) || 0;
  return {
    hue: hue || 0, sat: v("videoSaturation") || 1,
    glow: v("videoGlow"), soft: v("videoSoft"),
    vignette: v("videoVignette"), fade: v("videoFade"),
    rgb: v("videoRgbSplit"),
    static: v("videoStatic"),
    invert: v("videoInvert"),
    kaleido: v("videoKaleido"), swirl: v("videoSwirl"),
    echo: v("videoEcho"), scanlines: v("videoScanlines")
  };
}
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
  if (busy || zooming || videoRunning) return;   // never shoot while a video is generating — the video run has the engine on its own model
  const shot = capturePhoto();
  if (!await ensureReady()) return;
  disposeCard(); stopPlaying();
  const q = dims();
  card = { id: "dc" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6), no: nextNumber(), name: "Dream " + new Date().toLocaleDateString(), model: MODEL.short, preset: $("presetSel").selectedOptions[0] ? $("presetSel").selectedOptions[0].textContent : "", w: q.w, h: q.h, created: Date.now(), frames: [], pathFrame: 0, zoomTotal: 1, settings: Object.assign({}, settings) };
  seedCv.width = q.w; seedCv.height = q.h; seedCv.getContext("2d").drawImage(shot, 0, 0);
  seedCardId = card.id;
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
  await tensorToCanvas(cur, artCv);
  hueTotal = (settings.hueSpin || 0) % 360;   // the first frame takes the first spin step
  applyFx(artCv, cameraFx(hueTotal));
  await pushFrame();
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
const ZOOM_MAX = 64;  // total magnification vs the seed. It used to ease to a stop at 16x, so after ~100–150 frames the zoom looked frozen — now the cap is 64x and the ease never quite reaches zero, so a long dream keeps falling in (slowly) instead of stopping.
let deep = null, deepZoom = 1, viewDepth = 1, driftX = 0, driftY = 0, lastViewDepth = 1;
let hueTotal = 0;   // cumulative hue rotation for the Color spin effect; resets with a new dream
let pathPhase = 0;   // where the zoom centre is along its path shape (0..1 = one full loop)
function resetDeep() { if (deep) { deep.dispose(); deep = null; } deepZoom = 1; viewDepth = 1; driftX = 0; driftY = 0; joyCX = 0; joyCY = 0; lastViewDepth = 1; hueTotal = 0; pathPhase = 0; }
// ===== ZOOM PATH SHAPES =====
// The zoom centre follows a shape. Each shape is a closed loop: pathPoint
// returns {x, y} in [-1, 1] for a phase t in [0, 1) around the loop.
// "random" keeps the old smooth wander, "straight" holds the centre
// still, "joystick" is steered live by the stick. Path segment
// sets how many frames one loop takes (x6), Path range how far it roams.
const SQUARE_VERTS = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
const TRI_VERTS = [[0, -1], [1, 1], [-1, 1]];
const STAR_VERTS = (() => {   // 5-point star: 10 vertices, outer r=1, inner r=0.42, starting at the top
  const v = [];
  for (let i = 0; i < 10; i++) {
    const r = i % 2 === 0 ? 1 : 0.42;
    const a = (i / 10) * Math.PI * 2 - Math.PI / 2;
    v.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return v;
})();
function polyPoint(verts, t) {
  // Walk a polygon's perimeter at constant speed (by edge length).
  const n = verts.length;
  const lens = [];
  let tot = 0;
  for (let i = 0; i < n; i++) {
    const a = verts[i], b = verts[(i + 1) % n];
    const l = Math.hypot(b[0] - a[0], b[1] - a[1]);
    lens.push(l); tot += l;
  }
  if (!tot) return { x: verts[0][0], y: verts[0][1] };
  let d = (((t % 1) + 1) % 1) * tot;
  for (let i = 0; i < n; i++) {
    if (d <= lens[i] || i === n - 1) {
      const u = lens[i] ? Math.max(0, Math.min(1, d / lens[i])) : 0;
      const a = verts[i], b = verts[(i + 1) % n];
      return { x: a[0] + (b[0] - a[0]) * u, y: a[1] + (b[1] - a[1]) * u };
    }
    d -= lens[i];
  }
  return { x: verts[0][0], y: verts[0][1] };
}
function pathPoint(shape, t) {
  const a = t * Math.PI * 2;   // angle around the loop
  const tri = x => {   // triangle wave in [-1, 1], period 2π
    const u = (((x / (Math.PI * 2)) % 1) + 1) % 1;
    return u < 0.5 ? u * 4 - 1 : 3 - u * 4;
  };
  switch (shape) {
    case "circle": return { x: Math.cos(a), y: Math.sin(a) };
    case "figure8": return { x: Math.sin(a), y: Math.sin(2 * a) / 2 };   // infinity / lemniscate
    case "heart": {   // parametric heart, scaled to [-1, 1]; y flipped so the point is down
      const s = Math.sin(a), c = Math.cos(a);
      return { x: 16 * s * s * s / 16, y: -(13 * c - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)) / 17 };
    }
    case "star": return polyPoint(STAR_VERTS, t);
    case "lissajous": return { x: Math.sin(3 * a), y: Math.sin(2 * a + Math.PI / 2) };   // 3:2 woven curve
    case "zigzag": return { x: tri(2 * a), y: tri(a + Math.PI / 2) };
    case "bounce": return { x: tri(a), y: tri(2 * a + 1) };   // bounces off the edges like a DVD logo (x 1 cycle, y 2 cycles, so the loop closes)
    case "wave": return { x: Math.sin(a), y: 0.6 * Math.sin(2 * a) };
    case "square": return polyPoint(SQUARE_VERTS, t);
    case "triangle": return polyPoint(TRI_VERTS, t);
    case "spiral": {   // spirals out and back in over 2 turns, so the loop is continuous
      const r = Math.abs(Math.sin(Math.PI * t));
      return { x: r * Math.cos(2 * a), y: r * Math.sin(2 * a) };
    }
    case "leftright": return { x: tri(a), y: 0 };
    case "updown": return { x: 0, y: tri(a) };
    case "diagonal": return { x: tri(a), y: tri(a) };
    default: return { x: 0, y: 0 };
  }
}
// ===== JOYSTICK =====
// A game-style stick that steers the zoom centre live while the dream
// runs. Push and hold to drive the centre in that direction — speed
// scales with how far the stick is pushed. Release and the stick springs
// back to centre; the centre holds still where it stopped. Push up to
// look up, right to look right.
let joyCX = 0, joyCY = 0;   // the centre, in [-1, 1] — steered by the stick
let joyDX = 0, joyDY = 0;   // the stick's current push, -1..1 (0 = released)
const JOY_STEP = 0.15;      // centre travel per dream frame at full push
function joyShow() {
  const w = $("joyWrap"); if (!w) return;
  w.hidden = settings.path !== "joystick";
}
function joyInit() {
  const base = $("joyBase"), knob = $("joyKnob");
  if (!base || !knob) return;
  const R = 44;   // how far the knob travels from centre (base 140px, knob 52px)
  let held = false;
  const setKnob = () => {
    knob.style.transform = "translate(-50%,-50%) translate(" + (joyDX * R) + "px," + (joyDY * R) + "px)";
  };
  const fromEvent = e => {
    const r = base.getBoundingClientRect();
    let dx = (e.clientX - (r.left + r.width / 2)) / R;
    let dy = (e.clientY - (r.top + r.height / 2)) / R;
    const m = Math.hypot(dx, dy);
    if (m > 1) { dx /= m; dy /= m; }
    joyDX = dx; joyDY = dy;
    setKnob();
  };
  base.addEventListener("pointerdown", e => {
    e.preventDefault();
    held = true;
    knob.style.transition = "none";
    if (base.setPointerCapture && e.pointerId != null) { try { base.setPointerCapture(e.pointerId); } catch (err) {} }
    fromEvent(e);
  });
  base.addEventListener("pointermove", e => { if (!held) return; e.preventDefault(); fromEvent(e); });
  const release = () => {
    if (!held) return;
    held = false;
    joyDX = 0; joyDY = 0;
    knob.style.transition = "transform .18s ease-out";
    setKnob();
  };
  base.addEventListener("pointerup", release);
  base.addEventListener("pointercancel", release);
  setKnob();
}
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
        // Ease the zoom as the true depth approaches the cap — but never
        // to a full stop. The old ease hit exactly 0 at 16x, so after
        // ~100–150 frames the zoom looked frozen. Now the ease floors at
        // 0.15, so even at the 64x cap the view keeps falling in at a
        // gentle 0.45% per frame instead of stopping.
        const ease = Math.max(0.15, 1 - viewDepth / ZOOM_MAX);
        const zf = 1 - (1 - settings.zoomFactor) * ease;
        // Dream strength settles with the zoom: as the view stops
        // magnifying, the dreaming eases to a gentle drift instead of
        // churning in place (the boil that read as jumpy past frame 50).
        const dreamEase = 0.25 + 0.75 * ease;
        // The zoom centre follows the selected path shape. "random" keeps
        // the old smooth wander; "straight" holds it still; the rest trace
        // their shape; "joystick" is steered live by the stick. Path
        // segment sets the loop length (x6 frames), Path range how far it
        // roams (0.08 = the shape's full extent, higher = wider).
        const seg = Math.max(4, Math.round(settings.segFrames || 10));
        pathPhase = (pathPhase + 1 / (seg * 6)) % 1;
        let px, py;
        if (settings.path === "random") {
          // Smooth wander: the centre takes a small step each frame
          // instead of jumping somewhere random (the old random jump each
          // frame read as jitter once the view got deep).
          driftX = Math.max(-1, Math.min(1, driftX + (Math.random() * 2 - 1) * 0.2));
          driftY = Math.max(-1, Math.min(1, driftY + (Math.random() * 2 - 1) * 0.2));
          px = driftX; py = driftY;
        } else if (settings.path === "straight") {
          px = 0; py = 0;
        } else if (settings.path === "joystick") {
          // Joystick: the stick steers the centre. Push and hold to drive
          // it in that direction; release and it holds still. Speed scales
          // with how far the stick is pushed.
          joyCX = Math.max(-1, Math.min(1, joyCX + joyDX * JOY_STEP));
          joyCY = Math.max(-1, Math.min(1, joyCY + joyDY * JOY_STEP));
          px = joyCX; py = joyCY;
        } else {
          const pp = pathPoint(settings.path, pathPhase);
          px = pp.x; py = pp.y;
        }
        const rangeScale = (settings.range || 0.08) / 0.08;
        const cw = Math.max(16, Math.round(w * zf)), ch = Math.max(16, Math.round(h * zf));
        const ox = px * rangeScale * (w - cw) / 2, oy = py * rangeScale * (h - ch) / 2;
        const z = zoomCrop(deep, zf, ox, oy, (settings.sharpen || 0) * (0.4 + 0.6 / (1 + card.frames.length * 0.01)) * dreamEase);
        deep.dispose(); deep = z;
        viewDepth = viewDepth / zf;   // true depth: upscaling never reduces it, only zooming grows it
        deepZoom = deepZoom / zf;
        const t0 = performance.now();
        // Dream strength eases down with total frames but never fades out:
        // a long dream keeps dreaming, just gentler.
        const lrEff = settings.lr * (0.35 + 0.65 / (1 + card.frames.length * 0.012)) * dreamEase * (1 + (settings.pulse || 0) * Math.sin(made * 0.25));   // Pulse: the dream strength breathes in and out, one full breath every ~25 frames
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
        hueTotal = (hueTotal + (settings.hueSpin || 0)) % 360;   // Color spin: the hue keeps cycling as the dream falls in
        applyFx(artCv, cameraFx(hueTotal));
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
    // The video opens on the original starting image and returns to it at
    // the end — before, it jumped straight into the dream and ended on the
    // deepest frame, so both ends felt cut off. The starting image is the
    // undreamed seed when this card was shot/uploaded in this session, else
    // the card's first frame (the closest to the original a gallery-loaded
    // card has). Each hold is ~1s: the same image drawn and requested
    // repeatedly until a second has passed, so it holds in the recording.
    const HOLD_MS = 1000;
    let startImg = null, startIsCanvas = false;
    if (seedCardId === card.id && seedCv.width) { startImg = seedCv; startIsCanvas = true; }
    else startImg = await decodeBlob(card.frames[0]);
    async function holdStart() {
      const t0 = performance.now();
      do {
        if (startImg) {
          const c = artCv.getContext("2d"); c.clearRect(0, 0, card.w, card.h); c.drawImage(startImg, 0, 0, card.w, card.h);
          exCtx.clearRect(0, 0, exCv.width, exCv.height); exCtx.drawImage(startImg, 0, 0, exCv.width, exCv.height);
        }
        if (manualFrames) { vtrack.requestFrame(); await sleep(60); }
        else await sleep(1000 / fpsIn);
      } while (performance.now() - t0 < HOLD_MS);
    }
    setStatus("Recording video… opening…"); setBar(0);
    await holdStart();
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
    setStatus("Recording video… closing…"); setBar(1);
    await holdStart();
    if (startImg && !startIsCanvas && startImg.close) startImg.close();
    await sleep(300); rec.stop(); await done;
    const blob = new Blob(chunks, { type: mime || "video/webm" });
    const ext = (mime || "").includes("mp4") ? "mp4" : "webm";
    await downloadBlob(blob, "dreamcamera-" + String(card.no).padStart(4, "0") + "." + ext);
    setStatus("Video saved (" + (blob.size / 1048576).toFixed(1) + " MB" + (exLong > 3840 ? ", 4K — 8K is stills only" : "") + "). Also keep it in the gallery to watch later.");
  } catch (e) { console.error(e); setStatus("Video export failed: " + (e && e.message || e), true); }
  setBar(null); busy = false; startPlaying(); updateUI();
}
async function exportStill() {
  if (!card || !card.frames.length) return;
  const im = await decodeBlob(card.frames[card.frames.length - 1]);
  const wantLong = EXPORT_SIZE[settings.exportSize] || 1920;
  // Try the requested size; if the high-res dream runs out of memory,
  // fall back to 4K instead of crashing. (8K holds two 8K canvases at
  // once, and the old PNG encode + base64 save held hundreds of MB more
  // on top — see downloadBlob. 4K is the floor.)
  const attempts = wantLong > 3840 ? [wantLong, 3840] : [wantLong];
  let c = null, doneLong = 0;
  for (const attempt of attempts) {
    try {
      const exScale = Math.max(1, attempt / Math.max(card.w, card.h));
      c = document.createElement("canvas"); c.width = Math.round(card.w * exScale); c.height = Math.round(card.h * exScale);
      const ctx = c.getContext("2d"); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      ctx.drawImage(im, 0, 0, c.width, c.height);
      const trueHiRes = attempt > Math.max(card.w, card.h);
      if (trueHiRes) {
        if (!await ensureReady()) { setStatus("Could not load the model for the high-res still.", true); return; }
        await dreamTiled(c, settings.lr, 2, (d, t) => { setStatus("Dreaming the still at " + c.width + "×" + c.height + "… tile " + d + "/" + t); setBar(d / t); });
        setBar(null);
      }
      doneLong = attempt;
      break;
    } catch (e) {
      console.error("still at " + attempt + " failed", e);
      c = null;
      if (attempt <= 3840) break;
      setStatus("That size ran out of memory — trying 4K instead…");
      await tf.nextFrame();
    }
  }
  if (im && im.close) im.close();
  if (!c || !doneLong) { setStatus("Still export failed (out of memory). Try a smaller export size.", true); return; }
  // 8K saves as JPEG, not PNG: a PNG at 8K is 50–100MB and its encode
  // plus the save held hundreds of MB at once. JPEG at 0.95 is visually
  // near-lossless on a deepdream and a fifth the size.
  const is8k = doneLong >= 7680;
  const b = await toBlob(c, is8k ? "image/jpeg" : "image/png", is8k ? 0.95 : 1);
  await downloadBlob(b, "dreamcamera-" + String(card.no).padStart(4, "0") + (is8k ? ".jpg" : ".png"));
  setStatus("Still saved (" + c.width + "×" + c.height + (doneLong > Math.max(card.w, card.h) ? ", dreamed at full size" : "") + (doneLong < wantLong ? " — 8K ran out of memory, saved at 4K" : "") + ").");
}
async function downloadBlob(blob, name) {
  // In the app the WebView ignores the download attribute, so hand the
  // file to the app to save into Downloads. In a browser, download it.
  if (window.DreamCameraBridge && window.DreamCameraBridge.saveFile) {
    // Large files go in 5MB chunks: the old single-shot save turned the
    // whole file into one base64 data URL, so the blob + the base64
    // string + the decoded bytes were all in memory at once — that is
    // what crashed the app on 8K stills and big videos. Chunked, only
    // one 5MB piece is ever held at a time. (An older app without the
    // chunk methods falls back to the single-shot save.)
    const CHUNK = 5 * 1024 * 1024;
    if (window.DreamCameraBridge.saveFileChunk && blob.size > CHUNK) {
      try {
        window.DreamCameraBridge.saveFileStart(name, blob.type || "application/octet-stream");
        for (let off = 0; off < blob.size; off += CHUNK) {
          const part = blob.slice(off, Math.min(blob.size, off + CHUNK));
          const b64 = await new Promise((res, rej) => {
            const r = new FileReader();
            r.onload = () => res(String(r.result).split(",")[1] || "");
            r.onerror = () => rej(new Error("read failed"));
            r.readAsDataURL(part);
          });
          window.DreamCameraBridge.saveFileChunk(b64);
          await sleep(30);   // let the app write this chunk before the next
        }
        window.DreamCameraBridge.saveFileEnd();
        return;
      } catch (e) {
        console.error(e);
        try { window.DreamCameraBridge.saveFileEnd(); } catch (err) {}
        // fall through to the single-shot save
      }
    }
    // Single-shot save (small files, or an older app without chunking).
    await new Promise(res => {
      const r = new FileReader();
      r.onload = () => { try { window.DreamCameraBridge.saveFile(r.result, name, blob.type || "application/octet-stream"); } catch (e) { downloadBlobWeb(blob, name); } res(); };
      r.onerror = () => { downloadBlobWeb(blob, name); res(); };
      r.readAsDataURL(blob);
    });
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

// ===== TRIM FRAMES =====
// Remove a section of the open dream — any number of frames. Pick a
// frame range (1-based, as shown while it plays) and either remove just
// that range, or keep only that range and drop the rest. At least one
// frame must remain. If the trim cuts the end off, the stored depth is
// re-estimated from the frames left, so a later Continue resumes at the
// right depth. If the dream is already in the gallery, the saved copy is
// updated too.
function trimRange() {
  const total = card && card.frames.length || 0;
  if (!total) return null;
  let from = Math.floor(parseFloat($("trimFrom").value) || 0);
  let to = Math.floor(parseFloat($("trimTo").value) || 0);
  from = Math.max(1, Math.min(total, from));
  to = Math.max(1, Math.min(total, to));
  return { from, to, total };
}
function trimUpdate() {
  const total = card && card.frames.length || 0;
  const has = total > 0;
  $("trimFrom").max = total; $("trimTo").max = total;
  $("trimTotal").textContent = has ? "of " + total + " frames" : "";
  const r = trimRange();
  let msg = "", ok = false;
  if (!has) { msg = "No dream open yet."; }
  else if (!r || r.from > r.to) { msg = "Pick a range — From must be at or before To."; }
  else {
    const n = r.to - r.from + 1;
    if (n >= total) { msg = "That would remove every frame — keep at least one."; }
    else {
      ok = true;
      msg = "Removes frames " + r.from + "–" + r.to + " (" + n + " frame" + (n === 1 ? "" : "s") + "). " + (total - n) + " left.";
    }
  }
  $("trimPreview").textContent = msg;
  const dis = !has || !ok || busy || zooming;
  $("trimRemoveBtn").disabled = dis;
  $("trimKeepBtn").disabled = dis;
  return r;
}
async function trimApply(mode) {
  if (!card || busy || zooming) return;
  const r = trimUpdate();
  if (!r || r.from > r.to) return;
  const total = r.total, n = r.to - r.from + 1;
  if (n >= total) { setStatus("That would remove every frame — keep at least one.", true); return; }
  stopPlaying();
  const cutEnd = (r.to >= total);   // the trim takes the last frame off
  if (mode === "remove") {
    card.frames.splice(r.from - 1, n);
    setStatus("Removed frames " + r.from + "–" + r.to + " (" + n + "). " + card.frames.length + " frames left.");
  } else {
    card.frames = card.frames.slice(r.from - 1, r.to);
    setStatus("Kept frames " + r.from + "–" + r.to + " (" + n + "). " + card.frames.length + " frames left.");
  }
  if (cutEnd) card.viewDepth = Math.min(ZOOM_MAX, 1 + card.frames.length * 0.12);   // re-estimate the depth from the frames left
  card.pathFrame = Math.min(card.pathFrame || 0, card.frames.length - 1);
  // Keep the range boxes in range now that the total changed.
  $("trimFrom").value = Math.max(1, Math.min(card.frames.length, r.from));
  $("trimTo").value = Math.max(1, Math.min(card.frames.length, r.to));
  // If this dream is already in the gallery, update the saved copy too.
  try {
    const ex = await idbDo("readonly", st => st.get(card.id));
    if (ex) {
      const rec = { id: card.id, created: card.created, no: card.no, name: card.name, model: card.model, preset: card.preset, w: card.w, h: card.h, frames: card.frames.slice(), thumb: card.frames[0], zoomTotal: card.zoomTotal, viewDepth: card.viewDepth, settings: card.settings };
      await idbDo("readwrite", st => st.put(rec));
      renderGallery();
    }
  } catch (e) { console.error(e); }
  startPlaying(); updateUI(); trimUpdate();
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
  $("shootBtn").disabled = busy || zooming || videoRunning;
  $("stopBtn").disabled = !(busy || zooming);
  $("continueBtn").disabled = !card || !card.frames.length || busy || zooming || card.frames.length >= TOTAL_MAX;
  $("videoBtn").disabled = !card || !card.frames.length || busy || zooming;
  $("stillBtn").disabled = !card || !card.frames.length || busy || zooming;
  $("keepBtn").disabled = !card || !card.frames.length || busy || zooming;
  $("modelNote").textContent = MODEL.note || "";
  trimUpdate();
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
  const map = { steps: "steps", lr: "lr", zoomFactor: "zoomFactor", range: "range", segFrames: "segFrames", frames: "frames", playFps: "playFps", playFpsOut: "playFpsOut", sharpen: "sharpen", octaves: "octaves", jitter: "jitter", hueSpin: "hueSpin", saturation: "saturation", pulse: "pulse", glow: "glow", soft: "soft", vignette: "vignette", fade: "fade", rgbSplit: "rgbSplit", staticAmt: "staticAmt", invertAmt: "invertAmt", kaleido: "kaleido", swirl: "swirl", echo: "echo", scanlines: "scanlines" };
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
  if (id === "hueSpin") return (+v).toFixed(1);
  if (id === "saturation") return (+v).toFixed(2);
  if (id === "pulse") return (+v).toFixed(2);
  if (id === "glow" || id === "vignette" || id === "fade" || id === "staticAmt" || id === "invertAmt" || id === "kaleido" || id === "swirl" || id === "echo" || id === "scanlines") return (+v).toFixed(2);
  if (id === "soft") return (+v).toFixed(2);
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
  bindRange("hueSpin", "hueSpin", v => (+v).toFixed(1));
  bindRange("saturation", "saturation", v => (+v).toFixed(2));
  bindRange("pulse", "pulse", v => (+v).toFixed(2));
  bindRange("glow", "glow", v => (+v).toFixed(2));
  bindRange("soft", "soft", v => (+v).toFixed(2));
  bindRange("vignette", "vignette", v => (+v).toFixed(2));
  bindRange("fade", "fade", v => (+v).toFixed(2));
  bindRange("rgbSplit", "rgbSplit", v => String(v));
  bindRange("staticAmt", "staticAmt", v => (+v).toFixed(2));
  bindRange("invertAmt", "invertAmt", v => (+v).toFixed(2));
  bindRange("kaleido", "kaleido", v => (+v).toFixed(2));
  bindRange("swirl", "swirl", v => (+v).toFixed(2));
  bindRange("echo", "echo", v => (+v).toFixed(2));
  bindRange("scanlines", "scanlines", v => (+v).toFixed(2));
  joyInit(); joyShow();   // the joystick (steers the zoom centre while dreaming)
  $("path").value = settings.path;
  $("path").addEventListener("change", () => { settings.path = $("path").value; saveSettings(); joyShow(); });
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
  $("trimFrom").addEventListener("input", trimUpdate);
  $("trimTo").addEventListener("input", trimUpdate);
  $("trimRemoveBtn").onclick = () => trimApply("remove");
  $("trimKeepBtn").onclick = () => trimApply("keep");
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
      seedCardId = card.id;
      artCv.width = q.w; artCv.height = q.h; artCv.getContext("2d").drawImage(c, 0, 0);
      $("viewBox").classList.add("dreaming"); $("viewHint").textContent = "";
      URL.revokeObjectURL(url);
      await startDream(); if (card && card.frames.length) await zoomRun(); updateUI();
    };
    im.src = url;
  });
}

// ===== VIDEO GENERATOR =====
// Clean rebuild (v1.57). One job: upload a video of any length, turn it
// into frames, deepdream each frame with the video tab's own model, and
// put the frames back together as an MP4. No modes, no presets, no style
// photo — the generator is the whole tab.
//
// v1.60: frames come from WebCodecs VideoDecoder, not a <video> element.
// Seeking is GONE — seekTo/seekFrame/rVFC were removed, not tweaked.
// The v1.57–v1.59 saga proved the <video> path cannot be trusted on this
// device: the playhead reads at the target while drawImage returns the
// previous frame (v1.57/1.58), and requestVideoFrameCallback never fires
// for the element even when it is visible to the compositor (v1.59:
// every frame "fallback", intro logo still at 2.1s). VideoDecoder hands
// us decoded VideoFrames with exact timestamps — no <video>, no
// compositor, no seeking, no rVFC — so a stale frame is impossible: a
// frame IS the decoded data at its time. A minimal MP4 demuxer
// (demuxMP4, below) reads the file's own sample table; it was verified
// against Coby's real file (1080x2340 H.264, 317 frames, avc1.4d0033).
//
// v1.62: 4K output — Output size now offers 2160 (4K, the default), which
// keeps the source aspect (9:16 → 2160×3840, 16:9 → 3840×2160, 1:1 →
// 2160×2160) at a 30 Mbps bitrate, and falls back to 1024 with a clear
// note if 4K recording fails on the device. The dream still runs at 192
// for speed; the 4K is an upscale of that dream, so it is sharper on a
// high-res screen but adds no new dream detail. Also: the demuxer now
// accepts H.265/HEVC (hvc1/hev1 + hvcC), VP9 (vp09 + vpcC), and AV1
// (av01 + av1C), not just H.264 — a real MP4 in another codec was
// rejected as "not H.264" — and the run checks VideoDecoder.
// isConfigSupported for the file's codec first, so an unsupported codec
// fails with its real name instead of a misleading error.
// v1.61: the decoder is no longer fatal on its first error — v1.60
// stopped after 2 frames when the decoder died, and the post-loop
// status overwrote the decoder-error message with a bare "Stopped",
// so no error was ever shown. Now a decoder error recreates the
// decoder and resumes from the last key frame (bounded to 3 recoveries),
// bad samples are skipped instead of fed, and the status always names
// why the run stopped (stopReason).
// The hard-won fixes from the v1.50–v1.56 video saga are built in, not
// patched on:
//  - the dream runs at a capped 192 short side (v1.59, was 256) and is
//    upscaled onto the output canvas, so the output keeps the source
//    dimensions. 192 is 1.83x faster than 256 and the dream stays visible.
//  - carryover: half of the previous dreamed frame is blended into the
//    next, so the dream builds up and is visible (one isolated frame
//    changes ~0.01% and looks raw).
//  - if the dream engine is lost, the run switches to WASM then CPU
//    (bounded, once per run), shrinks to 128 with 1 step, retries the
//    frame, and says so. Five consecutive failures of any kind end it.
//  - live preview: the first frame shows immediately; each dreamed frame
//    is drawn as it is made; a skipped frame shows its raw frame.
//  - honest status: frame X of total + seconds in the video + seconds on
//    the frame + engine note; the end status reports the actual dreamed
//    count, never the planned total.
let videoSrc = null, videoMeta = null, videoFile = null;
let videoRunning = false, videoOutBlob = null, videoOutUrl = null, videoOutMime = null;
function setVideoStatus(t, err) { const el = $("videoStatus"); el.textContent = t; el.style.color = err ? "#ff7b7b" : "#7df9ff"; }
function setVideoBar(p) { const b = $("videoBar"); if (p == null) { b.style.visibility = "hidden"; return; } b.style.visibility = "visible"; b.firstChild.style.width = Math.round(Math.max(0, Math.min(1, p)) * 100) + "%"; }
function videoDims(vw, vh, shortOverride) {
  const short = shortOverride || parseInt(($("videoSize") && $("videoSize").value) || "512", 10) || 512;   // the output size — separate from the Camera quality, so a video can be big while the Camera stays fast
  // 4K (short 2160) keeps the source aspect: 9:16 → 2160×3840, 16:9 → 3840×2160, 1:1 → 2160×2160.
  // The scaled side is rounded to an even number — H.264 (4:2:0) needs
  // even width and height, and a plain Math.round can land odd (Coby's
  // 1080×2340 at 1024 gives 2219, which MediaRecorder cannot encode).
  const even = n => Math.max(2, Math.round(n / 2) * 2);
  if (vw >= vh) return { w: even(short * vw / vh), h: short };
  return { w: short, h: even(short * vh / vw) };
}
// The video DREAM runs smaller than the video itself: one gradient step
// through a free-size model (v1, the default) at 512×1109 is a huge GPU
// allocation in the Android WebView and loses the device ("[Device] is
// lost") — that is what forced the 256 cap in v1.54. v1.71: the Dream
// detail setting doubled at Coby's request — 384 is fast, 512 is the
// default, 768 is max detail. v1.72: 1080 and 1440 added at his request —
// at 768 his 1080×2340 source is dreamed at 768×1664 and stretched ~2.8×
// to the 2160 output, which is the blur he saw; at 1080 the dream runs
// at the source's own 1080×2340 (2× upscale), at 1440 it runs at
// 1440×3120 for a wider source (1.5× upscale). Measured against the
// real Inception v1 model (WASM, lr 0.06, steps 4): 384 = 48s/frame,
// 512 = 87s/frame (1.8x 384), 768 = 195s/frame (2.3x 512); 768 IS
// sharper than 512 (1.36x high-frequency energy at a common 1664x768).
// WASM times are WASM ONLY — never use them to estimate phone time:
// Coby reported 768 "doesnt take that long at all" on his phone, so the
// v1.71 phone estimate (10-16h for a 62s video at 768) was far too
// pessimistic and is withdrawn. 1080/1440 use ~2×/3.5× the pixels of
// 768, so the device-loss risk is real — if the GPU loses the device
// the backend falls back to WASM/CPU and the run still fails honestly.
// The output video keeps the size from Output size — the dreamed frame
// is upscaled onto it.
const VIDEO_DREAM_MAX = 384;   // the fast option / floor — the setting picks the actual cap
function videoDreamDetail() {
  const v = parseInt(($("videoDreamDetail") && $("videoDreamDetail").value) || "512", 10);
  return [384, 512, 768, 1080, 1440, 2160].includes(v) ? v : 512;   // v1.73: 2160 added (for a 4K source; dreamed tiled, see VIDEO_TILE_SIZE)
}
// v1.73: tiled video dream. A free-size model at a dream short side
// above one tile (768, the largest size proven on the phone) cannot run
// as a single pass — the WASM heap could not even hold the backward
// pass at 1080x2340, and a 2160x3840 pass would very likely lose the
// WebGPU device. So the video dreams in 768 tiles with 192 overlap
// (the Photos feather-blend, dreamTiled). The weave, zoom, noise and
// depth mask all run at full-frame level, so only the dream step itself
// is tiled; fixed-size models never tile (dreamSteps letterboxes them
// to their native size).
const VIDEO_TILE_SIZE = 768, VIDEO_TILE_OVERLAP = 192;
// v1.65: depth/detail separation. A per-pixel mask of how much local
// detail a frame has — |frame - blurred frame|, smoothed and normalized
// to [0,1] per frame, then DILATED so whole objects are covered, not
// just their outlines: a bare edge map is nearly all thin lines (a
// Flappy Bird frame is 95% flat, and a pipe's green interior is as flat
// as the sky), so without dilation only the outlines would dream hard
// and the object interiors would be damped like the sky. The dilation
// runs at quarter resolution (downsample, max-pool, upsample) so it is
// cheap: a 13-wide max-pool at 1/4 scale spreads ~48px at full scale at
// the default 512 dream size, enough to cover a pipe interior, and the
// upsample keeps the mask smooth. Edge-rich objects (bird, pipes, score, text)
// score near 1; flat background (sky) scores near 0. A true depth model
// was considered and rejected for video: it would need a new ~20MB model
// bundled plus a full inference per frame, and on flat 2D game art a
// photo-trained depth map is mostly noise — the edges already mark the
// objects. This mask is a few pool ops on the dream-size tensor
// (milliseconds), needs no new model, and tracks the objects as they move.
function videoDetailMask(t) {
  return tf.tidy(() => {
    const blur = tf.avgPool(t, 5, 1, "same");
    const det = tf.abs(tf.sub(t, blur)).mean(3, true);   // [1,H,W,1] detail
    const sm = tf.avgPool(det, 5, 1, "same");            // smooth so the mask does not speckle
    const mx = tf.max(sm);
    const m = sm.div(mx.add(1e-6)).clipByValue(0, 1);
    const h = m.shape[1], w = m.shape[2];
    const small = tf.avgPool(m, 4, 4, "same");           // quarter resolution
    const dil = tf.maxPool(small, 13, 1, "same");        // dilate ~24px at full scale
    return tf.image.resizeBilinear(dil, [h, w]).clipByValue(0, 1);
  });
}
// Carryover: one frame dreamed on its own changes by ~0.01% (a single
// dream step moves pixels ~0.025/255), so it looks identical to the raw
// frame — the run "skips" every frame to the eye. Blending in half of
// the previous dreamed frame lets the dream build up across frames (the
// way Photos builds up over a run) so it becomes clearly visible by ~frame 7.
const FRAME_WEAVE = 0.5;
function videoDreamDims(vd, max) {
  const cap = max || videoDreamDetail();   // v1.72: the Dream detail setting (512 default, up to 1440); an explicit max (e.g. the 128 fallback) still wins
  const short = Math.min(vd.w, vd.h);
  if (short <= cap) return { w: vd.w, h: vd.h };
  const s = cap / short;
  return { w: Math.max(1, Math.round(vd.w * s)), h: Math.max(1, Math.round(vd.h * s)) };
}
function switchTab(name) {
  const cam = name === "camera";
  $("cameraTab").hidden = !cam;
  $("videoTab").hidden = cam;
  $("tabCamera").classList.toggle("on", cam);
  $("tabVideo").classList.toggle("on", !cam);
  if (cam) applyRatio();
}
// The video tab's own model — all 12, independent of the Photos tab's
// model. Stored on its own so the choice survives a reload; settings.model
// (Photos) is never touched by the video tab.
const VIDEO_MODEL_KEY = "dreamcamera_video_model";
function videoModelKey() {
  try { const k = localStorage.getItem(VIDEO_MODEL_KEY); if (k && MODELS[k]) return k; } catch (e) {}
  return "v1";
}
function buildVideoModelSelect() {
  const sel = $("videoModelSel"); sel.innerHTML = "";
  for (const k of Object.keys(MODELS)) { const o = document.createElement("option"); o.value = k; o.textContent = MODELS[k].label; sel.appendChild(o); }
  sel.value = videoModelKey();
  $("videoModelNote").textContent = (MODELS[sel.value] || MODELS.v1).note || "";
  sel.onchange = () => {
    try { localStorage.setItem(VIDEO_MODEL_KEY, sel.value); } catch (e) {}
    $("videoModelNote").textContent = (MODELS[sel.value] || MODELS.v1).note || "";
    setVideoStatus(MODELS[sel.value].label + " selected — it loads when you generate.");
  };
}
// The video tab's own dream strength (learning rate). The Photos tab's lr
// slider is a different control — the generator is self-contained.
function videoLr() { return parseFloat($("videoLr").value) || 0.01; }
// v1.70: Ramp up — the Steps / Dream strength sliders set the START
// values. With the toggle on, both ramp from the set values at frame 0
// to the sliders' own maximums (Steps 5, Strength 0.100) at the last
// frame, evenly spaced. Persisted like the video model.
const VIDEO_RAMP_KEY = "dreamcamera_video_ramp";
function videoRampOn() { const el = $("videoRamp"); return !!(el && el.checked); }
// v1.78: the sliders ARE the start — no flags, no "did he touch it"
// tracking. The HTML defaults are the minimums (Steps 1, Strength
// 0.010), so an untouched slider starts at the lowest and a slider he
// set to 2 starts at 2. v1.77's touched-flags are gone: they were
// in-memory only, the slider values are never persisted, and browser
// form restoration after a reload restored the displayed value without
// firing a trusted input event — so the flag stayed false and the
// capture fell back to the minimum (1) even though the slider showed
// 2, and restoreRampSliders then wrote that 1 back visibly.
// Ramp progress for frame i of total — 0 at frame 0, 1 at the last
// frame; a 1-frame video dreams at the max.
function videoRampP(i, total) { return total > 1 ? i / (total - 1) : 1; }
function updateVideoUI() {
  $("videoDreamBtn").disabled = videoRunning || !videoMeta || busy || zooming;
  $("videoStopBtn").disabled = !videoRunning;
  $("videoSaveBtn").disabled = !videoOutBlob || videoRunning;
}
// Switch the live engine to the video tab's own model (and its own dream
// layers), without touching the Photos tab's model. The Photos model is
// put back by restorePhotoModel() at the end of every video run.
let savedPhotoLayers = null;
async function activateVideoModel() {
  const key = videoModelKey();
  // The video run uses the fastest working engine (WebGPU first), not
  // whatever engine the Photos tab happens to be on — a Photos session
  // left on WASM/CPU would otherwise make the whole video run at that
  // slow pace (v1.57 ran ~6s a frame). Reload the model only if the model
  // or the engine changes.
  if (!engineOrder.length) await benchEngines();
  const best = engineOrder[0];
  const curB = (typeof tf !== "undefined" && tf.getBackend) ? tf.getBackend() : null;
  const modelChanged = MODEL.short !== key;
  const engineChanged = !!(best && curB && curB !== best);
  if (modelChanged || engineChanged) {
    if (savedPhotoLayers == null) savedPhotoLayers = (settings.layers || []).slice();
    if (modelChanged) {
      MODEL = MODELS[key]; P = PRE[MODEL.pre];
      settings.layers = MODEL.pref.slice();
    }
    ready = false; model = null; midModel = null; grads = {};
  }
  return await ensureReady();
}
// Put the Photos tab's model (and its layers) back after a video run, so
// the next photo shoot uses the Photos model, not the video model. The
// engine loads lazily on the next shoot.
function restorePhotoModel() {
  try { if (model) model.dispose(); } catch (e) {}
  model = null; midModel = null; grads = {};
  const photo = MODELS[settings.model] || MODELS.v1;
  MODEL = photo; P = PRE[photo.pre];
  if (savedPhotoLayers) { settings.layers = savedPhotoLayers; savedPhotoLayers = null; }
  else settings.layers = photo.pref.slice();
  ready = false;
  saveSettings();   // persist the Photos layers again, so a save during the run cannot leave the video model's layers stored
  buildModelSelect(); buildLayers(); buildPresets(); updateUI();
}
// HEVC codec string (ISO/IEC 14496-15 Annex E.3, as VideoDecoder wants
// it): hev1.<profile_idc>.<compat>.<tier><level_idc>.<constraints>.
// compat: the 32 hvcC compatibility flags, bit-reversed — flag j maps to
// bit j of the value (flag[31] is the most significant bit, flag[0] the
// least), so a Main + Main 10 stream is "6". tier: L (Main) or H (High).
// constraints: the 6 constraint bytes, each as hex, separated by periods,
// trailing zero bytes omitted (progressive frame-only is "B0").
function hevcCodecSuffix(hvcC) {
  const space = (hvcC[1] >> 6) & 0x03;
  const spaceStr = space === 0 ? "" : ["", "A", "B", "C"][space];
  const profile = hvcC[1] & 0x1F;
  const tier = (hvcC[1] >> 5) & 0x01 ? "H" : "L";
  const level = hvcC[12];
  let compat = 0;
  const flags = ((hvcC[2] << 24) | (hvcC[3] << 16) | (hvcC[4] << 8) | hvcC[5]) >>> 0;
  for (let j = 0; j < 32; j++) if ((flags >>> (31 - j)) & 1) compat |= (1 << j);
  const consBytes = [];
  for (let i = 0; i < 6; i++) consBytes.push(hvcC[6 + i].toString(16).padStart(2, "0").toUpperCase());
  while (consBytes.length > 1 && consBytes[consBytes.length - 1] === "00") consBytes.pop();
  const cons = consBytes.join(".");
  return spaceStr + profile + "." + (compat >>> 0).toString(16).toUpperCase() + "." + tier + level + (cons ? "." + cons : "");
}
// VP9 codec string (vp09.PP.LL.DD): profile, level, bit depth, each
// 2 digits. The vpcC record is profile, level, bitDepth/chroma/range,
// primaries, transfer, matrix, initSize, initData — but ffmpeg writes a
// 4-byte version/flags prefix (01 00 00 00) before the record, so the
// fields sit at +4; a bare record has them at +0. Pick the offset whose
// bit depth is a real VP9 depth (8/10/12); the other offset reads 0
// (prefixed) or garbage (bare) there.
function vp9CodecString(vpcC) {
  const depthAt = o => (vpcC[o + 2] >> 4) & 0x0F;
  const valid = o => vpcC.length >= o + 3 && vpcC[o] <= 3 && (depthAt(o) === 8 || depthAt(o) === 10 || depthAt(o) === 12);
  const off = valid(4) ? 4 : 0;
  const profile = vpcC[off], level = vpcC[off + 1], depth = depthAt(off);
  return "vp09." + String(profile).padStart(2, "0") + "." + String(level).padStart(2, "0") + "." + String(depth).padStart(2, "0");
}
// Minimal MP4 demuxer (v1.60) — reads the file's own sample table so
// VideoDecoder can be fed exact samples. No <video>, no seeking.
// Handles: 32/64-bit box sizes, moov before or after mdat, stco/co64,
// stsc/stsz/stts/ctts (v0 + v1)/stss, uniform + per-sample sizes.
// Codecs: H.264 (avc1 + avcC), H.265/HEVC (hvc1/hev1 + hvcC), VP9
// (vp09 + vpcC), AV1 (av01 + av1C) — v1.62, because a real MP4 that is
// not H.264 (his phone records HEVC) was rejected as "not H.264".
// Throws on: no moov (fragmented MP4), no video track, unsupported
// video codec (named in the message), missing config record. Returns
// { duration (s), width, height, codec (e.g. "avc1.4d0033" or
// "hev1.1.6.L153.B0"), codecName ("H.264", "H.265/HEVC", "VP9", "AV1"),
// description (Uint8Array, the config record), samples: [{ offset, size,
// ctsUs, durUs, key }] } in file (decode) order. Verified against Coby's
// real file (1080x2340 H.264, 317 frames): duration 4.5662s, codec
// avc1.4d0033, 317 samples, cts==dts (no B-frames), all sample offsets in
// bounds, NAL lengths consistent, every target at 8/12/16 fps covered.
function demuxMP4(ab) {
  const buf = new Uint8Array(ab);
  const dv = new DataView(ab);
  function boxes(start, end) {
    const out = [];
    let off = start;
    while (off + 8 <= end) {
      let size = dv.getUint32(off);
      const type = String.fromCharCode(buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7]);
      let hdr = 8;
      if (size === 1) { size = Number(dv.getBigUint64(off + 8)); hdr = 16; }
      else if (size === 0) { size = end - off; }
      if (size < hdr || off + size > end) throw new Error("bad MP4 box " + type);
      out.push({ type, payload: off + hdr, end: off + size });
      off += size;
    }
    return out;
  }
  const find = (list, type) => list.find(b => b.type === type);
  const topBoxes = boxes(0, buf.length);
  const moov = find(topBoxes, "moov");
  if (!moov) throw new Error("no moov box — this file looks like a fragmented MP4, which is not supported");
  if (find(topBoxes, "moof")) throw new Error("fragmented MP4 (moof) — this file was saved as fragments (e.g. a video saved from this app); re-dreaming it is not supported yet");
  let video = null;
  for (const trak of boxes(moov.payload, moov.end).filter(b => b.type === "trak")) {
    const mdia = find(boxes(trak.payload, trak.end), "mdia");
    if (!mdia) continue;
    const mdiaBoxes = boxes(mdia.payload, mdia.end);
    const hdlr = find(mdiaBoxes, "hdlr");
    if (!hdlr) continue;
    const handler = String.fromCharCode(buf[hdlr.payload + 8], buf[hdlr.payload + 9], buf[hdlr.payload + 10], buf[hdlr.payload + 11]);
    if (handler !== "vide") continue;
    const mdhd = find(mdiaBoxes, "mdhd");
    const minf = find(mdiaBoxes, "minf");
    const stbl = find(boxes(minf.payload, minf.end), "stbl");
    const st = {};
    for (const b of boxes(stbl.payload, stbl.end)) st[b.type] = b;
    video = { mdhd, st };
    break;
  }
  if (!video) throw new Error("no video track in this file");
  const st = video.st;
  // mdhd: version 0 has 32-bit creation/modification times (timescale at
  // +12); version 1 has 64-bit times (timescale at +20). MediaRecorder
  // writes version 1, so reading +12 gives 0 and a bogus "bad timescale".
  const mdhdVer = buf[video.mdhd.payload];
  const timescale = dv.getUint32(video.mdhd.payload + (mdhdVer === 1 ? 20 : 12));
  if (!timescale) throw new Error("bad video timescale");
  // stsd -> video sample entry -> config record. H.264 (avc1 + avcC),
  // H.265/HEVC (hvc1/hev1 + hvcC), VP9 (vp09 + vpcC), AV1 (av01 + av1C).
  // A valid MP4 with any other video codec is rejected with its real
  // name — never "not H.264", which is wrong for a real MP4.
  if (!st.stsd) throw new Error("no sample description (stsd)");
  const entries = boxes(st.stsd.payload + 8, st.stsd.end);
  const entry = entries.find(b => b.type === "avc1" || b.type === "hvc1" || b.type === "hev1" || b.type === "vp09" || b.type === "av01");
  if (!entry) {
    const got = entries.length ? entries.map(b => b.type).join(", ") : "none";
    throw new Error("unsupported video codec in this MP4 (" + got + ") — this device needs H.264, H.265/HEVC, VP9, or AV1");
  }
  const width = dv.getUint16(entry.payload + 24);
  const height = dv.getUint16(entry.payload + 26);
  const children = boxes(entry.payload + 78, entry.end);
  const hex = n => n.toString(16).padStart(2, "0");
  let codec, codecName, description;
  if (entry.type === "avc1") {
    const avcC = find(children, "avcC");
    if (!avcC) throw new Error("no H.264 config (avcC)");
    description = new Uint8Array(buf.subarray(avcC.payload, avcC.end));
    codec = "avc1." + hex(description[1]) + hex(description[2]) + hex(description[3]);
    codecName = "H.264";
  } else if (entry.type === "hvc1" || entry.type === "hev1") {
    const hvcC = find(children, "hvcC");
    if (!hvcC) throw new Error("no H.265 config (hvcC)");
    description = new Uint8Array(buf.subarray(hvcC.payload, hvcC.end));
    codec = entry.type + "." + hevcCodecSuffix(description);
    codecName = "H.265/HEVC";
  } else if (entry.type === "vp09") {
    const vpcC = find(children, "vpcC");
    if (!vpcC) throw new Error("no VP9 config (vpcC)");
    description = new Uint8Array(buf.subarray(vpcC.payload, vpcC.end));
    codec = vp9CodecString(description);
    codecName = "VP9";
  } else {   // av01
    const av1C = find(children, "av1C");
    if (!av1C) throw new Error("no AV1 config (av1C)");
    description = new Uint8Array(buf.subarray(av1C.payload, av1C.end));
    const profile = (description[1] >> 5) & 0x07;
    const levelIdx = description[1] & 0x1F;
    const tier = (description[2] >> 7) & 0x01;
    const depth = (description[2] >> 5) & 0x01 ? 12 : ((description[2] >> 6) & 0x01 ? 10 : 8);
    codec = "av01." + profile + "." + String(levelIdx).padStart(2, "0") + (tier ? "H" : "M") + "." + String(depth).padStart(2, "0");
    codecName = "AV1";
  }
  // stts -> per-sample decode timestamps + durations
  if (!st.stts) throw new Error("no time-to-sample table (stts)");
  const dts = [], dur = [];
  let t = 0;
  const sttsCount = dv.getUint32(st.stts.payload + 4);
  for (let k = 0; k < sttsCount; k++) {
    const count = dv.getUint32(st.stts.payload + 8 + k * 8);
    const delta = dv.getUint32(st.stts.payload + 12 + k * 8);
    for (let j = 0; j < count; j++) { dts.push(t); dur.push(delta); t += delta; }
  }
  const sampleCount = dts.length;
  if (!sampleCount) throw new Error("no video samples in this file");
  // ctts -> composition offsets (absent when there are no B-frames)
  const cto = new Array(sampleCount).fill(0);
  if (st.ctts) {
    const ver = buf[st.ctts.payload];
    const cnt = dv.getUint32(st.ctts.payload + 4);
    let k = 0;
    for (let i = 0; i < cnt; i++) {
      const count = dv.getUint32(st.ctts.payload + 8 + i * 8);
      const off = ver === 1 ? dv.getInt32(st.ctts.payload + 12 + i * 8) : dv.getUint32(st.ctts.payload + 12 + i * 8);
      for (let j = 0; j < count && k < sampleCount; j++) cto[k++] = off;
    }
  }
  // stss -> sync (key) samples, 1-based; absent means every sample is a key
  let sync = null;
  if (st.stss) {
    sync = new Set();
    const cnt = dv.getUint32(st.stss.payload + 4);
    for (let i = 0; i < cnt; i++) sync.add(dv.getUint32(st.stss.payload + 8 + i * 4));
  }
  // stsz -> sample sizes
  if (!st.stsz) throw new Error("no sample sizes (stsz)");
  const uniformSize = dv.getUint32(st.stsz.payload + 4);
  const sizes = new Array(sampleCount);
  if (uniformSize) sizes.fill(uniformSize);
  else for (let i = 0; i < sampleCount; i++) sizes[i] = dv.getUint32(st.stsz.payload + 12 + i * 4);
  // stsc -> samples per chunk
  if (!st.stsc) throw new Error("no sample-to-chunk table (stsc)");
  const scCount = dv.getUint32(st.stsc.payload + 4);
  const scEntries = [];
  for (let i = 0; i < scCount; i++) scEntries.push({ firstChunk: dv.getUint32(st.stsc.payload + 8 + i * 12), perChunk: dv.getUint32(st.stsc.payload + 12 + i * 12) });
  // stco / co64 -> chunk offsets
  let chunkOffsets = [];
  if (st.co64) {
    const cnt = dv.getUint32(st.co64.payload + 4);
    for (let i = 0; i < cnt; i++) chunkOffsets.push(Number(dv.getBigUint64(st.co64.payload + 8 + i * 8)));
  } else if (st.stco) {
    const cnt = dv.getUint32(st.stco.payload + 4);
    for (let i = 0; i < cnt; i++) chunkOffsets.push(dv.getUint32(st.stco.payload + 8 + i * 4));
  } else throw new Error("no chunk offsets (stco/co64)");
  // sample table in file (decode) order
  const samples = [];
  let sIdx = 0;
  for (let c = 0; c < chunkOffsets.length && sIdx < sampleCount; c++) {
    const chunkNo = c + 1;
    let per = 1;
    for (const e of scEntries) if (e.firstChunk <= chunkNo) per = e.perChunk;
    let off = chunkOffsets[c];
    for (let j = 0; j < per && sIdx < sampleCount; j++, sIdx++) {
      samples.push({
        offset: off, size: sizes[sIdx],
        ctsUs: Math.round((dts[sIdx] + cto[sIdx]) * 1e6 / timescale),
        durUs: Math.round(dur[sIdx] * 1e6 / timescale),
        key: sync ? sync.has(sIdx + 1) : true,
      });
      off += sizes[sIdx];
    }
  }
  if (!samples.length) throw new Error("no video samples in this file");
  // Duration: the mdhd track duration is authoritative. With B-frames
  // the composition times are not monotonic in decode order, so the
  // latest (cts+dur) overestimates — this HEVC file reads 2.25s that
  // way vs the real 2.008s. Fall back to it only if mdhd has no duration.
  const mdhdDurRaw = mdhdVer === 1 ? Number(dv.getBigUint64(video.mdhd.payload + 24)) : dv.getUint32(video.mdhd.payload + 16);
  let duration = mdhdDurRaw / timescale;
  if (!duration || !isFinite(duration)) {
    duration = 0;
    for (const s of samples) duration = Math.max(duration, (s.ctsUs + s.durUs) / 1e6);
  }
  return { duration, width, height, codec, codecName, description, samples };
}
// ---- Minimal MP4 muxer (v1.63) ----
// The inverse of demuxMP4: takes H.264 (AVC) samples and writes a
// non-fragmented MP4 (ftyp + moov + mdat) with EXACT timing — every
// sample lasts exactly 1/fps seconds, so the file is exactly
// total/fps = the source duration. Used by the VideoEncoder recorder,
// because MediaRecorder + canvas.captureStream(0) + requestFrame
// timestamps each frame by when it was produced (the dream takes
// ~0.3s a frame): at 1024 the same 749 frames came out as a 223s
// slow-motion video (v1.61 file: 716 frames, 223.5s, ~0.3s between
// frames), and at 4K the recorder re-times them to ~24 fps, so the
// same run comes out 31s (v1.62). The code never set the output
// timing, so the duration was whatever the recorder did. Here the
// timing is set explicitly. The file is non-fragmented (moov before
// mdat), so this app's demuxer can read it back — a saved video can
// be dreamed again.
// v1.64: the VideoEncoder's decoderConfig.description is a BufferSource,
// and Chrome delivers it as an ArrayBuffer — which has NO .length
// property (it has .byteLength). v1.63 stored it directly as veConfig, so
// muxMP4Video's `!avcC.length` check threw "no H.264 config (avcC)" even
// though the config bytes were present — the run dreamed everything and
// then failed at write time. Normalize any BufferSource to a Uint8Array.
function toU8(buf) {
  if (!buf) return null;
  if (buf instanceof Uint8Array) return buf;
  if (buf instanceof ArrayBuffer) return new Uint8Array(buf);
  if (ArrayBuffer.isView(buf)) return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  return null;
}
// v1.64: build an avcC from the SPS/PPS NAL units in a keyframe chunk.
// VideoEncoder emits avc1 chunks as length-prefixed (AVCC) NAL units, and
// a keyframe normally starts with SPS (type 7) and PPS (type 8). Used when
// the encoder's metadata description is missing or empty, so the run does
// not depend on the platform surfacing the config in metadata.
function avcCFromKeyChunk(data) {
  if (!data || !data.length) return null;
  let sps = null, pps = null;
  // AVCC: 4-byte big-endian length + NAL. Annex B fallback: 00 00 01 / 00 00 00 01 start codes.
  const isAnnexB = (data.length >= 4 && data[0] === 0 && data[1] === 0 && data[2] === 0 && data[3] === 1) ||
    (data.length >= 3 && data[0] === 0 && data[1] === 0 && data[2] === 1);
  if (isAnnexB) {
    let at = 0;
    while (at + 3 < data.length) {
      let sc = -1, scLen = 0;
      if (data[at] === 0 && data[at + 1] === 0 && data[at + 2] === 1) { sc = at; scLen = 3; }
      else if (at + 4 < data.length && data[at] === 0 && data[at + 1] === 0 && data[at + 2] === 0 && data[at + 3] === 1) { sc = at; scLen = 4; }
      if (sc < 0) { at++; continue; }
      let end = sc + scLen;
      while (end + 2 < data.length && !(data[end] === 0 && data[end + 1] === 0 && (data[end + 2] === 1 || (end + 3 < data.length && data[end + 2] === 0 && data[end + 3] === 1)))) end++;
      const nal = data.subarray(sc + scLen, end);
      if (nal.length) {
        const type = nal[0] & 0x1f;
        if (type === 7 && !sps) sps = nal;
        else if (type === 8 && !pps) pps = nal;
      }
      if (sps && pps) break;
      at = end;
    }
  } else {
    let at = 0;
    while (at + 4 <= data.length) {
      const len = (data[at] << 24) | (data[at + 1] << 16) | (data[at + 2] << 8) | data[at + 3];
      if (len <= 0 || at + 4 + len > data.length) break;
      const nal = data.subarray(at + 4, at + 4 + len);
      const type = nal[0] & 0x1f;
      if (type === 7 && !sps) sps = nal;
      else if (type === 8 && !pps) pps = nal;
      if (sps && pps) break;
      at += 4 + len;
    }
  }
  if (!sps || !pps) return null;
  // avcC: version(1) profile compat level 0xFF numSPS(0xE0|n) sps* numPPS pps*
  const out = new Uint8Array(11 + sps.length + pps.length);
  let o = 0;
  out[o++] = 1; out[o++] = sps[1]; out[o++] = sps[2]; out[o++] = sps[3];
  out[o++] = 0xFF;   // 6 reserved bits + lengthSizeMinusOne(3) = 4-byte NAL lengths
  out[o++] = 0xE1;   // 3 reserved bits + numSPS(1)
  out[o++] = (sps.length >> 8) & 0xff; out[o++] = sps.length & 0xff;
  out.set(sps, o); o += sps.length;
  out[o++] = 1;   // numPPS
  out[o++] = (pps.length >> 8) & 0xff; out[o++] = pps.length & 0xff;
  out.set(pps, o); o += pps.length;
  return out;
}
function muxMP4Video(samples, avcC, width, height, fps) {
  const n = samples.length;
  if (!n) throw new Error("no frames to write");
  avcC = toU8(avcC);   // v1.64: tolerate an ArrayBuffer config (Chrome's description type) — .length is only on the view
  if (!avcC || !avcC.length) throw new Error("no H.264 config (avcC)");
  // Timescale 90000: 90000/fps is exact for fps 8/12/16 (11250/7500/5625).
  const TS = 90000, DELTA = Math.round(TS / fps);
  if (DELTA * fps !== TS) throw new Error("unsupported fps " + fps);
  const u32 = v => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v >>> 0); return b; };
  const u16 = v => { const b = new Uint8Array(2); new DataView(b.buffer).setUint16(0, v & 0xffff); return b; };
  const str = s => { const b = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); return b; };
  const zeros = k => new Uint8Array(k);
  const cat = parts => { let len = 0; for (const p of parts) len += p.length; const o = new Uint8Array(len); let at = 0; for (const p of parts) { o.set(p, at); at += p.length; } return o; };
  const box = (type, ...payload) => { const body = cat(payload); return cat([u32(body.length + 8), str(type), body]); };
  const verFlags = (v, f) => new Uint8Array([v, (f >> 16) & 255, (f >> 8) & 255, f & 255]);
  const mvhdDur = Math.round(n * 1000 / fps);   // ms
  const mdhdDur = n * DELTA;   // in TS units
  const keyIdx = [];
  for (let i = 0; i < n; i++) if (samples[i].key) keyIdx.push(i + 1);   // 1-based
  if (!keyIdx.length) keyIdx.push(1);
  const sizes = samples.map(s => s.data.length);
  const ftyp = box("ftyp", str("isom"), u32(512), str("isom"), str("iso2"), str("avc1"), str("mp41"));
  const mvhd = box("mvhd", verFlags(0, 0), u32(0), u32(0), u32(1000), u32(mvhdDur), u32(0x00010000), u16(0x0100), u16(0), zeros(8),
    u32(0x00010000), u32(0), u32(0), u32(0), u32(0x00010000), u32(0), u32(0), u32(0), u32(0x40000000), zeros(24), u32(2));
  const tkhd = box("tkhd", verFlags(0, 7), u32(0), u32(0), u32(1), u32(0), u32(mvhdDur), zeros(8),
    u16(0), u16(0), u16(0), u16(0),
    u32(0x00010000), u32(0), u32(0), u32(0), u32(0x00010000), u32(0), u32(0), u32(0), u32(0x40000000),
    u32(width << 16), u32(height << 16));
  const mdhd = box("mdhd", verFlags(0, 0), u32(0), u32(0), u32(TS), u32(mdhdDur), u16(0x55c4), u16(0));
  const hdlr = box("hdlr", verFlags(0, 0), u32(0), str("vide"), zeros(12), str("VideoHandler\0"));
  const vmhd = box("vmhd", verFlags(0, 1), u16(0), u16(0), u16(0), u16(0));
  const dinf = box("dinf", box("dref", verFlags(0, 0), u32(1), box("url ", verFlags(0, 1))));
  // avc1 sample entry: 78 fixed bytes, then the avcC box. (demuxMP4
  // reads width at +24, height at +26, and avcC at +78.)
  const avc1Fixed = cat([zeros(6), u16(1), u16(0), u16(0), zeros(12), u16(width), u16(height),
    u32(0x00480000), u32(0x00480000), u32(0), u16(1), zeros(32), u16(0x0018), u16(0xffff)]);
  const avc1 = box("avc1", avc1Fixed, box("avcC", avcC));
  const stsd = box("stsd", verFlags(0, 0), u32(1), avc1);
  const stts = box("stts", verFlags(0, 0), u32(1), u32(n), u32(DELTA));
  const stss = box("stss", verFlags(0, 0), u32(keyIdx.length), ...keyIdx.map(k => u32(k)));
  const stsc = box("stsc", verFlags(0, 0), u32(1), u32(1), u32(n), u32(1));
  const stsz = box("stsz", verFlags(0, 0), u32(0), u32(n), ...sizes.map(s => u32(s)));
  // stco needs the mdat data offset = ftyp + moov + mdat header. moov's
  // size does not depend on the offset value (it is a fixed 4 bytes), so
  // build moov once with a placeholder, measure it, then rebuild with
  // the real offset.
  const buildMoov = chunkOff => {
    const stco = box("stco", verFlags(0, 0), u32(1), u32(chunkOff));
    const stbl = box("stbl", stsd, stts, stss, stsc, stsz, stco);
    const minf = box("minf", vmhd, dinf, stbl);
    const mdia = box("mdia", mdhd, hdlr, minf);
    const trak = box("trak", tkhd, mdia);
    return box("moov", mvhd, trak);
  };
  let moov = buildMoov(0);
  const chunkOff = ftyp.length + moov.length + 8;
  moov = buildMoov(chunkOff);
  const mdatHead = cat([u32(8 + sizes.reduce((a, b) => a + b, 0)), str("mdat")]);
  return new Blob([ftyp, moov, mdatHead, ...samples.map(s => s.data)], { type: "video/mp4" });
}
// A lost WebGPU device kills every frame after it — the old code just
// reported "Video dream failed" and stopped. Instead, drop the dead
// backend and carry on on WASM/CPU, retrying the same frame.
function isGpuError(e) {
  const m = String((e && e.message) || e || "").toLowerCase();
  return (m.includes("device") && m.includes("lost")) || m.includes("mapasync") || m.includes("gpubuffer") || m.includes("context lost") || m.includes("out of memory") || m.includes("allocation failed");
}
async function switchVideoBackend() {
  // Returns the backend name we landed on, or null if none works.
  for (const name of ["wasm", "cpu"]) {
    try { await activate(name); return name; } catch (e) { console.warn("video backend " + name + " failed", e); }
  }
  return null;
}
// A backend switch must not hang the run: if the fallback engine does
// not load within the timeout, it counts as failed. (The underlying
// switch keeps running in the background; the run aborts, and the next
// run re-activates cleanly via ensureReady.)
async function switchVideoBackendTimed(ms) {
  let timer = null;
  try {
    return await Promise.race([
      switchVideoBackend(),
      new Promise(res => { timer = setTimeout(() => res(null), ms); })
    ]);
  } finally { if (timer) clearTimeout(timer); }
}
async function generateVideo() {
  if (videoRunning) return;
  if (busy || zooming) { setVideoStatus("Finish the photo dream first, then generate.", true); return; }
  if (!videoMeta) { setVideoStatus("Pick a video first — it takes a second to read.", true); return; }
  const vModel = MODELS[videoModelKey()] || MODELS.v1;
  setVideoStatus("Loading " + vModel.label + "…");
  if (!await activateVideoModel()) { setVideoStatus("Could not load " + vModel.label + " — try another model.", true); restorePhotoModel(); return; }
  // v1.60: frames come from WebCodecs VideoDecoder, not a <video> element.
  // Seeking/rVFC is proven broken on this device (v1.59: every seek fell
  // back to the playhead and drew the intro frame at 2.1s), so there is no
  // seek path at all in this function — no seekTo, no seekFrame, no
  // <video>, no rVFC. If this device has no VideoDecoder, the run fails
  // with a clear message instead of dreaming stale frames.
  if (typeof VideoDecoder !== "function") {
    setVideoStatus("This device can't decode video this way (WebCodecs VideoDecoder is missing) — try updating the app / WebView.", true);
    restorePhotoModel(); return;
  }
  let fileBuf = null;
  try {
    if (videoFile) fileBuf = await videoFile.arrayBuffer();
    else if (videoSrc) fileBuf = await (await fetch(videoSrc)).arrayBuffer();
  } catch (e) { fileBuf = null; }
  if (!fileBuf || !fileBuf.byteLength) { setVideoStatus("Could not read that video file.", true); restorePhotoModel(); return; }
  let demux;
  try { demux = demuxMP4(fileBuf); }
  catch (e) { setVideoStatus("Could not read that video: " + (e && e.message || e) + ".", true); restorePhotoModel(); return; }
  // Only offer what this device can actually decode: ask VideoDecoder
  // whether the file's codec is supported before configuring it. HEVC
  // is hardware on the S25 Ultra; VP9/AV1 depend on the WebView. If the
  // check itself fails, let configure() decide — its error names the codec.
  let codecSupported = true;
  try {
    if (typeof VideoDecoder.isConfigSupported === "function") {
      const r = await VideoDecoder.isConfigSupported({ codec: demux.codec, description: demux.description, codedWidth: demux.width, codedHeight: demux.height });
      codecSupported = !!(r && r.supported);
    }
  } catch (e) { codecSupported = true; }
  if (!codecSupported) {
    setVideoStatus("This video is " + demux.codecName + " (" + demux.codec + "), which this device cannot decode. Try an H.264 MP4.", true);
    restorePhotoModel(); return;
  }
  videoRunning = true; updateVideoUI();
  let rec = null, done = null, prev = null, mime = null, vtrack = null, manualFrames = false;
  let veEncoder = null, veConfig = null, veError = null, recorderKind = null, recKindNote = "";   // v1.63: VideoEncoder recorder state (declared here so the catch/finally can close it)
  const veChunks = [];   // VideoEncoder output: {data: Uint8Array, key: bool} in presentation order
  let dreamed = 0, skipped = 0, engineSwitched = null, videoDone = false;
  let decoder = null, held = null, heldTsUs = -1, decodedFrames = 0, reusedFrames = 0;
  let stopReason = null, decoderRecoveries = 0, badSamples = 0;   // v1.61: stopReason is WHY the run stopped — the post-loop status must never hide it again (v1.60 overwrote the decoder-error message with a bare "Stopped")
  // v1.67: Ramp up state (declared here so the finally block can restore
  // the sliders and free the noise seed). The sliders are captured as the
  // START values once at run start; the ramp climbs from those stored
  // values to the sliders' own maximums — never read from the sliders
  // once they are being driven, otherwise the ramp would compound on
  // itself. (v1.70: Coby — "if I set the steps or dream strength thats
  // where the ramp starts, right? thats how I want it." The v1.66–v1.69
  // reading had the sliders as the END; they are the START now.)
  let rampOnRun = false, rampStepsStart = 1, rampLrStart = 0.01, rampStepsTop = 5, rampLrTop = 0.10, rampNoise = null;
  let tileScratch = {};   // v1.73: reused across frames by the tiled dream path (declared here so the finally can free it)
  let tileNote = "";      // v1.73: " · tiled NxM" when the run tiled — named in the Done status
  // v1.67: drive the Steps / Dream strength sliders so the ramp is visible
  // while it runs (Coby: "make it so i can see the sliders go up").
  // v1.68: snap lr to the slider's 0.005 step grid before setting .value
  // (a raw float can be rejected/ignored by the range input), dispatch
  // an input event so the browser treats it as a real change and repaints
  // the thumb, and force a reflow. The label is also set directly.
  function driveRampSliders(steps, lr) {
    try {
      const sEl = $("videoSteps"), lEl = $("videoLr");
      if (sEl) { sEl.value = String(steps); const v = $("videoStepsVal"); if (v) v.textContent = String(steps); try { sEl.dispatchEvent(new Event("input", { bubbles: true })); void sEl.offsetWidth; } catch (e) {} }
      if (lEl) { const lrSnap = Math.round(lr / 0.005) * 0.005; lEl.value = String(lrSnap); const v = $("videoLrVal"); if (v) v.textContent = lrSnap.toFixed(3); try { lEl.dispatchEvent(new Event("input", { bubbles: true })); void lEl.offsetWidth; } catch (e) {} }
    } catch (e) {}
  }
  function restoreRampSliders() {
    if (!rampOnRun) return;
    try {
      const sEl = $("videoSteps"), lEl = $("videoLr");
      if (sEl) { sEl.value = String(rampStepsStart); const v = $("videoStepsVal"); if (v) v.textContent = String(rampStepsStart); try { sEl.dispatchEvent(new Event("input", { bubbles: true })); } catch (e) {} }
      if (lEl) { const lrSnap = Math.round(rampLrStart / 0.005) * 0.005; lEl.value = String(lrSnap); const v = $("videoLrVal"); if (v) v.textContent = lrSnap.toFixed(3); try { lEl.dispatchEvent(new Event("input", { bubbles: true })); } catch (e) {} }
    } catch (e) {}
  }
  try {
    const dur = demux.duration;
    const fps = parseInt($("videoFps").value, 10) || 12;
    const total = Math.max(1, Math.floor(dur * fps));   // every frame of the video, however long — no length cap, no frame cap
    let vd = videoDims(demux.width, demux.height);   // output size (2160 = 4K by default)
    let dd = videoDreamDims(vd);   // dream size (capped) — the output stays vd; let so the fallback can shrink it further
    // v1.67: capture the ramp start values + one fixed noise seed per run.
    // Flat content has no gradient for the dream to grab (v1.67 diagnosis:
    // a near-binary label sits at the clip bounds, so the per-frame dream
    // is ~0.0005 and the weave can only double it — invisible). The seed
    // gives flat areas texture to amplify; one pattern per run so the
    // weave accumulates it coherently instead of averaging random noise away.
    // v1.70: the sliders set where the ramp STARTS. The ramp climbs from
    // the captured start values to the sliders' own maximums (read from
    // the slider elements: Steps max 5, Strength max 0.100) by the last
    // frame. The v1.69 effective-end logic is gone — the sliders always
    // climb to the slider max now, which is above the start unless the
    // start is already at max (then the ramp has nowhere to climb and
    // the sliders correctly stay at max). The user's set values are the
    // ones restored after the run (restoreRampSliders).
    // v1.78: the sliders ARE the start — start = the slider value,
    // always. The HTML defaults are the minimums (Steps 1, Strength
    // 0.010), so an untouched slider starts at the lowest; a slider he
    // set to 2 starts at 2. No flag, no minimum-fallback for untouched.
    rampOnRun = videoRampOn();
    rampStepsStart = parseInt($("videoSteps").value, 10) || 1;
    rampLrStart = videoLr();
    rampStepsTop = ($("videoSteps") && parseInt($("videoSteps").max, 10)) || 5;
    rampLrTop = ($("videoLr") && parseFloat($("videoLr").max)) || 0.10;
    if (rampOnRun) { try { if (rampNoise) rampNoise.dispose(); } catch (e) {} rampNoise = tf.randomUniform([1, dd.h, dd.w, 3], -1, 1); }
    const dreamCv = document.createElement("canvas"); dreamCv.width = dd.w; dreamCv.height = dd.h;
    const outCv = document.createElement("canvas"); outCv.width = vd.w; outCv.height = vd.h;   // what the recorder + preview show
    let outNote = "";   // set if 4K recording fell back to 1024 — the end status must say so
    let recError = null;   // a mid-run encoder error — surfaced in the end status, never silent
    // 4K needs a much higher bitrate than 1024 or it comes out blocky:
    // 30 Mbps at the 4K size, 8 Mbps below it.
    function videoBpsFor(d) { return Math.min(d.w, d.h) >= 2160 ? 30000000 : 8000000; }
    // ---- Output recorder (v1.63) ----
    // The output video's timing must be the SOURCE's timing at the
    // Process fps: frame i plays at i/fps seconds, so the saved video is
    // exactly total/fps = the source duration. MediaRecorder +
    // canvas.captureStream(0) + requestFrame cannot do this: it
    // timestamps each frame by when it was produced (the dream takes
    // ~0.3s a frame), so at 1024 the same 749 frames came out as a 223s
    // slow-motion video (v1.61 file: 716 frames, 223.5s, ~0.3s between
    // frames), and at 4K the recorder re-times them to ~24 fps, so the
    // same run comes out 31s (v1.62). The code never set the output
    // timing, so the duration was whatever the recorder did.
    // WebCodecs VideoEncoder takes an explicit timestamp per frame, so
    // the output is exactly `total` frames at exactly the Process fps —
    // regardless of how slow the dream is or how big the canvas is.
    // MediaRecorder is kept as a fallback if VideoEncoder is unavailable
    // on this device (its output duration then follows the recorder, as
    // before — the end status says which recorder was used).
    const chunks = [];
    // Pick an H.264 codec this device's VideoEncoder supports. High
    // profile level 5.1 handles 4K; fall back through Main and Baseline.
    async function pickVeCodec() {
      if (typeof VideoEncoder === "undefined" || typeof VideoEncoder.isConfigSupported !== "function") return null;
      for (const c of ["avc1.640033", "avc1.4d0033", "avc1.4d0028", "avc1.42001f"]) {
        try {
          const r = await VideoEncoder.isConfigSupported({ codec: c, width: vd.w, height: vd.h, bitrate: videoBpsFor(vd), framerate: fps });
          if (r && r.supported) return c;
        } catch (e) {}
      }
      return null;
    }
    function startVideoEncoder(codec) {
      veChunks.length = 0; veConfig = null; veError = null;
      const enc = new VideoEncoder({
        output: (chunk, meta) => {
          // v1.64: capture the avcC robustly. The description is a
          // BufferSource (Chrome gives an ArrayBuffer — no .length), so
          // normalize it; only a NON-EMPTY config counts, and a later
          // keyframe may still supply one, so keep looking until one is
          // found. If the metadata has none, build it from the
          // keyframe's own SPS/PPS.
          if (!veConfig || !veConfig.length) {
            if (meta && meta.decoderConfig && meta.decoderConfig.description) {
              const d = toU8(meta.decoderConfig.description);
              if (d && d.length) veConfig = d;
            }
          }
          try {
            const buf = new Uint8Array(chunk.byteLength);
            chunk.copyTo(buf);
            const isKey = chunk.type === "key";
            veChunks.push({ data: buf, key: isKey });
            if (isKey && (!veConfig || !veConfig.length)) {
              const built = avcCFromKeyChunk(buf);
              if (built) veConfig = built;
            }
          } catch (e) { veError = veError || e; }
        },
        error: (e) => { veError = e; }
      });
      enc.configure({ codec, width: vd.w, height: vd.h, bitrate: videoBpsFor(vd), framerate: fps, latencyMode: "quality" });
      return enc;
    }
    // Present the frame currently on outCv to the recorder, at its OUTPUT
    // time: frame i plays at i/fps seconds. VideoEncoder: encode it with
    // that exact timestamp. MediaRecorder: requestFrame (manual) or sleep.
    async function presentFrame(i) {
      if (recorderKind === "ve") {
        const tsUs = Math.round(i * 1e6 / fps);
        let vf = null;
        try {
          vf = new VideoFrame(outCv, { timestamp: tsUs, duration: Math.round(1e6 / fps) });
          const needKey = (i === 0) || (i % Math.max(1, fps * 2) === 0);   // keyframe at the start and every ~2s
          veEncoder.encode(vf, { keyFrame: needKey });
        } catch (e) { veError = veError || e; throw e; }
        finally { if (vf) { try { vf.close(); } catch (e) {} } }
        // Backpressure: the hardware encoder eats 4K at 30 fps, so a
        // ~3 fps input never piles up — but if it ever does, wait for it
        // to drain instead of queuing unboundedly.
        if (veEncoder.encodeQueueSize > 16) await veEncoder.flush();
        return;
      }
      if (manualFrames) { vtrack.requestFrame(); await sleep(60); }
      else await sleep(1000 / fps);
    }
    function makeRecorder() {
      const s = outCv.captureStream(0);
      const vt = s.getVideoTracks()[0];
      const mf = !!(vt && typeof vt.requestFrame === "function");
      const st = mf ? s : outCv.captureStream(fps);
      const m = videoMime();   // MP4 first (video/mp4;codecs=avc1), then WebM — whatever this WebView supports
      const r = new MediaRecorder(st, m ? { mimeType: m, videoBitsPerSecond: videoBpsFor(vd) } : { videoBitsPerSecond: videoBpsFor(vd) });
      r.ondataavailable = e => { if (e.data && e.data.size) chunks.push(e.data); };
      r.onerror = e => { recError = e; };
      const d = new Promise(res => r.onstop = res);
      r.start(200);
      return { rec: r, done: d, vtrack: vt, manualFrames: mf, mime: m };
    }
    // v1.64: after the first VideoEncoder frame, prove the encoder gave
    // us a usable H.264 config (avcC) BEFORE dreaming the rest. v1.63
    // only discovered a missing config at mux time, failing the whole
    // run after all the dreaming. If there is still no usable avcC
    // after flushing the first keyframe, abandon VideoEncoder and fall
    // back to MediaRecorder (which worked in v1.62) — re-presenting
    // frame 0 — instead of failing at the end. Returns true if
    // recording can continue, false if no recorder could be started.
    async function ensureVeAvcC() {
      if (recorderKind !== "ve") return true;
      try { if (veEncoder) await veEncoder.flush(); } catch (e) {}
      if (!veChunks.length) await sleep(300);   // let the output callback land
      if (veConfig && veConfig.length) return true;
      for (const c of veChunks) {
        if (!c.key) continue;
        const built = avcCFromKeyChunk(c.data);
        if (built) { veConfig = built; return true; }
      }
      // No config from this encoder — fall back to MediaRecorder now,
      // while only frame 0 has been encoded. Discard the VE chunks;
      // frame 0 is re-recorded via MediaRecorder below. recorderKind
      // stays "ve" until MediaRecorder actually starts.
      try { if (veEncoder) veEncoder.close(); } catch (e) {}
      veEncoder = null; veChunks.length = 0; veConfig = null; veError = null;
      const tryMr = () => {
        const made = makeRecorder();
        rec = made.rec; done = made.done; vtrack = made.vtrack; manualFrames = made.manualFrames; mime = made.mime;
        recorderKind = "mr"; recKindNote = " · MediaRecorder (VideoEncoder gave no H.264 config)";
        return true;
      };
      let mrStarted = false;
      try { mrStarted = tryMr(); } catch (e) {
        if (Math.min(vd.w, vd.h) >= 2160) {
          vd = videoDims(demux.width, demux.height, 1024);
          dd = videoDreamDims(vd);
          dreamCv.width = dd.w; dreamCv.height = dd.h;
          outCv.width = vd.w; outCv.height = vd.h;
          pv.width = vd.w; pv.height = vd.h;
          outNote = " · 4K recording is not supported on this device, fell back to 1024";
          try { mrStarted = tryMr(); } catch (e2) { mrStarted = false; }
        }
      }
      if (!mrStarted) return false;
      try { await presentFrame(0); } catch (e) {}   // re-record frame 0 (still on outCv) via MediaRecorder
      return true;
    }
    // Try VideoEncoder first (exact timing). If it cannot be configured
    // at this size, shrink the OUTPUT to 1024 and retry once (the dream
    // stays 192; only the output shrinks). If that fails too, fall back
    // to MediaRecorder (its 4K fallback below still applies).
    const veAvailable = (typeof VideoEncoder === "function");
    let veCodec = null;
    if (veAvailable) { try { veCodec = await pickVeCodec(); } catch (e) {} }
    if (veCodec) {
      try { veEncoder = startVideoEncoder(veCodec); recorderKind = "ve"; recKindNote = " · VideoEncoder (" + veCodec + ")"; }
      catch (e) { veEncoder = null; }
    }
    if (recorderKind !== "ve" && veAvailable && Math.min(vd.w, vd.h) >= 2160) {
      vd = videoDims(demux.width, demux.height, 1024);
      dd = videoDreamDims(vd);
      dreamCv.width = dd.w; dreamCv.height = dd.h;
      outCv.width = vd.w; outCv.height = vd.h;
      outNote = " · 4K recording is not supported on this device, fell back to 1024";
      try { veCodec = await pickVeCodec(); } catch (e) { veCodec = null; }
      if (veCodec) {
        try { veEncoder = startVideoEncoder(veCodec); recorderKind = "ve"; recKindNote = " · VideoEncoder (" + veCodec + ")"; }
        catch (e) { veEncoder = null; }
      }
    }
    if (recorderKind !== "ve") {
      try {
        const made = makeRecorder();
        rec = made.rec; done = made.done; vtrack = made.vtrack; manualFrames = made.manualFrames; mime = made.mime;
        recorderKind = "mr"; recKindNote = " · MediaRecorder";
      } catch (e) {
        if (Math.min(vd.w, vd.h) >= 2160) {
          // 4K recording failed on this device — fall back to 1024 with a
          // clear note, instead of failing the run. (The dream keeps its
          // 192 size; only the output canvas shrinks.)
          vd = videoDims(demux.width, demux.height, 1024);
          dd = videoDreamDims(vd);
          dreamCv.width = dd.w; dreamCv.height = dd.h;
          outCv.width = vd.w; outCv.height = vd.h;
          outNote = " · 4K recording is not supported on this device, fell back to 1024";
          try {
            const made = makeRecorder();
            rec = made.rec; done = made.done; vtrack = made.vtrack; manualFrames = made.manualFrames; mime = made.mime;
            recorderKind = "mr"; recKindNote = " · MediaRecorder";
          } catch (e2) {
            setVideoStatus("Could not start the video recorder on this device: " + (e2 && e2.message || e2), true);
            return;
          }
        } else {
          setVideoStatus("Could not start the video recorder on this device: " + (e && e.message || e), true);
          return;
        }
      }
    }
    // Live preview: show each dreamed frame in the square as it is made, so
    // you watch the video build smoothly while it records. The finished
    // video takes over the square at the end.
    const pv = $("vidPrev"); pv.width = vd.w; pv.height = vd.h; pv.style.display = "block";
    $("vidOut").style.display = "none";
    $("videoHint").textContent = "Dreaming… watch it build here.";
    let vHueTotal = 0;   // vHueTotal: the video's own cumulative hue for Color spin (separate from the Camera's)
    // ---- VideoDecoder driver ----
    // The decoder hands us VideoFrames in presentation order, each with an
    // exact timestamp. No <video>, no compositor, no seeking, no rVFC — a
    // stale frame is impossible: a frame IS the decoded data at its time.
    // Samples are fed in file (decode) order with backpressure
    // (decodeQueueSize < 8), so decoded frames never pile up in memory.
    const samples = demux.samples;
    let sampleIdx = 0, pending = [], waiter = null, decoderDone = false, decoderError = null, flushStarted = false;
    const wake = () => { if (waiter) { const w = waiter; waiter = null; w(); } };
    // Make (and configure) a fresh VideoDecoder. Returns true on success.
    // Used once at the start and again after a decoder error, so one bad
    // chunk or decoder hiccup does not kill the whole run (v1.61: v1.60
    // died after 2 frames on the first decoder error, with no recovery).
    function startDecoder() {
      try {
        decoder = new VideoDecoder({
          output: (frame) => { decodedFrames++; pending.push(frame); wake(); },
          error: (e) => { decoderError = e; wake(); },
        });
        decoder.configure({ codec: demux.codec, description: demux.description, codedWidth: demux.width, codedHeight: demux.height, optimizeForLatency: true });
        return true;
      } catch (e) { decoderError = e; return false; }
    }
    // A decoder error must not kill the run: close the failed decoder,
    // make a fresh one, and resume feeding from the key frame with the
    // largest presentation time at or before the target (delta frames
    // after it need that key). The existing nextTargetFrame discards
    // frames before the target, so the run picks up at the right place.
    // Bounded: after 3 recoveries the file/decoder is fundamentally
    // broken and the run fails with a clear reason instead of looping.
    function recoverDecoder(targetUs) {
      if (decoderRecoveries >= 3) return false;
      decoderRecoveries++;
      try { if (decoder) decoder.close(); } catch (e) {}
      decoder = null;
      for (const f of pending) { try { f.close(); } catch (e) {} }
      pending.length = 0;
      if (held) { try { held.close(); } catch (e) {} held = null; heldTsUs = -1; }
      waiter = null; decoderDone = false; decoderError = null; flushStarted = false;
      let resumeIdx = 0, bestCts = -1;
      for (let k = 0; k < samples.length; k++) {
        if (samples[k].key && samples[k].ctsUs <= targetUs && samples[k].ctsUs > bestCts) { resumeIdx = k; bestCts = samples[k].ctsUs; }
      }
      sampleIdx = resumeIdx;
      return startDecoder();
    }
    if (!startDecoder()) {
      setVideoStatus("This device can't decode that video (" + (decoderError && decoderError.message || decoderError) + ") — try an MP4 (H.264) file.", true);
      return;
    }
    const feedSome = () => {
      if (decoderError) return;
      try {
        while (sampleIdx < samples.length && decoder.decodeQueueSize < 8) {
          const s = samples[sampleIdx++];
          // Never feed a sample whose bytes are outside the file — a bad
          // offset/size would hand the decoder garbage and kill the run.
          // Skip it (counted) instead of erroring the decoder.
          if (!(s.size > 0) || s.offset < 0 || s.offset + s.size > fileBuf.byteLength) { badSamples++; continue; }
          decoder.decode(new EncodedVideoChunk({ type: s.key ? "key" : "delta", timestamp: s.ctsUs, duration: s.durUs, data: new Uint8Array(fileBuf, s.offset, s.size) }));
        }
      } catch (e) { decoderError = decoderError || e; wake(); }
    };
    // Pump one decoded frame out of the decoder (presentation order), or
    // null when the stream is exhausted / errored / stopped.
    async function pumpFrame() {
      const deadline = Date.now() + 30000;
      while (videoRunning) {
        if (decoderError) return null;
        if (pending.length) return pending.shift();
        if (decoderDone) return null;
        if (sampleIdx < samples.length) {
          feedSome();
          if (pending.length) continue;
          await new Promise(r => { waiter = r; setTimeout(() => { if (waiter === r) { waiter = null; r(); } }, 200); });
          if (Date.now() > deadline) { decoderError = new Error("the video decoder stalled"); return null; }
          continue;
        }
        // all samples fed — flush once, then drain
        if (!flushStarted) {
          flushStarted = true;
          decoder.flush().then(() => { decoderDone = true; wake(); }).catch(e => { decoderError = e; wake(); });
        }
        await new Promise(r => { waiter = r; setTimeout(() => { if (waiter === r) { waiter = null; r(); } }, 200); });
        if (Date.now() > deadline) { decoderError = new Error("the video decoder stalled"); return null; }
      }
      return null;
    }
    // The frame for the next target time: the first decoded frame at or
    // after it. The held frame is reused when it already is the first
    // frame at/after the target (native fps below the Process fps) — the
    // dream still advances on it via carryover.
    async function nextTargetFrame(targetUs) {
      if (held && heldTsUs >= targetUs) { reusedFrames++; return { frame: held, tsUs: heldTsUs }; }
      while (videoRunning) {
        const f = await pumpFrame();
        if (!f) return null;
        const ts = f.timestamp;
        if (ts < targetUs) { try { f.close(); } catch (e) {} continue; }
        if (held && held !== f) { try { held.close(); } catch (e) {} }
        held = f; heldTsUs = ts;
        return { frame: f, tsUs: ts };
      }
      return null;
    }
    // A skipped frame still shows and records: draw the decoded frame raw
    // to the display canvas + preview, so the video keeps playing and the
    // recording stays in sync instead of stalling. The preview is what
    // tells you the run is alive — it must move on every frame, dreamed
    // or skipped.
    async function showRawFrame(frame, idx) {
      try { dreamCv.getContext("2d").drawImage(frame, 0, 0, dd.w, dd.h); } catch (e) {}
      const dc = outCv.getContext("2d");
      dc.clearRect(0, 0, outCv.width, outCv.height);
      dc.drawImage(dreamCv, 0, 0, outCv.width, outCv.height);
      vHueTotal = (vHueTotal + (parseFloat($("videoHueSpin").value) || 0)) % 360;
      applyFx(outCv, videoFx(vHueTotal));
      pv.getContext("2d").drawImage(outCv, 0, 0, pv.width, pv.height);
      await presentFrame(idx);   // record this raw frame at its output time (idx/fps)
    }
    let i = 0, consecFail = 0, curSrcTime = null, rampSteps = null, rampLr = null;
    // Status helpers. The source time is the ACTUAL time in the video of
    // the frame being dreamed — the decoded frame's own timestamp from
    // the decoder, not a playhead — so the frame number and the video
    // time can be seen to agree. The engine is always shown, so a slow
    // run names its engine.
    const engLabel = () => { try { const b = tf.getBackend(); return b ? " · engine: " + (LABEL[b] || b) : ""; } catch (e) { return ""; } };
    const srcTimeNote = () => (curSrcTime != null ? " · " + curSrcTime.toFixed(1) + "s in the video" : "");
    // v1.66: Ramp up note — the percent is always current (p =
    // i/(total-1)); the steps/strength appear once this frame's dream
    // has been computed (null before that, so no stale values from the
    // previous frame are ever shown).
    const rampNote = () => {
      if (!videoRampOn()) return "";
      const pct = Math.round(videoRampP(i, total) * 100);
      return " · ramp " + pct + "%" + (rampSteps != null ? " · " + rampSteps + " steps · " + rampLr.toFixed(3) : "");
    };
    // The frame loop: every frame either dreams and advances, or fails
    // cleanly. A frame gets at most two attempts — the first, and one
    // retry after a one-time backend switch. EVERY failure counts (decode,
    // GPU, or anything else); five consecutive failures end the run with
    // a clear status. No infinite retry, no silent hang, no lying status.
    // A per-frame timer keeps the status live while a frame is slow, so
    // you can always tell it is working and which frame it is on.
    while (i < total && videoRunning) {
      await sleep(0);   // let the UI breathe between frames — one long dream used to freeze the app
      rampSteps = null; rampLr = null;   // v1.66: this frame's ramp values, set when the dream runs — never show the previous frame's
      const frameT0 = performance.now();
      const frameTimer = setInterval(() => {
        // Don't overwrite a switch/skip/fail message — those are the
        // diagnostics. Only refresh the "Dreaming video…" line itself.
        const cur = $("videoStatus").textContent || "";
        if (cur.indexOf("Dreaming video…") !== 0) return;
        const s = Math.round((performance.now() - frameT0) / 1000);
        setVideoStatus("Dreaming video… frame " + (i + 1) + " of " + total + srcTimeNote() + " · " + s + "s on this frame" + engLabel() + rampNote());
      }, 2000);
      try {
        setVideoStatus("Dreaming video… frame " + (i + 1) + " of " + total + srcTimeNote() + engLabel() + rampNote());
        const targetT = Math.min(i / fps, Math.max(0, dur - 0.001));
        // The decoder hands us the frame AT targetT — no seek, no
        // compositor, no stale frame possible.
        const got = await nextTargetFrame(Math.round(targetT * 1e6));
        if (!got) {
          // No frame for this target: the decoder is exhausted or errored.
          if (decoderError) {
            // The decoder died — recover it and retry this frame instead
            // of killing the run. One decoder hiccup must not stop a
            // 749-frame run (v1.60 died after 2 frames this way).
            const errMsg = decoderError && decoderError.message || decoderError;
            if (recoverDecoder(Math.round(targetT * 1e6))) {
              setVideoStatus("The video decoder hiccuped (" + errMsg + ") — recovering and carrying on…");
              consecFail = 0;
              continue;   // retry the same frame target with the fresh decoder
            }
            stopReason = "Video dream failed: the video decoder stopped (" + errMsg + ") and could not recover · decoded " + decodedFrames + " frames · sample " + sampleIdx + " of " + samples.length;
            setVideoStatus(stopReason, true);
            break;
          }
          if (++consecFail >= 5) { stopReason = "Video dream failed: no frame for target " + targetT.toFixed(2) + "s at frame " + (i + 1) + " — try a different file."; setVideoStatus(stopReason, true); break; }
          setVideoStatus("Frame " + (i + 1) + " has no decoded frame (target " + targetT.toFixed(2) + "s) — skipping it…");
          skipped++; i++; continue;
        }
        const frame = got.frame;
        curSrcTime = got.tsUs / 1e6;   // the ACTUAL time of the decoded frame — the decoder's timestamp, not a playhead
        // The frame is confirmed at curSrcTime — show the frame number and
        // the real video time together, so the run's position is never a mystery.
        setVideoStatus("Dreaming video… frame " + (i + 1) + " of " + total + srcTimeNote() + engLabel() + rampNote());
        dreamCv.getContext("2d").drawImage(frame, 0, 0, dd.w, dd.h);   // dream small; the output stays vd
        if (i === 0) { try { pv.getContext("2d").drawImage(dreamCv, 0, 0, pv.width, pv.height); } catch (e) {} }   // first frame visible right away
        let cur = null, frameOk = false, frameErr = null, abortMsg = null;
        for (let attempt = 0; attempt < 2 && !frameOk && videoRunning; attempt++) {
          try {
            cur = canvasToTensor(dreamCv);
            // v1.67: Ramp up values for this frame (from the stored start
            // values, never from the sliders once they are being driven).
            // v1.70: the sliders set the START; with the toggle on, the
            // dream climbs from the start values to the sliders' own
            // maximums (Steps 5, Strength 0.100) by the last frame,
            // evenly spaced (p = i/(total-1)); off = the sliders apply
            // to every frame, as before. On the fallback engine the dream
            // is much slower: one step, so the run can still finish.
            // Floor: never 0 steps / 0 lr — 1 step and the slider minimum
            // lr, so the weave seeds from frame 0 (v1.66 left 12 frames
            // at 0 steps on a 92-frame video and the weave had nothing
            // to amplify). v1.68: weave ramps 0.5 -> 0.7 (was 0.9),
            // boost 1x -> 2x (was 8x), and the noise seed is scaled to
            // the model's range, 0.02 x (P.hi - P.lo) (was a fixed ±5
            // for every model — 250% of v3's [-1,1] range, instant
            // static). The depth washout eases to 30% — together they
            // make flat content actually dream instead of saturating
            // at ~0.0005 (v1.67 diagnosis), without the v1.67 runaway
            // that pinned the frame to the clip bounds by mid-video.
            // v1.76: rp falls back to 0 when ramp is off, so steps/lr stay
            // at the captured start values (the sliders) on every frame.
            // (v1.70 flipped the sliders to START but left the fallback
            // at 1, which made ramp-off runs use the slider MAXIMUMS
            // 5 / 0.100 on every frame instead of the set values.)
            const rp = rampOnRun ? videoRampP(i, total) : 0;
            const steps = engineSwitched ? 1 : Math.max(1, Math.round(rampStepsStart + (rampStepsTop - rampStepsStart) * rp));
            const lr = Math.max(0.01, rampLrStart + (rampLrTop - rampLrStart) * rp);
            rampSteps = steps; rampLr = lr;
            if (rampOnRun) driveRampSliders(steps, lr);
            const weaveEff = rampOnRun ? 0.5 + 0.2 * rp : FRAME_WEAVE;
            const boostF = rampOnRun ? 1 + rp : 1;
            const noiseAmp = rampOnRun ? 0.02 * (P.hi - P.lo) * rp : 0;
            const depthAmt = parseFloat($("videoDepth").value) || 0;
            const depthEff = rampOnRun ? depthAmt * (1 - 0.7 * rp) : depthAmt;
            const pulseF = 1 + (parseFloat($("videoPulse").value) || 0) * Math.sin(i * 0.25);   // Pulse: the dream strength breathes, one full breath every ~25 frames
            // Carryover: blend in some of the previous dreamed frame, so
            // the dream builds up across frames and is visible (v1.56 fix).
            // prev is null on frame 0, disposed on a backend switch, and
            // untouched on a skipped frame.
            if (prev) {
              const w = tf.tidy(() => cur.mul(1 - weaveEff).add(prev.mul(weaveEff)));
              cur.dispose(); cur = w;
            }
            // Zoom path: the zoom is driven by frame position, so it actually
            // moves over the video. In & out oscillates (sine wave) a few
            // times; In only grows linearly to the amount by the end.
            const zPath = $("videoZoom").value;
            if (zPath !== "0") {
              const p = total > 1 ? i / (total - 1) : 0;
              const cycles = Math.max(1, Math.min(5, parseInt($("videoZoomCycles").value, 10) || 3));
              const maxMag = parseFloat($("videoZoomAmount").value) || 2;
              const mag = zPath === "inout" ? 1 + (maxMag - 1) * (1 - Math.cos(2 * Math.PI * cycles * p)) / 2 : 1 + (maxMag - 1) * p;
              const z = zoomCrop(cur, 1 / mag, 0, 0, 0); cur.dispose(); cur = z;
            }
            // v1.65: depth separation. The mask is computed from this
            // frame's pre-dream tensor (after weave + zoom, so it aligns
            // with the dream), and the dreamed frame is blended back
            // toward the pre-dream frame where the mask is low — edges
            // and objects get the full dream, flat sky gets a gentler
            // one, so objects stand out instead of everything getting
            // the same wash. Depth 0 turns it off (old behaviour).
            // v1.68 noise seed (ramp only): one fixed pattern per run,
            // added just before the dream so flat areas have texture to
            // amplify. Without it the label's flat white has no gradient
            // and the dream stays ~0.0005. The amp is scaled to the
            // model's range (0.02 x (P.hi - P.lo)) — v1.67 used a fixed
            // ±5 for every model, which is 250% of v3's [-1,1] range.
            if (noiseAmp > 0 && rampNoise) {
              const n = tf.tidy(() => cur.add(rampNoise.mul(noiseAmp)));
              cur.dispose(); cur = n;
            }
            // rawKeep is the base for the boost and the depth blend, so
            // it is kept whenever the boost is on even if depth is 0.
            // v1.68: it is taken AFTER the noise seed is added, so the
            // boost amplifies only the dream's change on top of the
            // noise, never the injected noise itself. v1.67 took it
            // before the noise, so boostF up to 8x amplified the noise
            // and the weave carried it forward — a runaway that pinned
            // the frame to the clip bounds (pure static) by mid-video.
            let rawKeep = null, dmask = null;
            if (depthEff > 0 || boostF > 1) rawKeep = cur.clone();
            if (depthEff > 0) dmask = videoDetailMask(cur);
            // v1.73: tiled dream for a free-size model when the dream
            // short side is above one tile (768). The pre-dream tensor
            // (weave + zoom + noise already applied) goes onto dreamCv,
            // dreamTiled dreams it tile by tile in place, and it comes
            // back as a tensor for the boost/depth blend. Fixed-size
            // models and small dreams take the single pass, as before.
            // (One extra 8-bit round-trip per frame: the canvas holds
            // uint8, so the dream delta is quantized to 1/255 steps —
            // the Photos stills already accept this and dream visibly.)
            if (!MODEL.fixedSize && Math.min(dd.w, dd.h) > VIDEO_TILE_SIZE) {
              await tensorToCanvas(cur, dreamCv);
              cur.dispose();
              const grid = await dreamTiled(dreamCv, lr * pulseF, steps, null, tileScratch, VIDEO_TILE_SIZE, VIDEO_TILE_OVERLAP);
              if (grid) tileNote = " · tiled " + grid.nx + "×" + grid.ny;
              cur = canvasToTensor(dreamCv);
            } else {
              cur = dreamSteps(cur, lr * pulseF, steps, true);
            }
            // v1.68 boost (ramp only): amplify the dream delta so the end
            // is crazy. The delta is measured from the noisy pre-dream
            // tensor (rawKeep, taken after the noise seed), so the boost
            // grows the dream, not the noise. Capped at 2x (was 8x) —
            // the v1.56 weave carryover is already the mechanism that
            // makes video dreams visible.
            if (boostF > 1 && rawKeep) {
              const b = tf.tidy(() => rawKeep.add(cur.sub(rawKeep).mul(boostF)).clipByValue(P.lo, P.hi));
              cur.dispose(); cur = b;
            }
            if (depthEff > 0 && rawKeep && dmask) {
              const blended = tf.tidy(() => {
                const vis = dmask.mul(depthEff).add(1 - depthEff);   // [1-depth, 1]: flat keeps 1-depth of the dream
                return cur.mul(vis).add(rawKeep.mul(tf.scalar(1).sub(vis)));
              });
              cur.dispose(); rawKeep.dispose(); dmask.dispose();
              cur = blended;
            } else {
              if (rawKeep) { try { rawKeep.dispose(); } catch (e) {} }
              if (dmask) { try { dmask.dispose(); } catch (e) {} }
            }
            await tensorToCanvas(cur, outCv);   // upscales the small dream onto the vd output canvas
            frameOk = true;
          } catch (e) {
            frameErr = e;
            if (cur) { try { cur.dispose(); } catch (err) {} cur = null; }
            if (isGpuError(e) && !engineSwitched && attempt === 0) {
              // The dream engine died — drop it and carry on on WASM/CPU.
              // One switch per run, then one retry of this frame.
              if (prev) { try { prev.dispose(); } catch (err) {} prev = null; }
              setVideoStatus("The dream engine was lost — switching to a slower engine and carrying on…");
              const nb = await switchVideoBackendTimed(45000);
              if (!nb) { ready = false; abortMsg = "Video dream failed: the dream engine was lost and no fallback engine works."; break; }
              engineSwitched = nb;
              consecFail = 0;
              // Fallback is much slower: dream smaller so the run can still finish.
              dd = videoDreamDims(vd, 128);
              if (rampOnRun) { try { if (rampNoise) rampNoise.dispose(); } catch (e) {} rampNoise = tf.randomUniform([1, dd.h, dd.w, 3], -1, 1); }   // v1.67: the noise seed must match the new dream size
              dreamCv.width = dd.w; dreamCv.height = dd.h;
              dreamCv.getContext("2d").drawImage(frame, 0, 0, dd.w, dd.h);   // re-draw the same decoded frame at the new size (resizing clears the canvas)
              // Back to a "Dreaming video…" status so the per-frame timer
              // resumes updating it (it leaves switch/skip/fail messages alone).
              setVideoStatus("Dreaming video… frame " + (i + 1) + " of " + total + srcTimeNote() + " · engine: " + (LABEL[nb] || nb));
              continue;   // retry the same frame once, on the new engine
            }
            break;   // non-GPU error, or a GPU error after the switch: no more attempts for this frame
          }
        }
        if (abortMsg) { stopReason = abortMsg; setVideoStatus(stopReason, true); break; }
        if (!frameOk) {
          if (++consecFail >= 5) { stopReason = "Video dream failed: " + (frameErr && frameErr.message || frameErr) + " (frame " + (i + 1) + ")"; setVideoStatus(stopReason, true); break; }
          setVideoStatus("Frame " + (i + 1) + " failed — skipping it…");
          await showRawFrame(frame, i);
          skipped++; i++; continue;
        }
        // Color spin + pop: the hue keeps cycling as the video plays, and
        // the saturation pops. Applied to the display canvas, so the live
        // preview and the recording both get it.
        vHueTotal = (vHueTotal + (parseFloat($("videoHueSpin").value) || 0)) % 360;
        applyFx(outCv, videoFx(vHueTotal));
        pv.getContext("2d").drawImage(outCv, 0, 0, pv.width, pv.height);   // live preview: the new frame, in order
        if (prev) prev.dispose();
        prev = cur;
        setVideoStatus("Dreaming video… frame " + (i + 1) + " of " + total + srcTimeNote() + engLabel() + rampNote());
        setVideoBar((i + 1) / total);
        await presentFrame(i);   // record this dreamed frame at its output time (i/fps)
        if (i === 0 && recorderKind === "ve") {
          // v1.64: prove the encoder gave us an avcC now, before
          // dreaming the rest — if it did not, ensureVeAvcC falls back
          // to MediaRecorder (re-presenting frame 0). A failed
          // fallback stops the run with a clear reason.
          if (!(await ensureVeAvcC())) {
            stopReason = "Video dream failed: could not start a video recorder on this device (VideoEncoder gave no H.264 config and MediaRecorder failed to start).";
            setVideoStatus(stopReason, true);
            break;
          }
        }
        await tf.nextFrame();
        dreamed++;
        consecFail = 0;
        i++;
      } finally {
        clearInterval(frameTimer);
      }
    }
    videoDone = videoRunning && i >= total;
    // The output is exactly the video's frames — dreamed, or raw where a
    // frame had to be skipped — reassembled in order. No bookends: the
    // first and last frames are dreamed and recorded like every other.
    // Finish the recorder. VideoEncoder: flush, then mux the chunks into
    // an MP4 with exact 1/fps timing (the saved video is exactly
    // total/fps = the source duration). MediaRecorder: stop and use its
    // chunks as before.
    if (recorderKind === "ve") {
      try { if (veEncoder) { await veEncoder.flush(); veEncoder.close(); } } catch (e) { console.error(e); }
      veEncoder = null;
      if (veChunks.length && veConfig) {
        try { videoOutBlob = muxMP4Video(veChunks, veConfig, vd.w, vd.h, fps); videoOutMime = "video/mp4"; }
        catch (e) {
          console.error(e);
          if (!stopReason) { stopReason = "Video dream failed: could not write the output video (" + (e && e.message || e) + ")"; setVideoStatus(stopReason, true); }
          videoOutBlob = new Blob([], { type: "video/mp4" }); videoOutMime = "video/mp4";
        }
      } else {
        if (!stopReason) { stopReason = "Video dream failed: the video encoder produced no frames" + (veError ? " (" + (veError.message || veError) + ")" : "") + "."; setVideoStatus(stopReason, true); }
        videoOutBlob = new Blob([], { type: "video/mp4" }); videoOutMime = "video/mp4";
      }
    } else {
      try { if (rec) { rec.stop(); await done; } } catch (e) { console.error(e); }
      rec = null;
      videoOutBlob = new Blob(chunks, { type: mime || "video/webm" });
      videoOutMime = mime;
    }
    pv.style.display = "none"; $("vidOut").style.display = "";
    if (videoOutBlob.size) {
      if (videoOutUrl) URL.revokeObjectURL(videoOutUrl);
      videoOutUrl = URL.createObjectURL(videoOutBlob);
      $("vidOut").src = videoOutUrl;
      $("videoHint").textContent = "";
      const fmt = (videoOutBlob.type || "").indexOf("mp4") >= 0 ? "MP4" : "WebM";
      if (videoDone) {
        const swNote = engineSwitched ? " · engine switched to " + (LABEL[engineSwitched] || engineSwitched) : "";
        const recNote = decoderRecoveries ? " · decoder recovered " + decoderRecoveries + "×" : "";
        const badNote = badSamples ? " · " + badSamples + " bad samples skipped" : "";
        const recErrNote = recError ? " · recorder error: " + (recError.message || recError.error || recError) : "";
        const veErrNote = veError ? " · video encoder error: " + (veError.message || veError.error || veError) : "";
        setVideoStatus("Done — " + dreamed + " of " + total + " frames dreamed" + (skipped ? " · " + skipped + " skipped" : "") + " · " + decodedFrames + " frames decoded" + (reusedFrames ? " · " + reusedFrames + " reused" : "") + " · " + vd.w + "×" + vd.h + " · dream " + dd.w + "×" + dd.h + tileNote + " · depth " + (parseFloat($("videoDepth").value) || 0).toFixed(2) + (videoRampOn() ? " · ramp " + rampStepsStart + " → " + rampStepsTop + " steps, " + rampLrStart.toFixed(3) + " → " + rampLrTop.toFixed(3) : "") + " · " + vModel.label + " · " + fmt + swNote + recNote + badNote + recErrNote + veErrNote + recKindNote + outNote + ". Watch it above, or save it.");
      } else if (stopReason) {
        // A failure stopped the run — the reason MUST be shown. Never
        // overwrite it with the bare "Stopped" line (v1.60 hid the
        // decoder error this way: it dreamed 2 frames, the decoder died,
        // and the status just said "Stopped — dreamed 2 of 749 frames").
        setVideoStatus(stopReason + " — dreamed " + dreamed + " of " + total + " frames. The video above is only the part that finished.", true);
      } else if (!videoRunning && dreamed > 0) {
        // No failure — the user pressed stop. The bare "Stopped" line is
        // shown ONLY here (videoRunning false at loop exit).
        setVideoStatus("Stopped — dreamed " + dreamed + " of " + total + " frames. The video above is only the part that finished.");
      } else if (videoRunning) {
        // Should not happen (every break sets stopReason), but a silent
        // stop must never be possible again — always name a reason.
        setVideoStatus("Video dream stopped unexpectedly at frame " + (i + 1) + " of " + total + " (no reason recorded) — dreamed " + dreamed + " frames. The video above is only the part that finished.", true);
      }
      // else the failure message set in the loop stays — never claim Done for a run that dreamed almost nothing
    } else {
      videoOutBlob = null;
      $("videoHint").textContent = "Dreamed video plays here";
    }
  } catch (e) {
    console.error(e);
    setVideoStatus("Video dream failed: " + (e && e.message || e), true);
    try { if (rec) { rec.stop(); await done; } } catch (err) {}
    try { if (veEncoder) { veEncoder.close(); } } catch (err) {}
    veEncoder = null;
  } finally {
    if (prev) { try { prev.dispose(); } catch (e) {} prev = null; }
    if (tileScratch.resCv) { try { tileScratch.resCv.width = 0; tileScratch.tileCv.width = 0; tileScratch.tmpCv.width = 0; } catch (e) {} }   // v1.73: free the tiled-dream scratch canvases
    if (rampNoise) { try { rampNoise.dispose(); } catch (e) {} rampNoise = null; }   // v1.67: free the per-run noise seed
    restoreRampSliders();   // v1.67: the sliders go back to the user's set start values
    if (held) { try { held.close(); } catch (e) {} held = null; }
    if (decoder) { try { decoder.close(); } catch (e) {} decoder = null; }
    videoRunning = false;
    setVideoBar(null);
    restorePhotoModel();   // the Photos tab gets its own model back
    updateVideoUI();
  }
}
function initVideoTab() {
  $("tabCamera").onclick = () => switchTab("camera");
  $("tabVideo").onclick = () => switchTab("video");
  // Live tab: a launcher, not a web panel — it opens the native
  // live-camera dream activity and must not change the Camera/Video
  // tab state. On the public website there is no bridge, so show a note.
  $("tabLive").onclick = () => {
    if (window.DreamCameraBridge && window.DreamCameraBridge.openLive) window.DreamCameraBridge.openLive();
    else setStatus("Live is in the DreamCamera app");
  };
  buildVideoModelSelect();
  try { $("videoRamp").checked = localStorage.getItem(VIDEO_RAMP_KEY) === "1"; } catch (e) {}
  $("videoRamp").addEventListener("change", () => { try { localStorage.setItem(VIDEO_RAMP_KEY, $("videoRamp").checked ? "1" : "0"); } catch (e) {} });
  $("videoSteps").addEventListener("input", () => { $("videoStepsVal").textContent = $("videoSteps").value; });
  $("videoLr").addEventListener("input", () => { $("videoLrVal").textContent = (+$("videoLr").value).toFixed(3); });
  $("videoDepth").addEventListener("input", () => { $("videoDepthVal").textContent = (+$("videoDepth").value).toFixed(2); });
  $("videoZoomCycles").addEventListener("input", () => { $("videoZoomCyclesVal").textContent = $("videoZoomCycles").value; });
  $("videoZoomAmount").addEventListener("input", () => { $("videoZoomAmountVal").textContent = (+$("videoZoomAmount").value).toFixed(2) + "×"; });
  $("videoHueSpin").addEventListener("input", () => { $("videoHueSpinVal").textContent = (+$("videoHueSpin").value).toFixed(1); });
  $("videoSaturation").addEventListener("input", () => { $("videoSaturationVal").textContent = (+$("videoSaturation").value).toFixed(2); });
  $("videoPulse").addEventListener("input", () => { $("videoPulseVal").textContent = (+$("videoPulse").value).toFixed(2); });
  $("videoGlow").addEventListener("input", () => { $("videoGlowVal").textContent = (+$("videoGlow").value).toFixed(2); });
  $("videoSoft").addEventListener("input", () => { $("videoSoftVal").textContent = (+$("videoSoft").value).toFixed(2); });
  $("videoVignette").addEventListener("input", () => { $("videoVignetteVal").textContent = (+$("videoVignette").value).toFixed(2); });
  $("videoFade").addEventListener("input", () => { $("videoFadeVal").textContent = (+$("videoFade").value).toFixed(2); });
  $("videoRgbSplit").addEventListener("input", () => { $("videoRgbSplitVal").textContent = $("videoRgbSplit").value; });
  $("videoStatic").addEventListener("input", () => { $("videoStaticVal").textContent = (+$("videoStatic").value).toFixed(2); });
  $("videoInvert").addEventListener("input", () => { $("videoInvertVal").textContent = (+$("videoInvert").value).toFixed(2); });
  $("videoKaleido").addEventListener("input", () => { $("videoKaleidoVal").textContent = (+$("videoKaleido").value).toFixed(2); });
  $("videoSwirl").addEventListener("input", () => { $("videoSwirlVal").textContent = (+$("videoSwirl").value).toFixed(2); });
  $("videoEcho").addEventListener("input", () => { $("videoEchoVal").textContent = (+$("videoEcho").value).toFixed(2); });
  $("videoScanlines").addEventListener("input", () => { $("videoScanlinesVal").textContent = (+$("videoScanlines").value).toFixed(2); });
  $("videoPickBtn").onclick = () => $("videoIn").click();
  $("videoIn").addEventListener("change", () => {
    const f = $("videoIn").files[0]; if (!f) return;
    videoFile = f;
    if (videoSrc) URL.revokeObjectURL(videoSrc);
    videoSrc = URL.createObjectURL(f);
    const v = document.createElement("video");
    v.preload = "metadata"; v.src = videoSrc;
    v.onloadedmetadata = () => {
      const d = v.duration || 0;
      videoMeta = { duration: d, width: v.videoWidth, height: v.videoHeight };
      $("videoInfo").textContent = f.name + " · " + d.toFixed(1) + "s · " + v.videoWidth + "×" + v.videoHeight;
      setVideoStatus("Video ready. Pick a model and generate it.");
      updateVideoUI();
    };
    v.onerror = () => { setVideoStatus("Could not read that video.", true); videoMeta = null; updateVideoUI(); };
  });
  $("videoDreamBtn").onclick = generateVideo;
  $("videoStopBtn").onclick = () => { videoRunning = false; setVideoStatus("Stopping…"); };
  $("videoSaveBtn").onclick = () => {
    if (!videoOutBlob) return;
    const ext = (videoOutBlob.type || "").includes("mp4") ? "mp4" : "webm";
    downloadBlob(videoOutBlob, "dreamcamera-video-" + Date.now().toString(36) + "." + ext);
    setVideoStatus("Video saved as ." + ext + ".");
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
