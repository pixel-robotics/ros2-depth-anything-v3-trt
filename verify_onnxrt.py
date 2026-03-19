#!/usr/bin/env python3
"""
Verify ONNX Runtime output vs TensorRT on Jetson.
Reproduces the exact C++ preprocessing pipeline and compares depth output stats.

Usage:
  python3 verify_onnxrt.py /data/DA3METRIC-LARGE.onnx [image_path]

If no image is provided, generates a synthetic grayscale frame matching
the observed rosbag brightness (~107 mean, mono8 replicated to BGR).
"""

import sys
import os
import numpy as np
import cv2
import onnx
from onnx import numpy_helper, TensorProto
import onnxruntime as ort

MODEL_W, MODEL_H = 504, 280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Camera intrinsics from rosbag (basler_top_corrected_link)
FX_ORIG, FY_ORIG = 2300.0, 2300.0
SRC_W, SRC_H = 3840, 2160


def preprocess(bgr_image: np.ndarray) -> np.ndarray:
    """Exact replica of C++ TensorRTDepthAnything::preprocess()."""
    # Resize to model input (cubic, same as C++)
    resized = cv2.resize(bgr_image, (MODEL_W, MODEL_H), interpolation=cv2.INTER_CUBIC)

    # BGR -> RGB, float32, normalize with ImageNet stats
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

    # HWC -> NCHW
    tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 280, 504)
    return tensor.astype(np.float32)


def compute_stats(name: str, arr: np.ndarray):
    flat = arr.flatten()
    print(f"  {name:25s}  min={flat.min():.6f}  max={flat.max():.6f}  "
          f"mean={flat.mean():.6f}  std={flat.std():.6f}  "
          f"p95={np.percentile(flat, 95):.6f}  p99={np.percentile(flat, 99):.6f}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <model.onnx> [image_path]")
        sys.exit(1)

    onnx_path = sys.argv[1]
    image_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Load or create test image
    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Error: cannot read {image_path}")
            sys.exit(1)
        if len(img.shape) == 2:
            # mono -> BGR (same as cv_bridge toCvCopy BGR8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print(f"Loaded image: {image_path} ({img.shape[1]}x{img.shape[0]})")
    else:
        # Synthetic mono8 frame matching rosbag brightness (~107)
        gray = np.full((SRC_H, SRC_W), 107, dtype=np.uint8)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        print(f"Using synthetic mono8 frame (mean=107, {SRC_W}x{SRC_H})")

    img_mean = img.mean(axis=(0, 1))
    print(f"Input image mean BGR: {img_mean[0]:.2f} / {img_mean[1]:.2f} / {img_mean[2]:.2f}")

    # Preprocess
    input_tensor = preprocess(img)
    flat_input = input_tensor.flatten()
    print(f"\nPreprocessed tensor shape: {input_tensor.shape}")
    print(f"  INPUT  min={flat_input.min():.6f}  max={flat_input.max():.6f}  "
          f"mean={flat_input.mean():.6f}  std={flat_input.std():.6f}  "
          f"first4={flat_input[0]:.6f},{flat_input[1]:.6f},{flat_input[2]:.6f},{flat_input[3]:.6f}")

    # Convert fp16 model to fp32 for CPU inference and replace MS-domain Gelu with standard ops
    print(f"\nLoading ONNX model: {onnx_path}")
    fp32_path = onnx_path.replace(".onnx", "_fp32.onnx")
    if not os.path.exists(fp32_path):
        print("Converting fp16 -> fp32 and replacing MS Gelu (one-time, may take a minute)...")
        model = onnx.load(onnx_path)
        graph = model.graph

        # 1. Convert ALL initializers from fp16 to fp32
        for init in graph.initializer:
            if init.data_type == TensorProto.FLOAT16:
                arr = numpy_helper.to_array(init).astype(np.float32)
                new_init = numpy_helper.from_array(arr, name=init.name)
                init.CopyFrom(new_init)

        # 2. Convert ALL type protos (inputs, outputs, value_info)
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            if vi.type.HasField("tensor_type"):
                if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
                    vi.type.tensor_type.elem_type = TensorProto.FLOAT

        # 3. Convert Cast node attributes that target fp16, and Constant tensor attrs
        for node in graph.node:
            if node.op_type == "Cast":
                for attr in node.attribute:
                    if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                        attr.i = TensorProto.FLOAT
            for attr in node.attribute:
                if attr.type == 4 and attr.t.data_type == TensorProto.FLOAT16:  # TENSOR
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    attr.t.CopyFrom(numpy_helper.from_array(arr))

        # 4. Replace MS Gelu with standard ONNX Gelu (opset 20)
        for node in graph.node:
            if node.op_type == "Gelu" and node.domain == "com.microsoft":
                node.domain = ""

        # 5. Ensure opset >= 20 for standard Gelu support
        for opset in model.opset_import:
            if opset.domain == "" or opset.domain == "ai.onnx":
                opset.version = max(opset.version, 20)

        # 6. Remove ms-domain opset if no longer needed
        ms_nodes = [n for n in graph.node if n.domain == "com.microsoft"]
        if not ms_nodes:
            opsets_to_keep = [o for o in model.opset_import if o.domain != "com.microsoft"]
            del model.opset_import[:]
            model.opset_import.extend(opsets_to_keep)

        # 7. Clear value_info so shape inference can regenerate with correct types
        del graph.value_info[:]

        # 8. Run shape inference to fill in all intermediate tensor types as fp32
        print("Running shape inference...")
        model = onnx.shape_inference.infer_shapes(model)

        onnx.save(model, fp32_path)
        print(f"Saved fp32 model: {fp32_path}")
    else:
        print(f"Using cached fp32 model: {fp32_path}")

    sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]
    print(f"Input: {input_name}, Outputs: {output_names}")

    results = sess.run(output_names, {input_name: input_tensor})

    print(f"\n=== ONNX Runtime output (raw, no postprocessing) ===")
    for name, arr in zip(output_names, results):
        compute_stats(f"output[{name}]", arr)

    # Find depth output
    depth_idx = next((i for i, n in enumerate(output_names) if "depth" in n), 0)
    depth_raw = results[depth_idx].squeeze()

    # Apply same focal scaling as C++
    scale_x = MODEL_W / SRC_W
    scale_y = MODEL_H / SRC_H
    fx = FX_ORIG * scale_x
    fy = FY_ORIG * scale_y
    focal_px = 0.5 * (fx + fy)
    focal_scale = focal_px / 300.0 if focal_px > 0 else 1.0

    depth_scaled = depth_raw * focal_scale
    print(f"\n=== After focal scaling (fx={fx:.4f}, fy={fy:.4f}, scale={focal_scale:.6f}) ===")
    compute_stats("after_focal_scale", depth_scaled)

    # Repeat inference 10 times with same input to check determinism
    print(f"\n=== Determinism check (10 runs, same input) ===")
    means = []
    stds = []
    for i in range(10):
        r = sess.run(output_names, {input_name: input_tensor})
        d = r[depth_idx].squeeze()
        means.append(d.mean())
        stds.append(d.std())
        print(f"  run {i:2d}  mean={d.mean():.6f}  std={d.std():.6f}")

    print(f"\n  mean range: [{min(means):.6f}, {max(means):.6f}]  "
          f"(delta={max(means)-min(means):.8f})")
    print(f"  std  range: [{min(stds):.6f}, {max(stds):.6f}]  "
          f"(delta={max(stds)-min(stds):.8f})")

    # Now test with slightly different brightness (simulate frame-to-frame variation)
    print(f"\n=== Brightness sensitivity test (brightness 100-115, step 1) ===")
    print(f"  {'brightness':>10s}  {'depth_mean':>12s}  {'depth_std':>12s}  {'depth_max':>12s}")
    for brightness in range(100, 116):
        gray = np.full((SRC_H, SRC_W), brightness, dtype=np.uint8)
        test_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = preprocess(test_img)
        r = sess.run(output_names, {input_name: t})
        d = r[depth_idx].squeeze()
        print(f"  {brightness:10d}  {d.mean():12.6f}  {d.std():12.6f}  {d.max():12.6f}")


if __name__ == "__main__":
    main()
