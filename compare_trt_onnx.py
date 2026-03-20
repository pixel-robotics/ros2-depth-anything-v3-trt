#!/usr/bin/env python3
"""
Compare ONNX Runtime vs TensorRT output for the exact same input tensor.
Saves a preprocessed tensor to disk, runs ONNX RT, then you feed the same
tensor through TRT via the C++ node to compare.

Usage:
  python3 compare_trt_onnx.py /data/DA3METRIC-LARGE-fp32.onnx
"""

import sys
import numpy as np
import cv2
import onnxruntime as ort

MODEL_W, MODEL_H = 504, 280
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SRC_W, SRC_H = 3840, 2160


def preprocess(bgr_image):
    resized = cv2.resize(bgr_image, (MODEL_W, MODEL_H), interpolation=cv2.INTER_CUBIC)
    rgb = resized[:, :, ::-1].astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return rgb.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)


def stats(name, arr):
    flat = arr.flatten()
    print(f"  {name:20s}  min={flat.min():.6f}  max={flat.max():.6f}  "
          f"mean={flat.mean():.6f}  std={flat.std():.6f}")


def main():
    onnx_path = sys.argv[1] if len(sys.argv) > 1 else "/data/DA3METRIC-LARGE-fp32.onnx"

    # Use a fixed synthetic frame matching rosbag brightness
    gray = np.full((SRC_H, SRC_W), 112, dtype=np.uint8)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tensor = preprocess(img)

    print(f"Input tensor: shape={tensor.shape} min={tensor.min():.6f} max={tensor.max():.6f} mean={tensor.mean():.6f}")

    # Save raw tensor for optional C++ comparison
    tensor.tofile("/tmp/test_input_1x3x280x504.bin")
    print("Saved input tensor to /tmp/test_input_1x3x280x504.bin")

    # ONNX Runtime
    print(f"\n=== ONNX Runtime (CPU) on {onnx_path} ===")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_names = [o.name for o in sess.get_outputs()]

    results = sess.run(output_names, {input_name: tensor})
    for name, arr in zip(output_names, results):
        stats(name, arr)

    # Save ONNX outputs for comparison
    for name, arr in zip(output_names, results):
        path = f"/tmp/onnxrt_output_{name}.bin"
        arr.tofile(path)
        print(f"  Saved to {path}")

    # Now try TensorRT via Python if available
    try:
        import tensorrt as trt
        print(f"\n=== TensorRT {trt.__version__} (GPU) ===")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        # Build engine from ONNX
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        print(f"Parsing {onnx_path}...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"  Parse error: {parser.get_error(i)}")
                return

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        # Disable TF32 for strict fp32
        config.clear_flag(trt.BuilderFlag.TF32)

        print("Building engine (this takes a couple minutes on Jetson)...")
        engine_bytes = builder.build_serialized_network(network, config)
        if not engine_bytes:
            print("Engine build failed!")
            return

        engine = runtime.deserialize_cuda_engine(engine_bytes)
        context = engine.create_execution_context()

        import pycuda.driver as cuda
        import pycuda.autoinit

        # Allocate buffers
        input_idx = 0
        depth_idx = engine.get_binding_index("depth") if hasattr(engine, 'get_binding_index') else 1
        sky_idx = engine.get_binding_index("sky") if hasattr(engine, 'get_binding_index') else 2

        h_input = np.ascontiguousarray(tensor)
        d_input = cuda.mem_alloc(h_input.nbytes)

        # Get output shapes
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            mode = engine.get_tensor_mode(name)
            print(f"  Tensor '{name}': shape={list(shape)} mode={mode}")

        depth_shape = engine.get_tensor_shape("depth")
        sky_shape = engine.get_tensor_shape("sky")
        h_depth = np.empty([int(x) for x in depth_shape], dtype=np.float32)
        h_sky = np.empty([int(x) for x in sky_shape], dtype=np.float32)
        d_depth = cuda.mem_alloc(h_depth.nbytes)
        d_sky = cuda.mem_alloc(h_sky.nbytes)

        stream = cuda.Stream()

        # Set tensor addresses
        context.set_tensor_address("image", int(d_input))
        context.set_tensor_address("depth", int(d_depth))
        context.set_tensor_address("sky", int(d_sky))

        # Run inference
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream.handle)
        cuda.memcpy_dtoh_async(h_depth, d_depth, stream)
        cuda.memcpy_dtoh_async(h_sky, d_sky, stream)
        stream.synchronize()

        stats("TRT depth", h_depth)
        stats("TRT sky", h_sky)

        # Compare
        onnx_depth = results[0]
        print(f"\n=== Comparison ===")
        diff = np.abs(h_depth.flatten() - onnx_depth.flatten())
        print(f"  abs diff:  min={diff.min():.8f}  max={diff.max():.8f}  mean={diff.mean():.8f}")
        print(f"  ONNX RT depth mean={onnx_depth.mean():.6f}  TRT depth mean={h_depth.mean():.6f}")
        print(f"  Ratio TRT/ONNX = {h_depth.mean() / onnx_depth.mean():.4f}")

    except ImportError:
        print("\n=== TensorRT Python not available, skipping direct comparison ===")
        print("Compare manually: run the ROS node with debug logging and check after_trt_copy stats")


if __name__ == "__main__":
    main()
