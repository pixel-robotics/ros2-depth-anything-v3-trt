/**
 * Standalone TRT vs ONNX RT comparison tool.
 * Loads a raw float32 tensor from disk, runs TRT inference, prints stats.
 * Compare output against ONNX RT reference saved by compare_trt_onnx.py.
 *
 * Build: added to CMakeLists.txt as compare_trt target
 * Usage: ./compare_trt <engine_or_onnx_path> [input_bin] [onnx_depth_ref_bin]
 */

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::cerr << "[TRT] " << msg << std::endl;
  }
} gLogger;

std::vector<float> readBin(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::cerr << "Cannot open " << path << std::endl; return {}; }
  size_t bytes = f.tellg();
  f.seekg(0);
  std::vector<float> data(bytes / sizeof(float));
  f.read(reinterpret_cast<char*>(data.data()), bytes);
  return data;
}

void printStats(const char* name, const float* data, size_t n) {
  float mn = data[0], mx = data[0];
  double sum = 0, sum2 = 0;
  for (size_t i = 0; i < n; ++i) {
    if (data[i] < mn) mn = data[i];
    if (data[i] > mx) mx = data[i];
    sum += data[i];
    sum2 += double(data[i]) * data[i];
  }
  double mean = sum / n;
  double std_dev = std::sqrt(sum2 / n - mean * mean);
  printf("  %-20s min=%.6f  max=%.6f  mean=%.6f  std=%.6f\n", name, mn, mx, mean, std_dev);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <onnx_path> [input.bin] [onnx_depth_ref.bin]" << std::endl;
    return 1;
  }

  std::string onnx_path = argv[1];
  std::string input_path = argc > 2 ? argv[2] : "/tmp/test_input_1x3x280x504.bin";
  std::string ref_path = argc > 3 ? argv[3] : "/tmp/onnxrt_output_depth.bin";

  // Load input
  auto input = readBin(input_path);
  if (input.empty()) return 1;
  std::cout << "Input: " << input.size() << " floats from " << input_path << std::endl;
  printStats("input", input.data(), input.size());

  // Build TRT engine from ONNX
  auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0U));
  auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));

  std::cout << "Parsing ONNX: " << onnx_path << std::endl;
  if (!parser->parseFromFile(onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    std::cerr << "Failed to parse ONNX" << std::endl;
    return 1;
  }

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
  config->clearFlag(nvinfer1::BuilderFlag::kTF32);

  std::cout << "Building engine (may take a few minutes)..." << std::endl;
  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!plan) { std::cerr << "Engine build failed" << std::endl; return 1; }

  auto runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
  auto engine = std::unique_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()));
  auto context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());

  // Find tensor info
  std::cout << "Engine tensors:" << std::endl;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    auto dims = engine->getTensorShape(name);
    auto mode = engine->getTensorIOMode(name);
    std::cout << "  " << name << ": [";
    for (int d = 0; d < dims.nbDims; ++d) std::cout << (d ? "," : "") << dims.d[d];
    std::cout << "] " << (mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT" : "OUTPUT") << std::endl;
  }

  // Allocate GPU buffers
  float *d_input, *d_depth, *d_sky;
  cudaMalloc(&d_input, input.size() * sizeof(float));

  // Get depth/sky sizes
  auto depth_dims = engine->getTensorShape("depth");
  auto sky_dims = engine->getTensorShape("sky");
  size_t depth_n = 1, sky_n = 1;
  for (int i = 0; i < depth_dims.nbDims; ++i) depth_n *= depth_dims.d[i];
  for (int i = 0; i < sky_dims.nbDims; ++i) sky_n *= sky_dims.d[i];

  cudaMalloc(&d_depth, depth_n * sizeof(float));
  cudaMalloc(&d_sky, sky_n * sizeof(float));

  context->setTensorAddress("image", d_input);
  context->setTensorAddress("depth", d_depth);
  context->setTensorAddress("sky", d_sky);

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Run inference
  cudaMemcpyAsync(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
  if (!context->enqueueV3(stream)) {
    std::cerr << "Inference failed!" << std::endl;
    return 1;
  }

  std::vector<float> h_depth(depth_n), h_sky(sky_n);
  cudaMemcpyAsync(h_depth.data(), d_depth, depth_n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_sky.data(), d_sky, sky_n * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  std::cout << "\n=== TRT output ===" << std::endl;
  printStats("TRT depth", h_depth.data(), depth_n);
  printStats("TRT sky", h_sky.data(), sky_n);

  // Load and compare ONNX RT reference
  auto ref = readBin(ref_path);
  if (!ref.empty() && ref.size() == depth_n) {
    std::cout << "\n=== ONNX RT reference ===" << std::endl;
    printStats("ONNX depth", ref.data(), ref.size());

    double max_diff = 0, sum_diff = 0;
    for (size_t i = 0; i < depth_n; ++i) {
      double d = std::abs(h_depth[i] - ref[i]);
      if (d > max_diff) max_diff = d;
      sum_diff += d;
    }
    printf("\n=== Comparison ===\n");
    printf("  abs diff:  max=%.8f  mean=%.8f\n", max_diff, sum_diff / depth_n);
    printf("  TRT mean=%.6f  ONNX mean=%.6f  ratio=%.4f\n",
           h_depth[0] ? double(std::accumulate(h_depth.begin(), h_depth.end(), 0.0)) / depth_n : 0,
           ref[0] ? double(std::accumulate(ref.begin(), ref.end(), 0.0)) / ref.size() : 0,
           double(std::accumulate(h_depth.begin(), h_depth.end(), 0.0)) /
           double(std::accumulate(ref.begin(), ref.end(), 0.0)));
  } else {
    std::cout << "No ONNX reference at " << ref_path << " (or size mismatch)" << std::endl;
  }

  cudaFree(d_input); cudaFree(d_depth); cudaFree(d_sky);
  cudaStreamDestroy(stream);
  return 0;
}
