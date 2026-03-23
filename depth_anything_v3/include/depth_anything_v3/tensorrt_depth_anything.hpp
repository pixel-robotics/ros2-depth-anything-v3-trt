// Copyright 2025 Institute for Automotive Engineering (ika), RWTH Aachen University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DEPTH_ANYTHING_V3__TENSORRT_DEPTH_ANYTHING_HPP_
#define DEPTH_ANYTHING_V3__TENSORRT_DEPTH_ANYTHING_HPP_

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <tensorrt_common/tensorrt_common.hpp>
#include <vector>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace depth_anything_v3
{
using cuda_utils::CudaUniquePtr;
using cuda_utils::CudaUniquePtrHost;
using cuda_utils::makeCudaStream;
using cuda_utils::StreamUniquePtr;

/**
 * @class TensorRTDepthAnything
 * @brief TensorRT/ONNX Runtime DepthAnythingV3 wrapper for depth estimation and point cloud generation
 */
class TensorRTDepthAnything
{
public:
  TensorRTDepthAnything(
    const std::string & model_path, const std::string & precision,
    const tensorrt_common::BuildConfig build_config = tensorrt_common::BuildConfig(),
    const bool use_gpu_preprocess = false, std::string calibration_image_list_file = std::string(),
    const tensorrt_common::BatchConfig & batch_config = {1, 1, 1},
    const size_t max_workspace_size = (1 << 30),
    const std::string & backend = "tensorrt");

  bool doInference(const std::vector<cv::Mat> & images, const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor = 1, bool colorize_pointcloud = false);

  void buildPlaneProjectionPointCloud(
    const sensor_msgs::msg::CameraInfo & camera_info,
    const geometry_msgs::msg::TransformStamped & camera_to_plane_tf,
    int downsample_factor,
    const cv::Mat & rgb_image = cv::Mat(),
    const cv::Mat & mask = cv::Mat());

  void initPreprocessBuffer(int width, int height);

  void printProfiling(void);

  const cv::Mat& getDepthImage() const;

  const sensor_msgs::msg::PointCloud2& getPointCloud() const;

private:
  void preprocess(const std::vector<cv::Mat> & images);

  bool infer();

#ifdef USE_ONNXRUNTIME
  bool inferOrt();
#endif

  void postprocess(const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor = 1, const cv::Mat & rgb_image = cv::Mat());

  void buildPointCloud(
    const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor,
    const cv::Mat & rgb_image);
public:
  void setSkyThreshold(float threshold) { sky_threshold_ = threshold; }

  std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;

  // Input/output buffers
  std::vector<float> input_h_;
  CudaUniquePtr<float[]> input_d_;

  // Output buffer for predicted depth
  CudaUniquePtr<float[]> depth_d_;
  CudaUniquePtrHost<float[]> depth_h_;
  size_t depth_elem_num_{};
  // Output buffer for predicted sky
  CudaUniquePtr<float[]> sky_d_;
  CudaUniquePtrHost<float[]> sky_h_;
  size_t sky_elem_num_{};
  std::vector<CudaUniquePtr<float[]>> extra_output_buffers_;
  cv::Mat model_depth_;
  cv::Mat sky_mask_;

  StreamUniquePtr stream_{makeCudaStream()};

  int batch_size_;

  // preprocessing parameters
  bool use_gpu_preprocess_;
  CudaUniquePtrHost<unsigned char[]> image_buf_h_;
  CudaUniquePtr<unsigned char[]> image_buf_d_;

  int src_width_;
  int src_height_;
  int input_width_ = 504;
  int input_height_ = 280;

  // Results
  cv::Mat depth_image_;
  double scale_x_{1.0};
  double scale_y_{1.0};
  float sky_threshold_{0.3f};
  const float sky_depth_cap_{200.0f};
  sensor_msgs::msg::PointCloud2 point_cloud_;

  // Backend selection
  std::string backend_{"tensorrt"};
  std::string model_path_;

#ifdef USE_ONNXRUNTIME
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
#endif
};

}  // namespace depth_anything_v3

#endif  // DEPTH_ANYTHING_V3__TENSORRT_DEPTH_ANYTHING_HPP_
