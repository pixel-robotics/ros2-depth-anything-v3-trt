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

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <cstring>
#include <iostream>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <rclcpp/rclcpp.hpp>

#include "depth_anything_v3/tensorrt_depth_anything.hpp"
#include "cuda_utils/cuda_check_error.hpp"
#include "cuda_utils/cuda_unique_ptr.hpp"

namespace
{
namespace fs = std::filesystem;

static size_t volumeFromDims(const nvinfer1::Dims & dims, int batch_size)
{
  return std::accumulate(dims.d, dims.d + dims.nbDims, size_t{1},
    [batch_size](size_t acc, int dim) {
      return acc * (dim == -1 ? static_cast<size_t>(batch_size) : static_cast<size_t>(dim));
    });
}

// Simple depth to point cloud conversion using camera intrinsics
static void depthImageToPointCloud(
  const cv::Mat & depth_image,
  const sensor_msgs::msg::CameraInfo & camera_info,
  sensor_msgs::msg::PointCloud2 & cloud_msg,
  const std::string & frame_id,
  int downsample_factor = 1,
  const cv::Mat & rgb_image = cv::Mat(),
  const cv::Mat & non_sky_mask = cv::Mat())
{
  cloud_msg.header.frame_id = frame_id;
  
  const int downsampled_height = (depth_image.rows + downsample_factor - 1) / downsample_factor;
  const int downsampled_width = (depth_image.cols + downsample_factor - 1) / downsample_factor;
  
  cloud_msg.height = downsampled_height;
  cloud_msg.width = downsampled_width;
  cloud_msg.is_dense = false;
  cloud_msg.is_bigendian = false;

  sensor_msgs::PointCloud2Modifier pcd_modifier(cloud_msg);
  const bool has_color = !rgb_image.empty();
  if (has_color) {
    pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
  } else {
    pcd_modifier.setPointCloud2FieldsByString(1, "xyz");
  }

  const double fx = camera_info.k[0];
  const double fy = camera_info.k[4]; 
  const double cx = camera_info.k[2];
  const double cy = camera_info.k[5];

  sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
  std::unique_ptr<sensor_msgs::PointCloud2Iterator<uint8_t>> iter_r, iter_g, iter_b;
  if (has_color) {
    iter_r = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "r");
    iter_g = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "g");
    iter_b = std::make_unique<sensor_msgs::PointCloud2Iterator<uint8_t>>(cloud_msg, "b");
  }

  const float bad_point = std::numeric_limits<float>::quiet_NaN();

  const bool use_mask = !non_sky_mask.empty();

  for (int v = 0; v < depth_image.rows; v += downsample_factor) {
    for (int u = 0; u < depth_image.cols; u += downsample_factor) {
      if (use_mask && non_sky_mask.at<uint8_t>(v, u) == 0) {
        *iter_x = *iter_y = *iter_z = bad_point;
        if (has_color) { **iter_r = **iter_g = **iter_b = 0; }
        ++iter_x; ++iter_y; ++iter_z;
        if (has_color) { ++(*iter_r); ++(*iter_g); ++(*iter_b); }
        continue;
      }

      const float depth = depth_image.at<float>(v, u);
      
      if (depth <= 0.0f || !std::isfinite(depth)) {
        *iter_x = *iter_y = *iter_z = bad_point;
        if (has_color) { **iter_r = **iter_g = **iter_b = 0; }
      } else {
        *iter_x = static_cast<float>((u - cx) * depth / fx);
        *iter_y = static_cast<float>((v - cy) * depth / fy);
        *iter_z = depth;
        if (has_color) {
          const cv::Vec3b rgb = rgb_image.at<cv::Vec3b>(v, u);
          **iter_r = rgb[2];
          **iter_g = rgb[1];
          **iter_b = rgb[0];
        }
      }
      ++iter_x; ++iter_y; ++iter_z;
      if (has_color) { ++(*iter_r); ++(*iter_g); ++(*iter_b); }
    }
  }
}

} // anonymous namespace

namespace depth_anything_v3
{

TensorRTDepthAnything::TensorRTDepthAnything(
  const std::string & model_path, const std::string & precision,
  tensorrt_common::BuildConfig build_config, const bool use_gpu_preprocess,
  std::string /* calibration_image_list_path */, const tensorrt_common::BatchConfig & batch_config,
  const size_t max_workspace_size)
: batch_size_(batch_config[2]), use_gpu_preprocess_(use_gpu_preprocess)
{
  src_width_ = -1;
  src_height_ = -1;

  if (!fs::exists(model_path)) {
    throw std::runtime_error("Model file does not exist: " + model_path);
  }

  // Initialize TensorRT common
  trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
    model_path, precision, nullptr, batch_config, max_workspace_size, build_config);
  trt_common_->setup();

  auto * engine = trt_common_->getEngine();
  depth_elem_num_ = 0;
  sky_elem_num_ = 0;
  for (int i = 0; i < trt_common_->getNbIOTensors(); ++i) {
    const char * name = engine->getIOTensorName(i);
    const auto dims = trt_common_->getBindingDimensions(i);
    if (name && std::string(name) == "depth") {
      depth_elem_num_ = volumeFromDims(dims, batch_size_);
    } else if (name && std::string(name) == "sky") {
      sky_elem_num_ = volumeFromDims(dims, batch_size_);
    }
  }
  if (depth_elem_num_ == 0) {
    // Fallback to the first output binding if names are unavailable
    const auto dims = trt_common_->getBindingDimensions(1);
    depth_elem_num_ = volumeFromDims(dims, batch_size_);
  }
  if (sky_elem_num_ == 0) {
    throw std::runtime_error("Expected TensorRT engine to expose 'sky' output tensor, but none was found");
  }

  // Allocate GPU/CPU memory for outputs
  depth_d_ = cuda_utils::make_unique<float[]>(depth_elem_num_);
  depth_h_ = cuda_utils::make_unique_host<float[]>(depth_elem_num_, cudaHostAllocDefault);
  sky_d_ = cuda_utils::make_unique<float[]>(sky_elem_num_);
  sky_h_ = cuda_utils::make_unique_host<float[]>(sky_elem_num_, cudaHostAllocDefault);

  // Get input dimensions
  const auto input_dims = trt_common_->getBindingDimensions(0);
  const int input_channels = input_dims.d[1];
  input_height_ = input_dims.d[2];
  input_width_ = input_dims.d[3];
  RCLCPP_INFO(rclcpp::get_logger("TensorRTDepthAnything"),
    "Model input dims: [%d, %d, %d, %d] (N, C, H, W)",
    input_dims.d[0], input_channels, input_height_, input_width_);

  // Allocate input memory
  const size_t input_elem_num = batch_size_ * input_channels * input_height_ * input_width_;
  input_d_ = cuda_utils::make_unique<float[]>(input_elem_num);
  input_h_.resize(input_elem_num);

}

void TensorRTDepthAnything::initPreprocessBuffer(int width, int height)
{
  src_width_ = width;
  src_height_ = height;
  scale_x_ = static_cast<double>(input_width_) / static_cast<double>(src_width_);
  scale_y_ = static_cast<double>(input_height_) / static_cast<double>(src_height_);
  
  if (use_gpu_preprocess_) {
    const size_t image_size = src_width_ * src_height_ * 3; // RGB
    image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(
      image_size * batch_size_, cudaHostAllocDefault);
    image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(image_size * batch_size_);
  }
}

bool TensorRTDepthAnything::doInference(
  const std::vector<cv::Mat> & images, 
  const sensor_msgs::msg::CameraInfo & camera_info,
  int downsample_factor,
  bool colorize_pointcloud)
{
  if (images.empty()) {
    RCLCPP_ERROR(rclcpp::get_logger("TensorRTDepthAnything"), "No images provided for inference.");
    return false;
  }

  if (images.size() != static_cast<size_t>(batch_size_)) {
    RCLCPP_ERROR(
      rclcpp::get_logger("TensorRTDepthAnything"),
      "Batch size mismatch. Expected: %d, got: %zu", batch_size_, images.size());
    return false;
  }

  // Preprocess (GPU preprocessing not yet implemented, using CPU)
  preprocess(images);

  // Run inference
  if (!infer()) {
    return false;
  }

  // Postprocess with downsampling
  cv::Mat rgb_for_pointcloud = colorize_pointcloud ? images[0] : cv::Mat();
  postprocess(camera_info, downsample_factor, rgb_for_pointcloud);
  
  return true;
}

void TensorRTDepthAnything::preprocess(const std::vector<cv::Mat> & images)
{
  const auto batch_size = images.size();
  auto input_dims = trt_common_->getBindingDimensions(0);
  if (input_dims.d[0] == -1) {
    input_dims.d[0] = batch_size_;
  }
  trt_common_->setBindingDimensions(0, input_dims);

  input_height_ = input_dims.d[2];
  input_width_ = input_dims.d[3];
  const int input_chan = input_dims.d[1];
  scale_x_ = static_cast<double>(input_width_) / static_cast<double>(src_width_);
  scale_y_ = static_cast<double>(input_height_) / static_cast<double>(src_height_);


  std::vector<cv::Mat> resized_images;
  resized_images.reserve(batch_size);
  for (const auto & image : images) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_width_, input_height_), 0, 0, cv::INTER_CUBIC);
    resized_images.emplace_back(resized_image);
  }

  const size_t volume = batch_size * input_chan * input_height_ * input_width_;
  input_h_.assign(volume, 0.0f);

  const std::vector<float> mean{0.485f, 0.456f, 0.406f};
  const std::vector<float> std_vals{0.229f, 0.224f, 0.225f};

  const size_t strides_cv[4] = {
    static_cast<size_t>(input_width_ * input_chan * input_height_),
    static_cast<size_t>(input_width_ * input_chan),
    static_cast<size_t>(input_chan), 1};
  const size_t strides[4] = {
    static_cast<size_t>(input_height_ * input_width_ * input_chan),
    static_cast<size_t>(input_height_ * input_width_),
    static_cast<size_t>(input_width_), 1};

  for (size_t n = 0; n < batch_size; ++n) {
    const auto & img = resized_images[n];
    const auto * src_ptr = img.data;
    for (int h = 0; h < input_height_; ++h) {
      for (int w = 0; w < input_width_; ++w) {
        for (int c = 0; c < input_chan; ++c) {
          const size_t offset_cv =
            h * strides_cv[1] + w * strides_cv[2] + (input_chan - c - 1) * strides_cv[3];
          const size_t offset =
            n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3];
          const float value = static_cast<float>(src_ptr[offset_cv]) / 255.0f;
          input_h_[offset] = (value - mean[c]) / std_vals[c];
        }
      }
    }
  }

  auto * engine = trt_common_->getEngine();
  for (int i = 0; i < trt_common_->getNbIOTensors(); ++i) {
    const char * name = engine->getIOTensorName(i);
    const auto dims = trt_common_->getBindingDimensions(i);
    const size_t required_output_elems = volumeFromDims(dims, batch_size_);
    if (name && std::string(name) == "depth") {
      if (required_output_elems != depth_elem_num_) {
        depth_elem_num_ = required_output_elems;
        depth_d_ = cuda_utils::make_unique<float[]>(depth_elem_num_);
        depth_h_ = cuda_utils::make_unique_host<float[]>(depth_elem_num_, cudaHostAllocDefault);
      }
    } else if (name && std::string(name) == "sky") {
      if (required_output_elems != sky_elem_num_) {
        sky_elem_num_ = required_output_elems;
        sky_d_ = cuda_utils::make_unique<float[]>(sky_elem_num_);
        sky_h_ = cuda_utils::make_unique_host<float[]>(sky_elem_num_, cudaHostAllocDefault);
      }
    }
  }

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice,
    *stream_));
}

bool TensorRTDepthAnything::infer()
{
  auto * context = trt_common_->getContext();
  auto * engine = trt_common_->getEngine();

  extra_output_buffers_.clear();

  for (int i = 0; i < trt_common_->getNbIOTensors(); ++i) {
    const char * name = engine->getIOTensorName(i);
    const std::string tensor_name = name ? std::string(name) : std::string();
    const bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;

    void * buffer_ptr = nullptr;
    if (is_input || tensor_name.find("input") != std::string::npos ||
        tensor_name.find("image") != std::string::npos) {
      buffer_ptr = input_d_.get();
    } else if (tensor_name == "depth" || tensor_name.find("depth") != std::string::npos) {
      buffer_ptr = depth_d_.get();
    } else if (tensor_name == "sky" || tensor_name.find("sky") != std::string::npos) {
      if (!sky_d_ && sky_elem_num_ > 0) {
        sky_d_ = cuda_utils::make_unique<float[]>(sky_elem_num_);
        sky_h_ = cuda_utils::make_unique_host<float[]>(sky_elem_num_, cudaHostAllocDefault);
      }
      buffer_ptr = sky_d_ ? static_cast<void *>(sky_d_.get()) : static_cast<void *>(depth_d_.get());
    } else {
      const auto dims = trt_common_->getBindingDimensions(i);
      const size_t elem_count = volumeFromDims(dims, batch_size_);
      extra_output_buffers_.push_back(cuda_utils::make_unique<float[]>(elem_count));
      buffer_ptr = extra_output_buffers_.back().get();
    }

    context->setTensorAddress(name, buffer_ptr);
  }

  if (!trt_common_->enqueueV3(*stream_)) {
    return false;
  }

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    depth_h_.get(), depth_d_.get(), depth_elem_num_ * sizeof(float),
    cudaMemcpyDeviceToHost, *stream_));

  if (sky_d_ && sky_h_) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      sky_h_.get(), sky_d_.get(), sky_elem_num_ * sizeof(float),
      cudaMemcpyDeviceToHost, *stream_));
  }

  CHECK_CUDA_ERROR(cudaStreamSynchronize(*stream_));

  return true;
}

void TensorRTDepthAnything::postprocess(
  const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor, const cv::Mat & rgb_image)
{
  const auto output_dims = trt_common_->getBindingDimensions(1);
  const int height = output_dims.nbDims > 2 ? output_dims.d[2] : input_height_;
  const int width = output_dims.nbDims > 3 ? output_dims.d[3] : input_width_;

  // Use depth output directly
  const float * depth_ptr = depth_h_.get();
  const size_t plane_size = static_cast<size_t>(height) * width;
  model_depth_.create(height, width, CV_32FC1);
  std::memcpy(model_depth_.data, depth_ptr, plane_size * sizeof(float));

  // Sky output
  sky_mask_.release();
  cv::Mat sky_pred(height, width, CV_32FC1, const_cast<float *>(sky_h_.get()));
  sky_mask_ = sky_pred < sky_threshold_;

  // Inspect raw output before scaling.
  double raw_min = 0.0, raw_max = 0.0;
  cv::minMaxLoc(model_depth_, &raw_min, &raw_max);
  const double raw_mean = cv::mean(model_depth_)[0];
  const int sky_pixel_count = cv::countNonZero(~sky_mask_);
  RCLCPP_INFO(
    rclcpp::get_logger("TensorRTDepthAnything"),
    "Raw net output min/max/mean: %.6f / %.6f / %.6f  sky_pixels: %d / %zu",
    raw_min, raw_max, raw_mean, sky_pixel_count, plane_size);

  // Clean and scale to metric depth.
  cv::Mat depth_map = model_depth_.clone();
  depth_map.setTo(0.0f, depth_map <= 0.0f);

  // Use original intrinsics for metric conversion per spec.
  const double fx = camera_info.k[0] * scale_x_;
  const double fy = camera_info.k[4] * scale_y_;
  const double focal_pixels = 0.5 * (fx + fy);
  const double focal_scale = focal_pixels > 0.0 ? focal_pixels / 300.0 : 1.0;
  depth_map *= static_cast<float>(focal_scale);

  // Handle sky: set sky pixels to max depth derived from non-sky regions.
  if (!sky_mask_.empty()) {
    std::vector<float> valid_depths;
    valid_depths.reserve(plane_size);
    const uint8_t * mask_ptr = sky_mask_.ptr<uint8_t>(0);
    const float * depth_ptr_flat = reinterpret_cast<const float *>(depth_map.data);
    for (size_t idx = 0; idx < plane_size; ++idx) {
      if (mask_ptr[idx]) {
        const float val = depth_ptr_flat[idx];
        if (std::isfinite(val) && val > 0.0f) {
          valid_depths.push_back(val);
        }
      }
    }
    if (!valid_depths.empty()) {
      size_t sample_size = valid_depths.size();
      const size_t max_sample = 100000;
      if (sample_size > max_sample) {
        const size_t step = sample_size / max_sample;
        std::vector<float> sampled;
        sampled.reserve(max_sample);
        for (size_t i = 0; i < sample_size && sampled.size() < max_sample; i += step) {
          sampled.push_back(valid_depths[i]);
        }
        valid_depths.swap(sampled);
      }
      const size_t idx = static_cast<size_t>(0.99 * (valid_depths.size() - 1));
      std::nth_element(valid_depths.begin(), valid_depths.begin() + idx, valid_depths.end());
      const float max_depth = std::min(valid_depths[idx], sky_depth_cap_);
      cv::Mat depth_map_reshaped(height, width, CV_32FC1, depth_map.data);
      depth_map_reshaped.setTo(max_depth, ~sky_mask_);
    }
  }

  // Temporal smoothing (EMA) to stabilize depth across frames.
  if (prev_depth_.empty() || prev_depth_.size() != depth_map.size()) {
    prev_depth_ = depth_map.clone();
  } else {
    constexpr float alpha = 0.3f;  // 0.0 = no change, 1.0 = no smoothing
    cv::addWeighted(depth_map, alpha, prev_depth_, 1.0 - alpha, 0.0, depth_map);
    prev_depth_ = depth_map.clone();
  }

  // Persist scaled depth for point cloud generation at network resolution.
  model_depth_ = depth_map.clone();

  double min_metric = 0.0, max_metric = 0.0;
  cv::minMaxLoc(model_depth_, &min_metric, &max_metric);
  RCLCPP_INFO(
    rclcpp::get_logger("TensorRTDepthAnything"),
    "Metric depth (after scale) min/max: %.6f / %.6f m  focal_px: %.1f  focal_scale: %.4f",
    min_metric, max_metric, focal_pixels, focal_scale);

  cv::resize(model_depth_, depth_image_, cv::Size(src_width_, src_height_), 0, 0, cv::INTER_CUBIC);


  cv::Mat colorized = rgb_image;
  if (!colorized.empty() &&
      (colorized.rows != depth_image_.rows || colorized.cols != depth_image_.cols)) {
    cv::resize(colorized, colorized, depth_image_.size(), 0, 0, cv::INTER_LINEAR);
  }

  buildPointCloud(camera_info, downsample_factor, colorized);
}


void TensorRTDepthAnything::buildPointCloud(
  const sensor_msgs::msg::CameraInfo & camera_info, int downsample_factor,
  const cv::Mat & rgb_image)
{

  // Resize color to match model output if provided.
  cv::Mat color;
  if (!rgb_image.empty()) {
    if (rgb_image.type() == CV_8UC3) {
      color = rgb_image.clone();
    } else {
      rgb_image.convertTo(color, CV_8UC3);
    }
    if (color.rows != model_depth_.rows || color.cols != model_depth_.cols) {
      cv::resize(color, color, model_depth_.size(), 0, 0, cv::INTER_LINEAR);
    }
  }

  // Scale intrinsics to model output resolution.
  sensor_msgs::msg::CameraInfo cam_scaled = camera_info;
  cam_scaled.k[0] = camera_info.k[0] * scale_x_;
  cam_scaled.k[4] = camera_info.k[4] * scale_y_;
  cam_scaled.k[2] = camera_info.k[2] * scale_x_;
  cam_scaled.k[5] = camera_info.k[5] * scale_y_;
  cam_scaled.width = model_depth_.cols;
  cam_scaled.height = model_depth_.rows;

  const std::string frame_id =
    camera_info.header.frame_id.empty() ? "camera_link" : camera_info.header.frame_id;

  depthImageToPointCloud(
    model_depth_, cam_scaled, point_cloud_, frame_id, downsample_factor, color, sky_mask_);

  // Preserve original timestamp
  point_cloud_.header.stamp = camera_info.header.stamp;
}
const cv::Mat& TensorRTDepthAnything::getDepthImage() const
{
  return depth_image_;
}

const sensor_msgs::msg::PointCloud2& TensorRTDepthAnything::getPointCloud() const
{
  return point_cloud_;
}

void TensorRTDepthAnything::printProfiling()
{
  trt_common_->printProfiling();
}

} // namespace depth_anything_v3
