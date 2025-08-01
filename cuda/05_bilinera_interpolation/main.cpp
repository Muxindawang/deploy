#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include "timer.hpp"
#include "preprocess.hpp"

int main() {
  Timer timer;

  std::string file_path = "data/deer.png";
  std::string output_prefix = "results/";
  std::string output_path = "";

  cv::Mat input = cv::imread(file_path);
  int tar_h = 500, tar_w = 250;
  int tactis;

  cv::Mat resizedInput_cpu;
  cv::Mat resizedInput_gpu;

  resizedInput_cpu = preprocess_cpu(input, tar_h, tar_w, timer, tactis);
  output_path = output_prefix + getPrefix(file_path) + "_resized_bilinear_cpu.png";
  cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
  cv::imwrite(output_path, resizedInput_cpu);

  tactis = 0;
  resizedInput_gpu = preprocess_gpu(input, tar_h, tar_w, timer, tactis);
  output_path = output_prefix + getPrefix(file_path) + "_resized_nearest_gpu.png";
  cv::cvtColor(resizedInput_cpu, resizedInput_cpu, cv::COLOR_RGB2BGR);
  cv::imwrite(output_path, resizedInput_gpu);

  printf("save as ", output_path);
  return 0;
}