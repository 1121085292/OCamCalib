#include <filesystem>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "calibration.h"
#include "chessboard.h"
#include "params_gflags.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
  ParseShortFlags(&argc, &argv);
  // 初始化 gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::vector<std::string> image_files;
  for (const auto& entry : fs::directory_iterator(FLAGS_input_file)) {
    if (entry.path().extension() == ".jpg") {
      image_files.push_back(entry.path());
    }
  }
  // 读取图像文件并初始化校准参数
  cv::Mat image = cv::imread(image_files[0]);
  cv::Size frame_size = image.size();
  cv::Size pattern_size = cv::Size(FLAGS_width, FLAGS_height);
  Calibration calibration(FLAGS_output_file_format, FLAGS_output_file_prefix,
                          pattern_size, frame_size, FLAGS_square_size,
                          FLAGS_taylor_order);

  for (size_t i = 0; i < image_files.size(); ++i) {
    cv::Mat image = cv::imread(image_files[i]);
    // 角点检测和校准
    Chessboard chessboard(image, pattern_size, FLAGS_square_size);
    chessboard.FindChessboardCorners();

    if (chessboard.Found()) {
      calibration.AddChessboardCorners(chessboard.GetCorners());
    }
  }

  calibration.CalibrateHelper();
  // debug
  // calibration.Print();

  // 清理 gflags
  gflags::ShutDownCommandLineFlags();
  return 0;
}