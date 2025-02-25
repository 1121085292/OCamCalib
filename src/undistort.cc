#include <iostream>

#include "utils.h"

struct Roi {
  int left_bound = 0;
  int righe_bound = 0;
  int top_bound = 0;
  int bottom_bound = 0;
};

int main() {
  std::string calib_file = "";
  OcamModel ocam_model;
  if (!GetOcamModel(calib_file, ocam_model)) {
    std::cerr << "Failed to load ocam model" << std::endl;
  }

  PrintOcamModel(ocam_model);

  cv::Mat mapx(ocam_model.height, ocam_model.width, CV_32FC1);
  cv::Mat mapy(ocam_model.height, ocam_model.width, CV_32FC1);
  CreatePerspeciveUndistortionLUT(ocam_model, 4.0, mapx, mapy);

  cv::Mat img = cv::imread("");

  // 添加roi区域
  Roi roi;
  roi.righe_bound = roi.righe_bound == 0 ? ocam_model.width : roi.righe_bound;
  roi.bottom_bound =
      roi.bottom_bound == 0 ? ocam_model.height : roi.bottom_bound;

  cv::Rect roi_rect(roi.left_bound, roi.top_bound,
                    roi.righe_bound - roi.left_bound,
                    roi.bottom_bound - roi.top_bound);

  cv::Mat undistorted;
  cv::remap(img, undistorted, mapx, mapy, cv::INTER_LINEAR);

  // 调整图像大小输出
  cv::Mat cropped(undistorted, roi_rect);
  cv::Mat output;
  cv::resize(cropped, output, img.size());
  cv::imwrite("undistorted.png", output);
  return 0;
}