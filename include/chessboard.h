#pragma once

#include <opencv2/opencv.hpp>

class Chessboard {
 public:
  Chessboard(const cv::Mat& img, const cv::Size& pattern_size,
             float square_size);
  void FindChessboardCorners();

  bool Found() const { return found_; }
  std::vector<cv::Point2f> GetCorners() const { return corners_; }

 private:
  cv::Mat img_;
  cv::Size pattern_size_;
  float square_size_;
  cv::Mat sketch_;
  bool found_ = false;
  std::vector<cv::Point2f> corners_;
};
