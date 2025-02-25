#include "chessboard.h"

Chessboard::Chessboard(const cv::Mat& img, const cv::Size& pattern_size,
                       float square_size)
    : pattern_size_(pattern_size), square_size_(square_size) {
  if (img.channels() == 1) {
    cv::cvtColor(img, sketch_, cv::COLOR_GRAY2BGR);
    img.copyTo(img_);
  } else {
    img.copyTo(sketch_);
    cv::cvtColor(img, img_, cv::COLOR_BGR2GRAY);
  }
}

void Chessboard::FindChessboardCorners() {
  found_ = cv::findChessboardCorners(
      img_, pattern_size_, corners_,
      cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
          cv::CALIB_CB_FILTER_QUADS | cv::CALIB_CB_FAST_CHECK);
  if (found_) {
    cv::cornerSubPix(
        img_, corners_, cv::Size(5, 5), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30,
                         0.1));
    cv::drawChessboardCorners(sketch_, pattern_size_, corners_, found_);
    cv::imshow("chessboard", sketch_);
    cv::waitKey(50);
  }
}
