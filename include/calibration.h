#pragma once
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "calib_data.h"
#include "prova_cost.h"
#include "utils.h"

class Calibration {
 public:
  Calibration(const std::string& output_file_format,
              const std::string& output_file_prefix_,
              const cv::Size& board_size, const cv::Size& image_size,
              float square_size, int taylor_order);

  void Print();

  void CalibrateHelper();

  void Calibrate();

  void AddChessboardCorners(const std::vector<cv::Point2f>& corners);

  void ReprojectPoints();

  void FindCenter();

  void Optimize();

  bool ExportData();

 private:
  int PlotRR(const std::vector<Eigen::Matrix3d>& rr, const Eigen::VectorXd& Xpt,
             const Eigen::VectorXd& Ypt);

  void OmniFindParametersFun();

  double ReprojectPointsFun();

  void ReprojectpointsQuiet(Eigen::MatrixXd& M);

  Eigen::VectorXd FindInvPoly(double radius);

  void FindInvPolyHelper(double radius, int n, Eigen::VectorXd& inv_poly,
                         Eigen::VectorXd& err);

  std::string output_file_format_;
  std::string output_file_prefix_;
  cv::Size board_size_;
  cv::Size image_size_;
  float square_size_;
  int taylor_order_;
  size_t size_;
  std::shared_ptr<CalibData> calib_data_;
};