#pragma once

#include <Eigen/Dense>
#include <vector>

struct OcamModel {
  Eigen::VectorXd ss;  // 畸变系数
  Eigen::VectorXd inv_poly;
  int width;
  int height;
  double xc;
  double yc;
  double c = 1.0;
  double d = 0.0;
  double e = 0.0;
};

struct CalibData {
  Eigen::VectorXd Xt;  // 世界坐标 X
  Eigen::VectorXd Yt;  // 世界坐标 Y

  std::vector<Eigen::VectorXd> Xp_abs;  // 所有图像中角点坐标 X
  std::vector<Eigen::VectorXd> Yp_abs;  // 所有图像中角点坐标 Y

  std::vector<Eigen::MatrixXd> RRfin;
  OcamModel ocam_model;
};