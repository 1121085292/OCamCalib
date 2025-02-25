#include "prova_cost.h"

bool ExtrinsicCost::operator()(const double* const x, double* residuals) const {
  // 提取内参
  double c = int_par_(0);
  double d = int_par_(1);
  double e = int_par_(2);
  double xc = int_par_(3);
  double yc = int_par_(4);

  // 解析优化变量：旋转向量和平移向量
  Eigen::Map<const Eigen::Vector3d> rvec(x);
  Eigen::Map<const Eigen::Vector3d> t(x + 3);

  // 计算旋转矩阵
  Eigen::Matrix3d R;
  std::tie(R, std::ignore) = RodroguesFromRotationVector(rvec);

  // 转换3D点
  Eigen::MatrixXd Mc(3, num_points_);
  for (int i = 0; i < num_points_; ++i) {
    Mc.col(i) = R * M_.row(i).transpose() + t;
  }

  // 投影到像素坐标
  Eigen::VectorXd xp1(num_points_), yp1(num_points_);
  Omni3dToPixel(Mc, ss_, xp1, yp1);

  // 应用内参变换
  Eigen::VectorXd xp = xp1.array() * c + yp1.array() * d + xc;
  Eigen::VectorXd yp = xp1.array() * e + yp1.array() + yc;

  // 计算残差
  for (int i = 0; i < num_points_; ++i) {
    residuals[i] = sqrt((Xp_abs_(i) - xp[i]) * (Xp_abs_(i) - xp[i]) +
                        (Yp_abs_(i) - yp[i]) * (Yp_abs_(i) - yp[i]));
  }
  return true;
}

bool IntrinsicCost::operator()(const double* const x, double* residuals) const {
  // 解析 x（优化变量）
  double a = x[0];
  double b = x[1];
  double c = x[2];
  double d = x[3];
  double e = x[4];

  Eigen::Map<const Eigen::VectorXd> x_map(x_.data(), x_.size());
  assert(x_map.size() >= 5);
  Eigen::VectorXd ssc = x_map.tail(x_map.size() - 5);
  Eigen::MatrixXd M = M_;
  M.col(2).setOnes();
  // 计算世界坐标转换
  Eigen::MatrixXd Mc(3, num_points_ * images_nums_);
  Eigen::VectorXd Xpp(M_.rows() * images_nums_);
  Eigen::VectorXd Ypp(M_.rows() * images_nums_);
  // 计算旋转变换
  for (int i = 0; i < images_nums_; ++i) {
    Mc.block(0, i * M_.rows(), M_.cols(), M_.rows()) =
        RRfin_[i] * M.transpose();
    Xpp.block(i * M_.rows(), 0, Xp_abs_[i].rows(), 1) = Xp_abs_[i];
    Ypp.block(i * M_.rows(), 0, Yp_abs_[i].rows(), 1) = Yp_abs_[i];
  }
  // 计算投影
  Eigen::VectorXd xp1(num_points_ * images_nums_),
      yp1(num_points_ * images_nums_);
  Eigen::VectorXd new_ss = ss_.array() * ssc.array();
  Omni3dToPixel(Mc, new_ss, xp1, yp1);
  // 计算像素坐标变换
  Eigen::VectorXd xp = xp1.array() * c + yp1.array() * d + xc_ * a;
  Eigen::VectorXd yp = xp1.array() * e + yp1.array() + yc_ * b;
  // 计算误差（残差）
  for (int i = 0; i < num_points_ * images_nums_; ++i) {
    residuals[i] = sqrt((Xpp(i) - xp(i)) * (Xpp(i) - xp(i)) +
                        (Ypp(i) - yp(i)) * (Ypp(i) - yp(i)));
  }
  return true;
}