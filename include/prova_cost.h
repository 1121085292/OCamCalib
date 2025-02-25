#pragma once
#include <ceres/ceres.h>

#include <Eigen/Dense>

#include "utils.h"

class ExtrinsicCost {
 public:
  ExtrinsicCost(const Eigen::VectorXd& ss, const Eigen::VectorXd& Xp_abs,
                const Eigen::VectorXd& Yp_abs, const Eigen::VectorXd& int_par,
                const Eigen::MatrixXd& M)
      : ss_(ss), Xp_abs_(Xp_abs), Yp_abs_(Yp_abs), int_par_(int_par), M_(M) {
    num_points_ = M.rows();
  }
  bool operator()(const double* const x, double* residuals) const;

  static ceres::CostFunction* Create(const Eigen::VectorXd& ss,
                                     const Eigen::VectorXd& Xp_abs,
                                     const Eigen::VectorXd& Yp_abs,
                                     const Eigen::VectorXd& int_par,
                                     const Eigen::MatrixXd& M) {
    const int num_points = M.rows();
    return new ceres::NumericDiffCostFunction<ExtrinsicCost, ceres::CENTRAL,
                                              ceres::DYNAMIC, 6>(
        new ExtrinsicCost(ss, Xp_abs, Yp_abs, int_par, M),
        ceres::TAKE_OWNERSHIP, num_points);
  }

 private:
  Eigen::VectorXd ss_;
  Eigen::VectorXd Xp_abs_;
  Eigen::VectorXd Yp_abs_;
  Eigen::VectorXd int_par_;
  Eigen::MatrixXd M_;
  int num_points_;
};

class IntrinsicCost {
 public:
  IntrinsicCost(const Eigen::VectorXd& x, const double xc, const double yc,
                const Eigen::VectorXd& ss,
                const std::vector<Eigen::MatrixXd>& RRfin,
                const std::vector<Eigen::VectorXd>& Xp_abs,
                const std::vector<Eigen::VectorXd>& Yp_abs,
                const Eigen::MatrixXd& M)
      : x_(x),
        xc_(xc),
        yc_(yc),
        ss_(ss),
        RRfin_(RRfin),
        Xp_abs_(Xp_abs),
        Yp_abs_(Yp_abs),
        M_(M) {
    images_nums_ = RRfin.size();
    num_points_ = M.rows();
  }

  bool operator()(const double* const x, double* residuals) const;

  static ceres::CostFunction* Create(const Eigen::VectorXd& x, const double xc,
                                     const double yc, const Eigen::VectorXd& ss,
                                     const std::vector<Eigen::MatrixXd>& RRfin,
                                     const std::vector<Eigen::VectorXd>& Xp_abs,
                                     const std::vector<Eigen::VectorXd>& Yp_abs,
                                     const Eigen::MatrixXd& M) {
    const int num_points = M.rows() * Xp_abs.size();
    return new ceres::NumericDiffCostFunction<IntrinsicCost, ceres::CENTRAL,
                                              ceres::DYNAMIC, 10>(
        new IntrinsicCost(x, xc, yc, ss, RRfin, Xp_abs, Yp_abs, M),
        ceres::TAKE_OWNERSHIP, num_points);
  }

 private:
  Eigen::VectorXd x_;
  double xc_;
  double yc_;
  Eigen::VectorXd ss_;
  std::vector<Eigen::MatrixXd> RRfin_;
  std::vector<Eigen::VectorXd> Xp_abs_;
  std::vector<Eigen::VectorXd> Yp_abs_;
  Eigen::MatrixXd M_;
  int images_nums_;
  int num_points_;
};
