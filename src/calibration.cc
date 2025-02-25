#include "calibration.h"

Calibration::Calibration(const std::string& output_file_format,
                         const std::string& output_file_prefix,
                         const cv::Size& board_size, const cv::Size& image_size,
                         float square_size, int taylor_order)
    : output_file_format_(output_file_format),
      output_file_prefix_(output_file_prefix),
      board_size_(board_size),
      image_size_(image_size),
      square_size_(square_size),
      taylor_order_(taylor_order) {
  size_ = board_size_.width * board_size_.height;
  calib_data_.reset(new CalibData);
  calib_data_->Xt.resize(size_);
  calib_data_->Yt.resize(size_);
  calib_data_->ocam_model.xc = std::round(image_size_.height / 2.0);
  calib_data_->ocam_model.yc = std::round(image_size_.width / 2.0);
  calib_data_->ocam_model.height = image_size_.height;
  calib_data_->ocam_model.width = image_size_.width;
}

void Calibration::Print() {
  // 打印calib_data信息
  std::cout << "ss:\n" << calib_data_->ocam_model.ss << std::endl;
  std::cout << "xc: " << calib_data_->ocam_model.xc
            << ", yc: " << calib_data_->ocam_model.yc << std::endl;
  std::cout << "c: " << calib_data_->ocam_model.c
            << ", d: " << calib_data_->ocam_model.d
            << ", e: " << calib_data_->ocam_model.e << std::endl;
}

void Calibration::CalibrateHelper() {
  // 1.calibrate
  Calibrate();
  ReprojectPoints();
  // 2.find center
  FindCenter();
  // 3.optimize
  Optimize();
  // 4.export data
  ExportData();
}

void Calibration::Calibrate() {
  // 计算 Xp 和 Yp（图像坐标归一化）
  size_t num_images = calib_data_->Xp_abs.size();
  std::vector<Eigen::VectorXd> Yp(num_images);
  std::vector<Eigen::VectorXd> Xp(num_images);
  for (size_t i = 0; i < num_images; ++i) {
    Yp[i] = calib_data_->Yp_abs[i].array() - calib_data_->ocam_model.yc;
    Xp[i] = calib_data_->Xp_abs[i].array() - calib_data_->ocam_model.xc;
  }
  // 计算旋转和平移矩阵
  for (size_t i = 0; i < num_images; ++i) {
    const Eigen::VectorXd& Xpt = Xp[i];
    const Eigen::VectorXd& Ypt = Yp[i];

    // 构建A矩阵
    Eigen::MatrixXd A(calib_data_->Xt.size(), 6);
    A.col(0) = calib_data_->Xt.array() * Ypt.array();
    A.col(1) = calib_data_->Yt.array() * Ypt.array();
    A.col(2) = -calib_data_->Xt.array() * Xpt.array();
    A.col(3) = -calib_data_->Yt.array() * Xpt.array();
    A.col(4) = Ypt;
    A.col(5) = -Xpt;

    // svd求解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd V = svd.matrixV().col(5);
    // 解析出旋转和平移参数
    double r11 = V(0), r12 = V(1), r21 = V(2), r22 = V(3), t1 = V(4), t2 = V(5);
    double AA = std::pow((r11 * r12 + r21 * r22), 2);
    double BB = r11 * r11 + r21 * r21;
    double CC = r12 * r12 + r22 * r22;

    double r32_2_squared_1 =
        (-(CC - BB) + std::sqrt(std::pow((CC - BB), 2) + 4 * AA)) / 2.0;
    double r32_2_squared_2 =
        (-(CC - BB) - std::sqrt(std::pow((CC - BB), 2) + 4 * AA)) / 2.0;

    std::vector<double> r32_2;
    if (r32_2_squared_1 >= 0) r32_2.push_back(r32_2_squared_1);
    if (r32_2_squared_2 >= 0) r32_2.push_back(r32_2_squared_2);
    // 计算旋转矩阵 R
    std::vector<double> r31, r32;
    std::vector<double> sg = {1.0, -1.0};
    for (size_t i = 0; i < r32_2.size(); ++i) {
      for (size_t j = 0; j < sg.size(); ++j) {
        double sqrtR32_2 = sg[j] * std::sqrt(r32_2[i]);
        r32.push_back(sqrtR32_2);
        if (r32_2[i] < 0.00000001 * (r12 * r12 + r22 * r22)) {
          r31.push_back(std::sqrt(CC - BB));
          r31.push_back(-std::sqrt(CC - BB));
          r32.push_back(sqrtR32_2);
        } else {
          r31.push_back(-(r11 * r12 + r21 * r22) / sqrtR32_2);
        }
      }
    }
    // 计算最终旋转矩阵
    std::vector<Eigen::Matrix3d> rr;
    for (size_t i = 0; i < r32.size(); ++i) {
      for (size_t j = 0; j < sg.size(); ++j) {
        double Lb = 1.0 / sqrt(r11 * r11 + r21 * r21 + r31[i] * r31[i]);
        Eigen::Matrix3d matrix;
        matrix << sg[j] * Lb * r11, sg[j] * Lb * r12, sg[j] * Lb * t1,
            sg[j] * Lb * r21, sg[j] * Lb * r22, sg[j] * Lb * t2,
            sg[j] * Lb * r31[i], sg[j] * Lb * r32[i], 0;
        rr.emplace_back(matrix);
      }
    }
    // 选取最佳旋转矩阵
    std::vector<Eigen::Matrix3d> rr1;
    double min_rr = std::numeric_limits<double>::infinity();
    int min_rr_ind = -1;
    for (size_t min_count = 0; min_count < rr.size(); ++min_count) {
      Eigen::Vector2d point1(rr[min_count](0, 2), rr[min_count](1, 2));
      Eigen::Vector2d point2(Xpt(0), Ypt(0));
      double dist = (point1 - point2).norm();

      if (dist < min_rr) {
        min_rr = dist;
        min_rr_ind = min_count;
      }
    }

    if (min_rr_ind != -1) {
      for (size_t i = 0; i < rr.size(); ++i) {
        if (rr[i](0, 2) * rr[min_rr_ind](0, 2) > 0 &&
            rr[i](1, 2) * rr[min_rr_ind](1, 2) > 0) {
          rr1.emplace_back(rr[i]);
        }
      }
    }

    if (rr1.empty()) {
      calib_data_->ocam_model.ss = Eigen::VectorXd::Zero(0);
      return;
    }

    int nm = PlotRR(rr1, Xpt, Ypt);
    Eigen::Matrix3d RRdef = rr1[nm];
    calib_data_->RRfin.push_back(RRdef);
  }
  OmniFindParametersFun();
}

void Calibration::AddChessboardCorners(
    const std::vector<cv::Point2f>& corners) {
  assert(corners.size() == size_);
  Eigen::VectorXd Xp(size_);
  Eigen::VectorXd Yp(size_);

  for (int i = 0; i < board_size_.height; ++i) {
    for (int j = 0; j < board_size_.width; ++j) {
      Xp(i * board_size_.width + j) = corners[i * board_size_.width + j].x;
      Yp(i * board_size_.width + j) = corners[i * board_size_.width + j].y;
      // 世界坐标
      calib_data_->Xt(i * board_size_.width + j) = square_size_ * i;
      calib_data_->Yt(i * board_size_.width + j) = square_size_ * j;
    }
  }
  // 所有图像中角点坐标
  calib_data_->Xp_abs.push_back(Yp);
  calib_data_->Yp_abs.push_back(Xp);
}

void Calibration::ReprojectPoints() {
  size_t num_images = calib_data_->Xp_abs.size();
  std::vector<double> err;
  std::vector<double> stderr;
  double MSE = 0;
  // 遍历每张图像
  for (size_t i = 0; i < num_images; ++i) {
    Eigen::MatrixXd mat(3, calib_data_->Xt.rows());
    mat.row(0) = calib_data_->Xt.transpose();
    mat.row(1) = calib_data_->Yt.transpose();
    mat.row(2) = Eigen::VectorXd::Ones(calib_data_->Xt.rows());
    Eigen::MatrixXd xx = calib_data_->RRfin[i] * mat;
    Eigen::VectorXd x, y;
    // 转换到图像坐标
    Omni3dToPixel(xx, calib_data_->ocam_model.ss, x, y);
    // 计算重投影误差
    auto stt = ((calib_data_->Xp_abs[i].array() - calib_data_->ocam_model.xc -
                 x.array())
                    .square() +
                (calib_data_->Yp_abs[i].array() - calib_data_->ocam_model.yc -
                 y.array())
                    .square())
                   .sqrt();
    err.emplace_back(stt.mean());
    stderr.emplace_back(std::sqrt((stt.array() - stt.mean()).square().mean()));
    // 计算统计误差
    MSE += stt.array().square().sum();
  }
}

void Calibration::FindCenter() {
  // 获取当前相机的标定参数
  double pxc = calib_data_->ocam_model.xc;
  double pyc = calib_data_->ocam_model.yc;
  double width = calib_data_->ocam_model.width;
  double height = calib_data_->ocam_model.height;
  double regwidth = width / 2.0;
  double regheight = height / 2.0;
  int xceil = 5;
  int yceil = 5;
  // 计算搜索区域边界
  double xregstart = pxc - regheight / 2.0;
  double xregstop = pxc + regheight / 2.0;
  double yregstart = pyc - regwidth / 2.0;
  double yregstop = pyc + regwidth / 2.0;
  // 进行多轮网格搜索
  for (int glc = 0; glc < 9; glc++) {
    // 计算步长
    double ystep = (yregstop - yregstart) / yceil;
    double xstep = (xregstop - xregstart) / xceil;
    // 生成 x 和 y 向量
    Eigen::MatrixXd yreg, xreg;
    int ysize = yceil + 1;
    int xsize = xceil + 1;
    yreg.resize(ysize, xsize);
    xreg.resize(ysize, xsize);
    for (int i = 0; i < ysize; i++) {
      for (int j = 0; j < xsize; j++) {
        yreg(i, j) = yregstart + i * ystep;
        xreg(i, j) = xregstart + j * xstep;
      }
    }

    Eigen::MatrixXd MSEA = Eigen::MatrixXd::Constant(
        xreg.rows(), xreg.cols(), std::numeric_limits<double>::infinity());

    for (int i = 0; i < static_cast<int>(xreg.rows()); i++) {
      for (int j = 0; j < static_cast<int>(xreg.cols()); j++) {
        calib_data_->ocam_model.xc = xreg(i, j);
        calib_data_->ocam_model.yc = yreg(i, j);
        Calibrate();
        if (calib_data_->RRfin.empty()) {
          MSEA(i, j) = std::numeric_limits<double>::infinity();
          continue;
        }
        double MSE = ReprojectPointsFun();
        if (!std::isnan(MSE)) {
          MSEA(i, j) = MSE;
        }
      }
    }

    // 找到 MSEA 中的最小值及其位置
    Eigen::MatrixXd::Index row, col;
    MSEA.minCoeff(&row, &col);

    // 更新 calib_data
    calib_data_->ocam_model.xc = xreg(row, col);
    calib_data_->ocam_model.yc = yreg(row, col);

    // 计算步长
    double dx_reg = std::abs((xregstop - xregstart) / xceil);
    double dy_reg = std::abs((yregstop - yregstart) / yceil);

    // 更新网格范围
    xregstart = calib_data_->ocam_model.xc - dx_reg;
    xregstop = calib_data_->ocam_model.xc + dx_reg;
    yregstart = calib_data_->ocam_model.yc - dy_reg;
    yregstop = calib_data_->ocam_model.yc + dy_reg;
  }
  Calibrate();
  ReprojectPoints();
}

void Calibration::Optimize() {
  // 设定优化的终止条件
  double tol_MSE = 1e-4;
  double MSE_old = 0;
  double MSE_new = std::numeric_limits<double>::infinity();
  int iter = 0;
  int max_iter = 100;

  Eigen::MatrixXd M(calib_data_->Xt.size(), 3);
  while (iter < max_iter && std::fabs(MSE_new - MSE_old) > tol_MSE) {
    iter++;
    Eigen::VectorXd int_par(5);
    int_par << calib_data_->ocam_model.c, calib_data_->ocam_model.d,
        calib_data_->ocam_model.e, calib_data_->ocam_model.xc,
        calib_data_->ocam_model.yc;

    M.block(0, 0, calib_data_->Xt.size(), 1) = calib_data_->Xt;
    M.block(0, 1, calib_data_->Yt.size(), 1) = calib_data_->Yt;
    M.block(0, 2, calib_data_->Xt.size(), 1) =
        Eigen::VectorXd::Zero(calib_data_->Xt.size());

    std::vector<Eigen::MatrixXd> RRfinOpt;
    // 外参优化
    for (size_t i = 0; i < calib_data_->Xp_abs.size(); ++i) {
      // 提取相机外参
      Eigen::MatrixXd R = calib_data_->RRfin[i];
      // 修正旋转矩阵，使其正交化
      assert(R.cols() == 3);
      Eigen::Vector3d v0 = R.col(0);
      Eigen::Vector3d v1 = R.col(1);
      Eigen::Vector3d v2 = v0.cross(v1);
      R.col(2) = v2;
      // 构造优化变量
      Eigen::Vector3d r, t;
      std::tie(r, std::ignore) = RodroguesFromRotationMatrix(R);
      t = calib_data_->RRfin[i].col(2);
      Eigen::VectorXd x0(6);
      x0 << r, t;

      // 创建问题
      ceres::Problem problem;

      // 设置优化选项
      ceres::Solver::Options options;
      options.max_num_iterations = 1000;
      options.minimizer_progress_to_stdout = true;

      // 添加残差项
      ceres::CostFunction* cost_function = ExtrinsicCost::Create(
          calib_data_->ocam_model.ss, calib_data_->Xp_abs[i],
          calib_data_->Yp_abs[i], int_par, M);
      problem.AddResidualBlock(cost_function, nullptr, x0.data());

      // 求解问题
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);

      // 更新外参
      Eigen::Matrix3d rotation;
      std::tie(rotation, std::ignore) = RodroguesFromRotationVector(x0.head(3));
      rotation.col(2) = x0.tail(3);
      RRfinOpt.emplace_back(rotation);
    }
    calib_data_->RRfin = RRfinOpt;

    // 内参优化
    Eigen::VectorXd f0(5 + calib_data_->ocam_model.ss.size());
    f0 << 1, 1, calib_data_->ocam_model.c, calib_data_->ocam_model.d,
        calib_data_->ocam_model.e,
        Eigen::VectorXd::Ones(calib_data_->ocam_model.ss.size());

    Eigen::VectorXd lb(5 + calib_data_->ocam_model.ss.size());
    lb << 0, 0, 0, -1, -1,
        Eigen::VectorXd::Zero(calib_data_->ocam_model.ss.size());

    Eigen::VectorXd ub(5 + calib_data_->ocam_model.ss.size());
    ub << 2, 2, 2, 1, 1,
        Eigen::VectorXd::Ones(calib_data_->ocam_model.ss.size()) * 2;

    Eigen::VectorXd ss0 = calib_data_->ocam_model.ss;
    // 创建问题
    ceres::Problem problem;

    // 设置优化选项
    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.minimizer_progress_to_stdout = true;

    // 添加残差块
    ceres::CostFunction* cost_function = IntrinsicCost::Create(
        f0, calib_data_->ocam_model.xc, calib_data_->ocam_model.yc, ss0,
        calib_data_->RRfin, calib_data_->Xp_abs, calib_data_->Yp_abs, M);
    problem.AddResidualBlock(cost_function, nullptr, f0.data());

    // 求解问题
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 更新内参
    calib_data_->ocam_model.ss = ss0.array() * f0.tail(f0.size() - 5).array();
    calib_data_->ocam_model.xc *= f0(0);
    calib_data_->ocam_model.yc *= f0(1);
    calib_data_->ocam_model.c = f0(2);
    calib_data_->ocam_model.d = f0(3);
    calib_data_->ocam_model.e = f0(4);

    MSE_old = MSE_new;
  }
  // 计算最终误差
  ReprojectpointsQuiet(M);
}

bool Calibration::ExportData() {
  // 计算逆映射 inv_poly
  calib_data_->ocam_model.inv_poly =
      FindInvPoly(std::sqrt(pow(calib_data_->ocam_model.width / 2, 2) +
                            pow(calib_data_->ocam_model.height / 2, 2)));
  if (output_file_format_ == "txt") {
    // TXT 格式导出
    std::ofstream outfile;
    outfile.open(output_file_prefix_ + "_calibration_data.txt");
    outfile << "#polynomial coefficients for the DIRECT mapping function.These "
               "are used by cam2world\n\n";
    outfile << calib_data_->ocam_model.ss.size() << " "
            << calib_data_->ocam_model.ss.transpose() << "\n\n";
    outfile
        << "#polynomial coefficients for the INVERSE mapping function.These "
           "are used by world2cam\n\n";
    outfile << calib_data_->ocam_model.inv_poly.size() << " "
            << calib_data_->ocam_model.inv_poly.transpose() << "\n\n";
    outfile << "#camera center in pixel coordinates\n\n";
    outfile << calib_data_->ocam_model.xc - 1 << " "
            << calib_data_->ocam_model.yc - 1 << "\n\n";
    outfile << "#affine parameters c, d, e\n\n";
    outfile << calib_data_->ocam_model.c << " " << calib_data_->ocam_model.d
            << " " << calib_data_->ocam_model.e << "\n\n";
    outfile << "#image size: height and width\n\n";
    outfile << calib_data_->ocam_model.height << " "
            << calib_data_->ocam_model.width;
    outfile.close();
  } else if (output_file_format_ == "yaml") {
    // YAML 格式导出
    cv::FileStorage fs(output_file_prefix_ + "_calibration_data.yaml",
                       cv::FileStorage::WRITE);
    fs << "height" << calib_data_->ocam_model.height;
    fs << "width" << calib_data_->ocam_model.width;

    cv::Mat poly_coeffs(5, 1, CV_64F, calib_data_->ocam_model.ss.data());
    fs << "poly_coeffs" << poly_coeffs;

    cv::Mat inv_poly(calib_data_->ocam_model.inv_poly.size(), 1, CV_64F,
                     calib_data_->ocam_model.inv_poly.data());
    fs << "inv_poly" << inv_poly;

    Eigen::Vector2d center(calib_data_->ocam_model.xc - 1,
                           calib_data_->ocam_model.yc - 1);
    cv::Mat center_mat(2, 1, CV_64F, center.data());
    fs << "center" << center_mat;

    Eigen::Matrix2d affine;
    affine << calib_data_->ocam_model.c, calib_data_->ocam_model.d,
        calib_data_->ocam_model.e, 1.0;
    cv::Mat affine_mat(2, 2, CV_64F, affine.data());
    fs << "affine" << affine_mat;

    fs.release();
  } else {
    std::cout << "Unsupported output file format" << std::endl;
    return false;
  }
  return true;
}

int Calibration::PlotRR(const std::vector<Eigen::Matrix3d>& rr,
                        const Eigen::VectorXd& Xpt,
                        const Eigen::VectorXd& Ypt) {
  int res = -1;
  for (size_t i = 0; i < rr.size(); ++i) {
    // 提取旋转矩阵中的参数
    Eigen::MatrixXd rrdef = rr[i];
    double r11 = rrdef(0, 0);
    double r21 = rrdef(1, 0);
    double r31 = rrdef(2, 0);
    double r12 = rrdef(0, 1);
    double r22 = rrdef(1, 1);
    double r32 = rrdef(2, 1);
    double t1 = rrdef(0, 2);
    double t2 = rrdef(1, 2);

    Eigen::VectorXd ma =
        r21 * calib_data_->Xt.array() + r22 * calib_data_->Yt.array() + t2;
    Eigen::VectorXd mb = Ypt.array() * (r31 * calib_data_->Xt.array() +
                                        r32 * calib_data_->Yt.array());
    Eigen::VectorXd mc =
        r11 * calib_data_->Xt.array() + r12 * calib_data_->Yt.array() + t1;
    Eigen::VectorXd md = Xpt.array() * (r31 * calib_data_->Xt.array() +
                                        r32 * calib_data_->Yt.array());
    Eigen::VectorXd rho = (Xpt.array().square() + Ypt.array().square()).sqrt();
    Eigen::VectorXd rho2 = (Xpt.array().square() + Ypt.array().square());
    // 计算投影模型的矩阵 PP1
    Eigen::MatrixXd PP1(ma.rows() + mc.rows(), 3);
    PP1.block(0, 0, ma.rows(), 1) = ma;
    PP1.block(0, 1, ma.rows(), 1) = ma.array() * rho.array();
    PP1.block(0, 2, ma.rows(), 1) = ma.array() * rho2.array();
    PP1.block(ma.rows(), 0, mc.rows(), 1) = mc;
    PP1.block(ma.rows(), 1, mc.rows(), 1) = mc.array() * rho.array();
    PP1.block(ma.rows(), 2, mc.rows(), 1) = mc.array() * rho2.array();
    // 构造 PP
    Eigen::MatrixXd PP(PP1.rows(), 4);
    PP.block(0, 0, PP1.rows(), 3) = PP1;
    PP.block(0, 3, Ypt.rows(), 1) = -Ypt;
    PP.block(Ypt.rows(), 0, Xpt.rows(), 1) = -Xpt;
    // 构造 QQ
    Eigen::VectorXd QQ(mb.size() + md.size());
    QQ << mb, md;
    // SVD 计算最优解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        PP, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd s = svd.solve(QQ);
    // 选择最佳 R
    Eigen::VectorXd ss = s.head(3);
    if (ss(2) >= 0) {
      res = i;
    }
  }
  return res;
}

void Calibration::OmniFindParametersFun() {
  size_t num_images = calib_data_->Xp_abs.size();
  std::vector<Eigen::VectorXd> Yp(num_images);
  std::vector<Eigen::VectorXd> Xp(num_images);
  // 归一化角点坐标 (Xp, Yp)
  for (size_t i = 0; i < num_images; ++i) {
    Yp[i] = calib_data_->Yp_abs[i].array() - calib_data_->ocam_model.yc;
    Xp[i] = calib_data_->Xp_abs[i].array() - calib_data_->ocam_model.xc;
  }
  int min_order = 4;
  if (taylor_order_ <= min_order) {
    // 初始化 PP 和 QQ
    Eigen::MatrixXd PP;
    Eigen::VectorXd QQ;
    for (size_t i = 0; i < num_images; ++i) {
      // 遍历所有标定图像，计算畸变模型
      Eigen::Matrix3d RRdef = calib_data_->RRfin[i];
      double r11 = RRdef(0, 0), r21 = RRdef(1, 0), r31 = RRdef(2, 0),
             r12 = RRdef(0, 1), r22 = RRdef(1, 1), r32 = RRdef(2, 1),
             t1 = RRdef(0, 2), t2 = RRdef(1, 2);

      const Eigen::MatrixXd& Xpt = Xp[i];
      const Eigen::MatrixXd& Ypt = Yp[i];

      Eigen::VectorXd ma =
          r21 * calib_data_->Xt.array() + r22 * calib_data_->Yt.array() + t2;
      Eigen::VectorXd mb = Ypt.array() * (r31 * calib_data_->Xt.array() +
                                          r32 * calib_data_->Yt.array());
      Eigen::VectorXd mc =
          r11 * calib_data_->Xt.array() + r12 * calib_data_->Yt.array() + t1;
      Eigen::VectorXd md = Xpt.array() * (r31 * calib_data_->Xt.array() +
                                          r32 * calib_data_->Yt.array());

      std::vector<Eigen::MatrixXd> rho;
      rho.emplace_back(Eigen::MatrixXd::Ones(Xpt.rows(), Xpt.cols()));
      for (int i = 2; i <= taylor_order_; ++i) {
        rho.emplace_back((Xpt.array().square() + Ypt.array().square())
                             .array()
                             .sqrt()
                             .pow(i));
      }
      // 构造 PP1
      Eigen::MatrixXd PP1(ma.rows() + mc.rows(), taylor_order_);
      PP1.block(0, 0, ma.rows(), 1) = ma;
      PP1.block(ma.rows(), 0, mc.rows(), 1) = mc;
      for (int j = 1; j < taylor_order_; ++j) {
        Eigen::VectorXd vec(ma.rows() + mc.rows());
        vec.block(0, 0, ma.rows(), 1) = ma.array() * rho[j].array();
        vec.block(ma.rows(), 0, mc.rows(), 1) = mc.array() * rho[j].array();
        PP1.block(0, j, ma.rows() + mc.rows(), 1) = vec;
      }
      // 更新 PP
      Eigen::MatrixXd new_pp(PP.rows() + PP1.rows(), PP1.cols() + i + 1);
      new_pp.block(0, 0, PP.rows(), PP.cols()) = PP;
      new_pp.block(0, PP.cols(), PP.rows(), 1) =
          Eigen::VectorXd::Zero(PP.rows(), 1);
      new_pp.block(PP.rows(), 0, PP1.rows(), PP1.cols()) = PP1;
      new_pp.block(PP.rows(), PP1.cols(), PP1.rows(), i) =
          Eigen::MatrixXd::Zero(PP1.rows(), i);
      new_pp.block(PP.rows(), PP1.cols() + i, Ypt.rows(), 1) = -Ypt;
      new_pp.block(PP.rows() + Ypt.rows(), PP1.cols() + i, Xpt.rows(), 1) =
          -Xpt;
      PP = new_pp;

      Eigen::MatrixXd new_qq(QQ.rows() + mb.rows() + md.rows(), 1);
      new_qq.block(0, 0, QQ.rows(), QQ.cols()) = QQ;
      new_qq.block(QQ.rows(), 0, mb.rows(), mb.cols()) = mb;
      new_qq.block(QQ.rows() + mb.rows(), 0, md.rows(), md.cols()) = md;
      QQ = new_qq;
    }
    // SVD 求解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        PP, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXd s = svd.solve(QQ);
    Eigen::VectorXd ss = s.head(taylor_order_);
    // 更新旋转矩阵
    for (size_t i = 0; i < num_images; ++i) {
      calib_data_->RRfin[i](2, 2) = s(ss.size() + i);
    }
    // 补充径向畸变参数
    Eigen::VectorXd new_ss(ss.size() + 1);
    new_ss << ss(0), 0, ss.tail(ss.size() - 1);
    calib_data_->ocam_model.ss = new_ss;
  }
}

double Calibration::ReprojectPointsFun() {
  // clang-format off
  /*
  m=[];
  xx=[];
  err=[];
  stderr=[];
  rhos=[];
  MSE=0;
  for i=ima_proc
      xx=RRfin(:,:,i)*[Xt';Yt';ones(size(Xt'))];
      [Xp_reprojected,Yp_reprojected]=omni3d2pixel(ss,xx, width, height); %convert 3D coordinates in 2D pixel coordinates
      if Xp_reprojected==NaN
          MSE=NaN;
          return;
      end        
      stt= sqrt( (Xp_abs(:,:,i)-xc-Xp_reprojected').^2 + (Yp_abs(:,:,i)-yc-Yp_reprojected').^2 ) ;
      err(i)=(mean(stt));
      stderr(i)=std(stt);
      MSE=MSE+sum( (  Xp_abs(:,:,i)-xc-Xp_reprojected').^2 + (Yp_abs(:,:,i)-yc-Yp_reprojected').^2 );
  end
  */
  // clang-format on
  size_t num_images = calib_data_->Xp_abs.size();
  std::vector<double> err;
  std::vector<double> stderr;
  double MSE = 0;
  for (size_t i = 0; i < num_images; ++i) {
    Eigen::MatrixXd mat(3, calib_data_->Xt.rows());
    mat.row(0) = calib_data_->Xt.transpose();
    mat.row(1) = calib_data_->Yt.transpose();
    mat.row(2) = Eigen::VectorXd::Ones(calib_data_->Xt.rows());
    Eigen::MatrixXd xx = calib_data_->RRfin[i] * mat;
    Eigen::VectorXd x, y;
    Omni3dToPixel(xx, calib_data_->ocam_model.ss, x, y);
    if ((x.array() != x.array()).any()) {
      MSE = std::numeric_limits<double>::quiet_NaN();
    }
    auto stt = ((calib_data_->Xp_abs[i].array() - calib_data_->ocam_model.xc -
                 x.array())
                    .square() +
                (calib_data_->Yp_abs[i].array() - calib_data_->ocam_model.yc -
                 y.array())
                    .square())
                   .sqrt();
    err.emplace_back(stt.mean());
    stderr.emplace_back(std::sqrt((stt.array() - stt.mean()).square().mean()));
    MSE += stt.array().square().sum();
  }
  return MSE;
}

void Calibration::ReprojectpointsQuiet(Eigen::MatrixXd& M) {
  // 转换 M 为齐次坐标
  M.col(2) = Eigen::VectorXd::Ones(M.rows());
  Eigen::MatrixXd Mc;
  Eigen::VectorXd Xpp, Ypp;
  double MSE = 0.0;
  std::vector<double> err, stderr;
  // 循环计算各视角的重投影误差
  for (size_t i = 0; i < calib_data_->Xp_abs.size(); ++i) {
    // 变换至相机坐标
    Mc = calib_data_->RRfin[i] * M.transpose();
    // 将 3D 点投影到 2D 像素坐标
    Eigen::MatrixXd m = World2Cam(Mc, calib_data_->ocam_model);
    Eigen::VectorXd xp = m.col(0);
    Eigen::VectorXd yp = m.col(1);
    // 计算投影误差
    Eigen::VectorXd stt = ((calib_data_->Xp_abs[i] - xp).array().square() +
                           (calib_data_->Yp_abs[i] - yp).array().square())
                              .sqrt();
    // 计算误差统计
    err.emplace_back(stt.mean());
    stderr.emplace_back(std::sqrt((stt.array() - stt.mean()).square().mean()));
    MSE += stt.array().square().sum();
  }
  std::cout << "Average error [pixels]\n\n"
            << std::accumulate(err.begin(), err.end(), 0.0) / err.size()
            << "\n";
  std::cout << "Sum of squared errors\n\n" << MSE << "\n";
}

Eigen::VectorXd Calibration::FindInvPoly(double radius) {
  Eigen::VectorXd inv_poly, err;
  double maxerr = std::numeric_limits<double>::infinity();
  int n = 1;
  // 迭代优化
  while (maxerr > 0.01) {
    n++;
    FindInvPolyHelper(radius, n, inv_poly, err);
    maxerr = err.maxCoeff();
  }
  return inv_poly;
}

void Calibration::FindInvPolyHelper(double radius, int n,
                                    Eigen::VectorXd& inv_poly,
                                    Eigen::VectorXd& err) {
  // 生成 theta 向量
  Eigen::VectorXd theta = Eigen::VectorXd::LinSpaced(
      int((1.20 - (-M_PI / 2)) / 0.01) + 1, -M_PI / 2, 1.20);
  Eigen::VectorXd m = theta.array().tan();
  // 计算 r（投影半径）
  Eigen::VectorXd r(theta.size());
  for (int i = 0; i < m.size(); ++i) {
    Eigen::VectorXd poly_coef = calib_data_->ocam_model.ss.reverse();
    poly_coef[poly_coef.size() - 2] -= m[i];

    std::vector<double> res;
    const auto& roots = ComputePolynomialRoots(poly_coef);
    for (const auto& root : roots) {
      if (root.imag() == 0 && root.real() > 0 && root.real() < radius) {
        res.emplace_back(root.real());
      }
      if (res.empty() || res.size() > 1) {
        r[i] = std::numeric_limits<double>::infinity();
      } else {
        r[i] = res[0];
      }
    }
  }
  // 过滤无效数据
  std::vector<double> valid_theta, valid_r;
  for (int i = 0; i < r.size(); ++i) {
    if (r(i) != std::numeric_limits<double>::infinity()) {
      valid_theta.emplace_back(theta(i));
      valid_r.emplace_back(r(i));
    }
  }
  Eigen::VectorXd new_theta =
      Eigen::Map<Eigen::VectorXd>(valid_theta.data(), valid_theta.size());
  Eigen::VectorXd new_r =
      Eigen::Map<Eigen::VectorXd>(valid_r.data(), valid_r.size());
  // 多项式拟合 polyfit
  // 创建 Vandermonde 矩阵
  Eigen::MatrixXd A(new_theta.size(), n + 1);
  for (int i = 0; i < new_theta.size(); i++) {
    for (int j = 0; j <= n; j++) {
      A(i, j) = pow(new_theta(i), j);
    }
  }
  // 使用 QR 分解求解线性最小二乘问题
  inv_poly = A.householderQr().solve(new_r);

  // 计算拟合误差
  err.resize(new_theta.size());
  for (int i = 0; i < new_theta.size(); ++i) {
    double theta_i = new_theta(i);
    double value = 0.0;
    for (int j = 0; j < inv_poly.size(); ++j) {
      value += inv_poly(j) * pow(theta_i, j);
    }
    err(i) = fabs(new_r(i) - value);
  }
}
