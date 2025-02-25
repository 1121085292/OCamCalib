#include "utils.h"

std::pair<Eigen::Matrix3d, Eigen::MatrixXd> RodroguesFromRotationVector(
    const Eigen::Vector3d& in) {
  // 计算旋转角度
  double theta = in.norm();
  Eigen::Matrix3d R;
  Eigen::MatrixXd dRdin(9, 3);
  // 处理小角度情况
  if (theta < std::numeric_limits<double>::epsilon()) {
    R = Eigen::Matrix3d::Identity();
    // Optional: Compute the derivative matrix dRdin
    dRdin << 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        -1, 0, 0, 0, 0, 0;
  } else {
    // 计算 Rodrigues 公式
    Eigen::Vector3d omega = in / theta;
    double alpha = cos(theta);
    double beta = sin(theta);
    double gamma = 1 - alpha;
    // 计算旋转矩阵
    Eigen::Matrix3d omegav;
    omegav << 0, -omega(2), omega(1), omega(2), 0, -omega(0), -omega(1),
        omega(0), 0;
    Eigen::Matrix3d A = omega * omega.transpose();
    Eigen::Map<Eigen::VectorXd> A_col(A.data(), A.size());

    R = Eigen::Matrix3d::Identity() * alpha + omegav * beta + A * gamma;
    // 计算 dRdin（旋转矩阵对旋转向量的导数）
    Eigen::MatrixXd dRdm1 = Eigen::MatrixXd::Zero(9, 21);
    dRdm1(0, 0) = 1;
    dRdm1(4, 0) = 1;
    dRdm1(8, 0) = 1;
    Eigen::Map<Eigen::VectorXd> omegav_col(omegav.data(), omegav.size());
    dRdm1.col(1) = omegav_col;
    dRdm1.col(2) = A_col;
    dRdm1.block(0, 3, 9, 9) = beta * Eigen::MatrixXd::Identity(9, 9);
    dRdm1.block(0, 12, 9, 9) = gamma * Eigen::MatrixXd::Identity(9, 9);

    Eigen::MatrixXd dm1dm2 = Eigen::MatrixXd::Zero(21, 4);
    dm1dm2(0, 3) = -sin(theta);
    dm1dm2(1, 3) = cos(theta);
    dm1dm2(2, 3) = sin(theta);
    dm1dm2.block(3, 0, 9, 3) << 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0,
        1, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0;
    dm1dm2.block(12, 0, 9, 3) << 2 * omega(0), omega(1), omega(2), omega(1), 0,
        0, omega(2), 0, 0, 0, omega(0), 0, omega(0), 2 * omega(1), omega(2), 0,
        omega(2), 0, 0, 0, omega(0), 0, 0, omega(1), omega(0), omega(1),
        2 * omega(2);

    Eigen::MatrixXd dm3din(4, 3);
    dm3din << Eigen::Matrix3d::Identity(), in.transpose() / theta;

    Eigen::MatrixXd dm2dm3(4, 4);
    dm2dm3.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() / theta;
    dm2dm3.block(0, 3, 3, 1) = -in / (theta * theta);
    dm2dm3.block(3, 0, 1, 3) = Eigen::RowVector3d::Zero();
    dm2dm3(3, 3) = 1;

    dRdin = dRdm1 * dm1dm2 * dm2dm3 * dm3din;
  }
  return {R, dRdin};
}

std::pair<Eigen::Vector3d, Eigen::MatrixXd> RodroguesFromRotationMatrix(
    const Eigen::Matrix3d& in) {
  // 使用 SVD 分解并确保旋转矩阵是正交的
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      in, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  // 如果行列式为负，调整 V 的最后一列符号，确保 R 是有效的旋转矩阵
  if (R.determinant() < 0) {
    Eigen::Matrix3d V = svd.matrixV();
    V.col(2) *= -1;
    R = svd.matrixU() * V.transpose();
  }

  // 计算旋转角度 theta
  double tr = std::clamp((R.trace() - 1.0) / 2.0, -1.0,
                         1.0);  // 限制 trace 值，防止出现 acos 错误
  double theta = std::acos(tr);
  // double tr = (R.trace() - 1.0) / 2.0;
  // double theta = std::acos(tr);

  // 定义一些中间变量，用于计算雅可比矩阵
  Eigen::VectorXd dtrdR(9);
  dtrdR << 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5;
  Eigen::Vector3d out;
  Eigen::MatrixXd dout(3, 9);

  if (std::sin(theta) >= 1e-5) {  // 如果 theta 不接近零
    double dthetadtr = -1 / std::sqrt(1 - tr * tr);
    Eigen::VectorXd dthetadR = dthetadtr * dtrdR;
    double vth = 1 / (2 * std::sin(theta));
    double dvthdtheta = -vth * std::cos(theta) / std::sin(theta);
    Eigen::Vector2d dvar1dtheta = Eigen::Vector2d(dvthdtheta, 1);
    Eigen::MatrixXd dvar1dR = dvar1dtheta * dthetadR.transpose();

    Eigen::Vector3d om1(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                        R(1, 0) - R(0, 1));
    Eigen::MatrixXd dom1dR(3, 9);
    dom1dR << 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
        -1, 0, 0, 0, 0, 0;
    Eigen::MatrixXd dvardR(5, 9);
    dvardR << dom1dR, dvar1dR;

    Eigen::Vector3d om = vth * om1;
    Eigen::MatrixXd domdvar(3, 5);
    domdvar.block(0, 0, 3, 3) = Eigen::Matrix3d::Identity() * vth;
    domdvar.block(0, 3, 3, 1) = om1;
    domdvar.block(0, 4, 3, 1) = Eigen::Vector3d::Zero();

    Eigen::VectorXd dthetadvar(5);
    dthetadvar << 0, 0, 0, 0, 1;
    Eigen::MatrixXd dvar2dvar(4, 5);
    dvar2dvar << domdvar, dthetadvar.transpose();

    out = om * theta;

    Eigen::MatrixXd domegadvar2(3, 4);
    domegadvar2 << theta * Eigen::Matrix3d::Identity(), om;
    dout = domegadvar2 * dvar2dvar * dvardR;
  } else {  // 当 theta 接近 0 或 180 度时的特殊处理
    if (tr > 0) {
      out = Eigen::Vector3d::Zero();
      dout << 0, 0, 0, 0, 0, 0.5, 0, -0.5, 0, 0, 0, -0.5, 0, 0, 0, 0.5, 0, 0, 0,
          0.5, 0, -0.5, 0, 0, 0, 0, 0;
    } else {
      Eigen::Vector3d sign_vector;
      sign_vector(0) = 1.0;
      sign_vector(1) = (R(0, 1) >= 0) ? 1.0 : -1.0;
      sign_vector(2) = (R(0, 2) >= 0) ? 1.0 : -1.0;
      out = theta * (((R.diagonal() +
                       Eigen::VectorXd::Constant(R.diagonal().size(), 1.0)) /
                      2.0)
                         .cwiseSqrt()
                         .array() *
                     sign_vector.array());
      dout.setConstant(std::numeric_limits<double>::quiet_NaN());
    }
  }
  return {out, dout};
}

void Omni3dToPixel(const Eigen::MatrixXd& xx0, const Eigen::VectorXd& ss,
                   Eigen::VectorXd& x, Eigen::VectorXd& y) {
  Eigen::MatrixXd xx = xx0;
  for (int i = 0; i < static_cast<int>(xx.cols()); ++i) {
    if (xx(0, i) == 0 && xx(1, i) == 0) {
      xx(0, i) = std::numeric_limits<double>::epsilon();
      xx(1, i) = std::numeric_limits<double>::epsilon();
    }
  }
  // 计算 m（投影参数）
  Eigen::VectorXd m =
      xx.row(2).array() /
      (xx.row(0).array().square() + xx.row(1).array().square()).sqrt();

  // 计算 rho（径向距离）
  Eigen::VectorXd rho(m.size());
  Eigen::VectorXd poly_coef = ss.reverse();
  Eigen::VectorXd poly_coef_tmp = poly_coef;
  for (int i = 0; i < static_cast<int>(m.size()); ++i) {
    poly_coef_tmp[poly_coef_tmp.size() - 2] =
        poly_coef[poly_coef.size() - 2] - m[i];
    std::vector<double> res;
    const auto& roots = ComputePolynomialRoots(poly_coef_tmp);
    for (const auto& root : roots) {
      if (root.imag() == 0 && root.real() > 0) {
        res.emplace_back(root.real());
      }
    }
    if (res.empty()) {
      rho(i) = std::numeric_limits<double>::quiet_NaN();
    } else {
      rho(i) = *std::min_element(res.begin(), res.end());
    }
  }
  // 计算 x, y（像素坐标）
  Eigen::VectorXd norm =
      (xx.row(0).array().square() + xx.row(1).array().square()).sqrt();
  x = xx.row(0).transpose().array() / norm.array() * rho.array();
  y = xx.row(1).transpose().array() / norm.array() * rho.array();
}

std::vector<std::complex<double>> ComputePolynomialRoots(
    const Eigen::VectorXd& coeffs) {
  std::vector<std::complex<double>> roots;

  Eigen::VectorXd reversed(coeffs.size());
  // 反转元素
  for (int i = 0; i < static_cast<int>(coeffs.size()); ++i) {
    reversed(i) = coeffs(coeffs.size() - 1 - i);
  }

  int n = reversed.size() - 1;
  Eigen::MatrixXd companion_matrix(n, n);
  companion_matrix.setZero();

  // 构造伴随矩阵
  for (int i = 0; i < n - 1; ++i) {
    companion_matrix(i + 1, i) = 1;
  }
  for (int i = 0; i < n; ++i) {
    companion_matrix(i, n - 1) = -reversed(i) / reversed(n);
  }

  // 计算特征值
  Eigen::EigenSolver<Eigen::MatrixXd> solver(companion_matrix);
  if (solver.info() != Eigen::Success) {
    std::cerr << "Failed to compute eigenvalues of companion matrix."
              << std::endl;
  } else {
    for (int i = 0; i < static_cast<int>(solver.eigenvalues().size()); ++i) {
      roots.push_back(solver.eigenvalues()[i]);
    }
  }
  return roots;
}

Eigen::MatrixXd World2Cam(const Eigen::MatrixXd& M, const OcamModel& model) {
  Eigen::VectorXd x, y;
  Omni3dToPixel(M, model.ss, x, y);
  Eigen::MatrixXd m(x.rows(), 2);
  m.col(0) = x.array() * model.c + y.array() * model.d + model.xc;
  m.col(1) = x.array() * model.e + y.array() + model.yc;
  return m;
}

bool GetOcamModel(const std::string& filename, OcamModel& model) {
  std::ifstream input_file(filename);
  if (!input_file.is_open()) {
    std::cout << "Error opening file: " << filename << std::endl;
    return false;
  }
  // 文件类型
  size_t pos = filename.find_last_of(".");
  if (pos == std::string::npos) return false;

  std::string file_type = filename.substr(pos + 1);
  if (file_type == "txt") {
    std::string line;
    // read polynomial coefficients
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::stringstream ss(line);
    int poly_size = 0;
    ss >> poly_size;
    model.ss.resize(poly_size);
    for (int i = 0; i < poly_size; ++i) {
      ss >> model.ss(i);
    }

    // read inverse polynomial coefficients
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::stringstream inv_poly(line);
    int inv_poly_size = 0;
    inv_poly >> inv_poly_size;
    model.inv_poly.resize(inv_poly_size);
    for (int i = 0; i < inv_poly_size; ++i) {
      inv_poly >> model.inv_poly(i);
    }

    // read camera center
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::stringstream center(line);
    center >> model.xc >> model.yc;

    // read affine parameters
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::stringstream affine(line);
    affine >> model.c >> model.d >> model.e;

    // read image size
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::getline(input_file, line);
    std::stringstream size(line);
    size >> model.height >> model.width;

    input_file.close();
    return true;
  } else if (file_type == "yaml") {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    // read image size
    fs["height"] >> model.height;
    fs["width"] >> model.width;
    // read polynomial coefficients
    cv::Mat poly_cv = fs["poly_coeffs"].mat();
    model.ss = Eigen::Map<Eigen::VectorXd>(poly_cv.ptr<double>(),
                                           poly_cv.cols * poly_cv.rows);
    // read inverse polynomial coefficients
    cv::Mat inv_poly_cv = fs["inv_poly"].mat();
    model.inv_poly = Eigen::Map<Eigen::VectorXd>(
        inv_poly_cv.ptr<double>(), inv_poly_cv.cols * inv_poly_cv.rows);
    // read camera center
    cv::Mat center_cv = fs["center"].mat();
    model.xc = center_cv.at<double>(0, 0);
    model.yc = center_cv.at<double>(1, 0);
    // read affine parameters
    cv::Mat affine_cv = fs["affine"].mat();
    model.c = affine_cv.at<double>(0, 0);
    model.d = affine_cv.at<double>(0, 1);
    model.e = affine_cv.at<double>(1, 0);

    fs.release();
    return true;
  }
  return false;
}

void CreatePerspeciveUndistortionLUT(const OcamModel& ocam_model, float sf,
                                     cv::Mat& mapx, cv::Mat& mapy) {
  int width = mapx.cols;
  int height = mapx.rows;
  float Nxc = height / 2.0;
  float Nyc = width / 2.0;
  float Nz = -width / sf;
  double M[3];
  double m[2];

  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++) {
      M[0] = (i - Nxc);
      M[1] = (j - Nyc);
      M[2] = Nz;
      World2Cam(ocam_model, m, M);
      mapx.at<float>(i, j) = static_cast<float>(m[1]);
      mapy.at<float>(i, j) = static_cast<float>(m[0]);
    }
}

void World2Cam(const OcamModel& ocam_model, double point2D[2],
               double point3D[3]) {
  Eigen::VectorXd inv_poly = ocam_model.inv_poly;

  double norm = sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1]);
  double theta = atan(point3D[2] / norm);
  double t, t_i;
  double rho, x, y;
  double invnorm;

  if (norm != 0) {
    invnorm = 1 / norm;
    t = theta;
    rho = inv_poly(0);
    t_i = 1;

    for (int i = 1; i < inv_poly.size(); i++) {
      t_i *= t;
      rho += t_i * inv_poly(i);
    }

    x = point3D[0] * invnorm * rho;
    y = point3D[1] * invnorm * rho;

    point2D[0] = x * ocam_model.c + y * ocam_model.d + ocam_model.xc;
    point2D[1] = x * ocam_model.e + y + ocam_model.yc;
  } else {
    point2D[0] = ocam_model.xc;
    point2D[1] = ocam_model.yc;
  }
}

void PrintOcamModel(const OcamModel& ocam_model) {
  std::cout << "OCamCalib model parameters" << std::endl
            << "pol: " << std::endl;

  std::cout << "\t" << ocam_model.ss.transpose() << "\n";
  std::cout << "int_poly: \n\t" << ocam_model.inv_poly.transpose() << "\n";

  std::cout << "xc:\t" << ocam_model.xc << std::endl
            << "yc:\t" << ocam_model.yc << std::endl
            << "c:\t" << ocam_model.c << std::endl
            << "d:\t" << ocam_model.d << std::endl
            << "e:\t" << ocam_model.e << std::endl
            << "width:\t" << ocam_model.width << std::endl
            << "height:\t" << ocam_model.height << std::endl;
}
