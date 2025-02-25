#include "utils.h"

#include <iostream>
/*
%%%% TEST OF dRdom:
om = randn(3,1);
dom = randn(3,1)/1000000;

[R1,dR1] = rodrigues(om);
R2 = rodrigues(om+dom);

R2a = R1 + reshape(dR1 * dom,3,3);

gain = norm(R2 - R1)/norm(R2 - R2a)
*/
void TestdRdom() {
  Eigen::Vector3d om = Eigen::Vector3d::Random();
  Eigen::Vector3d dom = Eigen::Vector3d::Random() / 1000000;

  auto [R1, dR1] = RodroguesFromRotationVector(om);
  auto [R2, _] = RodroguesFromRotationVector(om + dom);

  Eigen::VectorXd dR1_dom = dR1 * dom;
  Eigen::Matrix3d R2a = R1 + Eigen::Map<Eigen::Matrix3d>(dR1_dom.data());
  double gain = (R2 - R1).norm() / (R2 - R2a).norm();

  std::cout << "gain: " << gain << "\n";
}

/*
%%% TEST OF dOmdR:
om = randn(3,1);
R = rodrigues(om);
dom = randn(3,1)/10000;
dR = rodrigues(om+dom) - R;

[omc,domdR] = rodrigues(R);
[om2] = rodrigues(R+dR);

om_app = omc + domdR*dR(:);

gain = norm(om2 - omc)/norm(om2 - om_app)
*/
void TestdOmdR() {
  Eigen::Vector3d om = Eigen::Vector3d::Random();
  Eigen::Matrix3d R, dR;
  std::tie(R, std::ignore) = RodroguesFromRotationVector(om);
  Eigen::Vector3d dom = Eigen::Vector3d::Random() / 10000;
  std::tie(dR, std::ignore) = RodroguesFromRotationVector(om + dom);
  dR -= R;
  auto [omc, domdR] = RodroguesFromRotationMatrix(R);
  Eigen::Vector3d om2;
  std::tie(om2, std::ignore) = RodroguesFromRotationMatrix(R + dR);

  Eigen::Map<Eigen::VectorXd> dR_vec(dR.data(), dR.size());
  Eigen::Vector3d om_app = omc + domdR * dR_vec;

  double gain = (om2 - omc).norm() / (om2 - om_app).norm();

  std::cout << "gain: " << gain << std::endl;
}

int main() {
  TestdRdom();
  TestdOmdR();
  return 0;
}