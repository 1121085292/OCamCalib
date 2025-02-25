#pragma once

#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "calib_data.h"

std::pair<Eigen::Vector3d, Eigen::MatrixXd> RodroguesFromRotationMatrix(
    const Eigen::Matrix3d& in);

std::pair<Eigen::Matrix3d, Eigen::MatrixXd> RodroguesFromRotationVector(
    const Eigen::Vector3d& in);

// convert 3D coordinates vector into 2D pixel coordinates
void Omni3dToPixel(const Eigen::MatrixXd& xx0, const Eigen::VectorXd& ss,
                   Eigen::VectorXd& x, Eigen::VectorXd& y);

std::vector<std::complex<double>> ComputePolynomialRoots(
    const Eigen::VectorXd& coeffs);

//  M is a N×3 matrix containing the coordinates of the 3D points: M=[X，Y，Z]
//  "model" contains the model of the calibrated camera.
//  return is a Nx2 matrix containing the returned rows and columns of
//  the points after being reproject onto the image.
Eigen::MatrixXd World2Cam(const Eigen::MatrixXd& M, const OcamModel& model);

/*------------------------------------------------------------------------------
 This function reads the parameters of the omnidirectional camera model from
 a given TXT or Yaml file
------------------------------------------------------------------------------*/
bool GetOcamModel(const std::string& filename, OcamModel& model);

/*------------------------------------------------------------------------------
 Create Look Up Table for undistorting the image into a perspective image
 It assumes the the final image plane is perpendicular to the camera axis
------------------------------------------------------------------------------*/
void CreatePerspeciveUndistortionLUT(const OcamModel& ocam_model, float sf,
                                     cv::Mat& mapx, cv::Mat& mapy);

/*------------------------------------------------------------------------------
WORLD2CAM projects a 3D point on to the image
WORLD2CAM(POINT2D, POINT3D, OCAM_MODEL)
projects a 3D point (point3D) on to the image and returns the pixel coordinates
(point2D).

POINT3D = [X;Y;Z] are the coordinates of the 3D point.
OCAM_MODEL is the model of the calibrated camera.
POINT2D = [rows;cols] are the pixel coordinates of the reprojected point

Copyright (C) 2009 DAVIDE SCARAMUZZA
Author: Davide Scaramuzza - email: davide.scaramuzza@ieee.org

NOTE: the coordinates of "point2D" and "center" are already according to the C
convention, that is, start from 0 instead than from 1.
------------------------------------------------------------------------------*/
void World2Cam(const OcamModel& ocam_model, double point2D[2],
               double point3D[3]);

void PrintOcamModel(const OcamModel& ocam_model);
