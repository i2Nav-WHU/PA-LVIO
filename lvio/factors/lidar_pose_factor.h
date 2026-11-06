/*
 * Copyright (C) 2026 i2Nav Group, Wuhan University
 *
 *     Author : Hailiang Tang
 *    Contact : thl@whu.edu.cn
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "common/rotation.h"
#include "common/types.h"

#include <ceres/ceres.h>

// class LiDARPoseFactor : public ceres::SizedCostFunction<3, 7> {

// public:
//     LiDARPoseFactor(double *pose, double *std) {
//         t_w_b_ = Vector3d(pose[0], pose[1], pose[2]);
//         q_w_b_ = Quaterniond(pose[6], pose[3], pose[4], pose[5]);

//         sqrt_info_.setZero();
//         for (size_t k = 0; k < 3; k++) {
//             sqrt_info_(k, k) = 1.0 / std[k + 3];
//         }
//     }

//     bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
//         Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);

//         Vector3d t_w_b(parameters[0][0], parameters[0][1], parameters[0][2]);
//         Quaterniond q_w_b(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

//         residual.block<3, 1>(0, 0) = 2 * (q_w_b * q_w_b_.inverse()).vec();

//         residual = sqrt_info_ * residual;

//         if (jacobians) {
//             if (jacobians[0]) {
//                 Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> jaco(jacobians[0]);
//                 jaco.setZero();

//                 jaco.block<3, 3>(0, 3) = (Rotation::quaternionright(q_w_b_.inverse()) *
//                 Rotation::quaternionleft(q_w_b))
//                                              .bottomRightCorner<3, 3>();

//                 jaco = sqrt_info_ * jaco;
//             }
//         }

//         return true;
//     }

// private:
//     Vector3d t_w_b_;
//     Quaterniond q_w_b_;

//     Eigen::Matrix<double, 3, 3, Eigen::RowMajor> sqrt_info_;
// };

// class LiDARPoseFactor : public ceres::SizedCostFunction<6, 7> {

// public:
//     LiDARPoseFactor(double *pose, double *std) {
//         t_w_b_ = Vector3d(pose[0], pose[1], pose[2]);
//         q_w_b_ = Quaterniond(pose[6], pose[3], pose[4], pose[5]);

//         sqrt_info_.setZero();
//         for (size_t k = 0; k < 6; k++) {
//             sqrt_info_(k, k) = 1.0 / std[k];
//         }
//     }

//     bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
//         Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);

//         Vector3d t_w_b(parameters[0][0], parameters[0][1], parameters[0][2]);
//         Quaterniond q_w_b(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

//         residual.block<3, 1>(0, 0) = t_w_b - t_w_b_;
//         residual.block<3, 1>(3, 0) = 2 * (q_w_b * q_w_b_.inverse()).vec();

//         residual = sqrt_info_ * residual;

//         if (jacobians) {
//             if (jacobians[0]) {
//                 Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jaco(jacobians[0]);
//                 jaco.setZero();

//                 jaco.block<3, 3>(0, 0) = Matrix3d::Identity();
//                 Matrix3d jaco_rot      = (Rotation::quaternionright(q_w_b_.inverse()) *
//                 Rotation::quaternionleft(q_w_b))
//                                         .bottomRightCorner<3, 3>();
//                 jaco.block<3, 3>(3, 3) = jaco_rot;

//                 jaco = sqrt_info_ * jaco;
//             }
//         }

//         return true;
//     }

// private:
//     Vector3d t_w_b_;
//     Quaterniond q_w_b_;

//     Eigen::Matrix<double, 6, 6, Eigen::RowMajor> sqrt_info_;
// };

class LiDARPoseFactor : public ceres::SizedCostFunction<6, 7, 7> {

public:
    LiDARPoseFactor(double *pose, double *std) {
        t_w_l_ = Vector3d(pose[0], pose[1], pose[2]);
        q_w_l_ = Quaterniond(pose[6], pose[3], pose[4], pose[5]);

        sqrt_info_.setZero();
        for (size_t k = 0; k < 6; k++) {
            sqrt_info_(k, k) = 1.0 / std[k];
        }
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
        Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);

        Vector3d t_w_b(parameters[0][0], parameters[0][1], parameters[0][2]);
        Quaterniond q_w_b(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
        Vector3d t_b_l(parameters[1][0], parameters[1][1], parameters[1][2]);
        Quaterniond q_b_l(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Matrix3d R_w_b = q_w_b.toRotationMatrix();
        Vector3d t_w_l = t_w_b + R_w_b * t_b_l;

        residual.block<3, 1>(0, 0) = t_w_l - t_w_l_;
        residual.block<3, 1>(3, 0) = 2 * (q_w_b * q_b_l * q_w_l_.inverse()).vec();

        residual = sqrt_info_ * residual;

        if (jacobians) {
            if (jacobians[0]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jaco(jacobians[0]);
                jaco.setZero();

                jaco.block<3, 3>(0, 0) = Matrix3d::Identity();
                jaco.block<3, 3>(0, 3) = -R_w_b * Rotation::skewSymmetric(t_b_l);
                Matrix3d jaco_rot =
                    (Rotation::quaternionright(q_b_l * q_w_l_.inverse()) * Rotation::quaternionleft(q_w_b))
                        .bottomRightCorner<3, 3>();
                jaco.block<3, 3>(3, 3) = jaco_rot;

                jaco = sqrt_info_ * jaco;
            }

            if (jacobians[1]) {
                Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> jaco(jacobians[1]);
                jaco.setZero();

                jaco.block<3, 3>(0, 0) = R_w_b;
                Matrix3d jaco_rot =
                    (Rotation::quaternionright(q_w_l_.inverse()) * Rotation::quaternionleft(q_w_b * q_b_l))
                        .bottomRightCorner<3, 3>();
                jaco.block<3, 3>(3, 3) = jaco_rot;

                jaco = sqrt_info_ * jaco;
            }
        }

        return true;
    }

private:
    Vector3d t_w_l_;
    Quaterniond q_w_l_;

    Eigen::Matrix<double, 6, 6, Eigen::RowMajor> sqrt_info_;
};

// class LiDARPoseFactor : public ceres::SizedCostFunction<4, 7, 7> {

// public:
//     LiDARPoseFactor(double *pose, double *std) {
//         t_w_l_ = Vector3d(pose[0], pose[1], pose[2]);
//         q_w_l_ = Quaterniond(pose[6], pose[3], pose[4], pose[5]);

//         sqrt_info_.setZero();
//         for (size_t k = 0; k < 3; k++) {
//             sqrt_info_(k, k) = 1.0 / std[k];
//         }
//         sqrt_info_(3, 3) = 1.0 / std[5];
//     }

//     bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {
//         Eigen::Map<Eigen::Matrix<double, 4, 1>> residual(residuals);

//         Vector3d t_w_b(parameters[0][0], parameters[0][1], parameters[0][2]);
//         Quaterniond q_w_b(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
//         Vector3d t_b_l(parameters[1][0], parameters[1][1], parameters[1][2]);
//         Quaterniond q_b_l(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

//         Matrix3d R_w_b = q_w_b.toRotationMatrix();
//         Vector3d t_w_l = t_w_b + R_w_b * t_b_l;

//         residual.block<3, 1>(0, 0) = t_w_l - t_w_l_;
//         residual[3]                = 2 * (q_w_b * q_b_l * q_w_l_.inverse()).vec()[2];

//         residual = sqrt_info_ * residual;

//         if (jacobians) {
//             if (jacobians[0]) {
//                 Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> jaco(jacobians[0]);
//                 jaco.setZero();

//                 jaco.block<3, 3>(0, 0) = Matrix3d::Identity();
//                 jaco.block<3, 3>(0, 3) = -R_w_b * Rotation::skewSymmetric(t_b_l);
//                 Matrix3d jaco_rot =
//                     (Rotation::quaternionright(q_b_l * q_w_l_.inverse()) * Rotation::quaternionleft(q_w_b))
//                         .bottomRightCorner<3, 3>();
//                 jaco.block<1, 3>(3, 3) = jaco_rot.block(3, 0, 1, 3);

//                 jaco = sqrt_info_ * jaco;
//             }

//             if (jacobians[1]) {
//                 Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> jaco(jacobians[1]);
//                 jaco.setZero();

//                 jaco.block<3, 3>(0, 0) = R_w_b;
//                 Matrix3d jaco_rot =
//                     (Rotation::quaternionright(q_w_l_.inverse()) * Rotation::quaternionleft(q_w_b * q_b_l))
//                         .bottomRightCorner<3, 3>();
//                 jaco.block<1, 3>(3, 3) = jaco_rot.block(3, 0, 1, 3);

//                 jaco = sqrt_info_ * jaco;
//             }
//         }

//         return true;
//     }

// private:
//     Vector3d t_w_l_;
//     Quaterniond q_w_l_;

//     Eigen::Matrix<double, 4, 4, Eigen::RowMajor> sqrt_info_;
// };