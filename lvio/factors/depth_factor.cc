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

#include "factors/depth_factor.h"

PoseOnlyDepthFactor::PoseOnlyDepthFactor(Vector3d pts0, Vector3d pts2, Vector3d vel0, Vector3d vel2, double td0,
                                         double td2, double depth, double depth_std)
    : pts0_(std::move(pts0))
    , pts2_(std::move(pts2))
    , vel0_(std::move(vel0))
    , vel2_(std::move(vel2))
    , td0_(td0)
    , td2_(td2)
    , depth_(depth) {

    sqrt_info_depth_ = 1.0 / depth_std;
}

bool PoseOnlyDepthFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
    // 位姿
    Vector3d t_w_b0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond q_w_b0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d t_w_b2(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond q_w_b2(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // 外参
    Vector3d t_b_c(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond q_b_c(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    // 延时
    double td0 = parameters[3][0];
    double td2 = parameters[4][0];

    // 相机坐标系的位姿
    Matrix3d R_w_c0 = Rotation::quaternion2matrix(q_w_b0 * q_b_c);
    Vector3d t_w_c0 = q_w_b0 * t_b_c + t_w_b0;
    Matrix3d R_w_c2 = Rotation::quaternion2matrix(q_w_b2 * q_b_c);
    Vector3d t_w_c2 = q_w_b2 * t_b_c + t_w_b2;

    // 观测延时
    Vector3d pts_0_td = pts0_ - (td0 - td0_) * vel0_;
    Vector3d pts_2_td = pts2_ - (td2 - td2_) * vel2_;

    // 计算深度
    Matrix3d R_c2_c0   = R_w_c2.transpose() * R_w_c0;
    Vector3d t_c2_c0   = R_w_c2.transpose() * (t_w_c0 - t_w_c2);
    Vector3d pts_2_0   = R_c2_c0 * pts_0_td;
    Matrix3d skew_p2   = Rotation::skewSymmetric(pts_2_td);
    Vector3d norm_vec  = skew_p2 * t_c2_c0;
    Vector3d theta_vec = skew_p2 * pts_2_0;

    double norm          = (skew_p2 * t_c2_c0).norm();
    double theta         = (theta_vec).norm();
    double norm_inverse  = 1.0 / norm;
    double theta_inverse = 1.0 / theta;

    // 残差
    residuals[0] = sqrt_info_depth_ * (norm * theta_inverse - depth_);

    if (jacobians) {
        Matrix3d R_b_c   = q_b_c.toRotationMatrix();
        Matrix3d R_c_b   = q_b_c.toRotationMatrix().transpose();
        Matrix3d R_c2_w  = R_w_c2.transpose();
        Matrix3d R_c2_b0 = R_c2_w * q_w_b0.toRotationMatrix();
        Matrix3d R_b2_w  = q_w_b2.toRotationMatrix().transpose();

        Matrix3d skew_t_b_c        = Rotation::skewSymmetric(t_b_c);
        Matrix3d skew_t_c2_c0      = Rotation::skewSymmetric(t_c2_c0);
        Matrix3d skew_theta_prefix = Rotation::skewSymmetric(pts_2_0);
        Matrix3d norm_prefix       = (skew_p2 * R_c2_w).transpose();

        Eigen::RowVector3d jaco_norm_t = norm_inverse * (norm_prefix * norm_vec).transpose();
        double norm_theta_inverse2     = norm / (theta * theta);

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_depth(jacobians[0]);
            jacobian_depth.setZero();

            Matrix3d norm_r0_prefix  = -(skew_p2 * R_c2_b0 * skew_t_b_c).transpose();
            Matrix3d theta_r0_prefix = -(skew_p2 * R_c2_b0 * Rotation::skewSymmetric(R_b_c * pts_0_td)).transpose();

            Eigen::RowVector3d jaco_norm_r  = norm_inverse * (norm_r0_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r = theta_inverse * (theta_r0_prefix * theta_vec).transpose();

            jacobian_depth.block<1, 3>(0, 0) = jaco_norm_t * theta_inverse;
            jacobian_depth.block<1, 3>(0, 3) = jaco_norm_r * theta_inverse - jaco_theta_r * norm_theta_inverse2;
            jacobian_depth *= sqrt_info_depth_;
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_depth(jacobians[1]);
            jacobian_depth.setZero();

            Eigen::Matrix<double, 3, 6> jaco;
            Matrix3d norm_r2_prefix =
                (skew_p2 * R_c_b * skew_t_b_c + skew_p2 * R_c_b * Rotation::skewSymmetric(R_b2_w * (t_w_c0 - t_w_c2)))
                    .transpose();
            Matrix3d theta_r2_prefix =
                (skew_p2 * R_c_b * Rotation::skewSymmetric(R_b2_w * R_w_c0 * pts_0_td)).transpose();

            Eigen::RowVector3d jaco_norm_r  = norm_inverse * (norm_r2_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r = theta_inverse * (theta_r2_prefix * theta_vec).transpose();

            jacobian_depth.block<1, 3>(0, 0) = -jaco_norm_t * theta_inverse;
            jacobian_depth.block<1, 3>(0, 3) = jaco_norm_r * theta_inverse - jaco_theta_r * norm_theta_inverse2;
            jacobian_depth *= sqrt_info_depth_;
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jacobian_depth(jacobians[2]);
            jacobian_depth.setZero();

            Matrix3d norm_t_prefix = (skew_p2 * (R_c2_b0 - R_c_b)).transpose();
            Matrix3d norm_r_prefix = (skew_p2 * skew_t_c2_c0).transpose();

            Matrix3d theta_r0_prefix = -(skew_p2 * R_c2_c0 * Rotation::skewSymmetric(pts_0_td)).transpose();
            Matrix3d theta_r2_prefix = (skew_p2 * skew_theta_prefix).transpose();

            Eigen::RowVector3d jaco_norm_t = norm_inverse * (norm_t_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_norm_r = norm_inverse * (norm_r_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r =
                theta_inverse * ((theta_r0_prefix + theta_r2_prefix) * theta_vec).transpose();

            jacobian_depth.block<1, 3>(0, 0) = jaco_norm_t * theta_inverse;
            jacobian_depth.block<1, 3>(0, 3) = jaco_norm_r * theta_inverse - jaco_theta_r * norm_theta_inverse2;
            jacobian_depth *= sqrt_info_depth_;
        }

        if (jacobians[3]) {
            double jaco_theta_p0 = -theta_inverse * ((skew_p2 * R_c2_c0) * vel0_).transpose() * theta_vec;
            jacobians[3][0]      = sqrt_info_depth_ * jaco_theta_p0 * norm_theta_inverse2;
        }

        if (jacobians[4]) {
            double jaco_norm_p2 = -norm_inverse * (skew_t_c2_c0 * vel2_).transpose() * skew_t_c2_c0 * pts_2_td;
            double jaco_theta_p2 =
                -theta_inverse * (skew_theta_prefix * vel2_).transpose() * skew_theta_prefix * pts_2_td;

            jacobians[4][0] = sqrt_info_depth_ * (jaco_norm_p2 * theta_inverse - jaco_theta_p2 * norm_theta_inverse2);
        }
    }

    return true;
}
