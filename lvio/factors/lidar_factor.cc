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

#include "factors/lidar_factor.h"

#include "common/rotation.h"
#include "lidar/pointcloud.h"

LidarBAFactor::LidarBAFactor(vector<Vector3d> points_lidar, vector<Vector3d> vels, vector<Vector3d> omegas,
                             vector<double> tds, double std)
    : points_lidar_(points_lidar)
    , vels_(vels)
    , omegas_(omegas)
    , tds_(tds) {

    points_size_   = points_lidar_.size();
    sqrt_info_     = 1.0 / (std * std * sqrt(2.0));
    num_n_         = static_cast<double>(points_size_);
    num_n_inverse_ = 1.0 / num_n_;

    // 参数块维度
    vector<int> parameter_block_sizes;
    for (size_t k = 0; k < points_size_; k++) {
        parameter_block_sizes.push_back(7);
    }
    // 外参和时间偏移
    parameter_block_sizes.push_back(7);
    parameter_block_sizes.push_back(1);
    *mutable_parameter_block_sizes() = parameter_block_sizes;

    // 残差维度
    set_num_residuals(1);
}

bool LidarBAFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
    // Pose_world_imu
    vector<Vector3d> t_w_b;
    vector<Quaterniond> q_w_b;
    for (size_t k = 0; k < points_size_; k++) {
        t_w_b.emplace_back(parameters[k][0], parameters[k][1], parameters[k][2]);
        q_w_b.emplace_back(parameters[k][6], parameters[k][3], parameters[k][4], parameters[k][5]);
    }

    // Pose_imu_lidar
    Vector3d t_b_l(parameters[points_size_][0], parameters[points_size_][1], parameters[points_size_][2]);
    Quaterniond q_b_l(parameters[points_size_][6], parameters[points_size_][3], parameters[points_size_][4],
                      parameters[points_size_][5]);
    Matrix3d R_b_l = q_b_l.toRotationMatrix();

    // td_imu_lidar
    double td = parameters[points_size_ + 1][0];

    // 位姿变换
    vector<Matrix3d> R_w_b(points_size_), R_b_bb(points_size_), R_w_bb(points_size_);
    vector<Vector3d> t_w_bb(points_size_);
    vector<Vector3d> points_world(points_size_), points_body(points_size_);
    for (size_t k = 0; k < points_size_; k++) {
        double dt = td - tds_[k];
        R_w_b[k]  = q_w_b[k].toRotationMatrix();
        R_b_bb[k] = Matrix3d::Identity() + Rotation::skewSymmetric(omegas_[k] * dt);
        R_w_bb[k] = R_w_b[k] * R_b_bb[k];
        t_w_bb[k] = t_w_b[k] + vels_[k] * dt;

        points_body[k]  = R_b_l * points_lidar_[k] + t_b_l;
        points_world[k] = R_w_bb[k] * points_body[k] + t_w_bb[k];
    }

    // 平面估计
    Vector3d normal_vector, center;
    double lambda;
    lidar::PointCloudCommon::planeFitting(points_world, normal_vector, center, lambda);

    // 残差计算
    residuals[0] = sqrt_info_ * lambda;

    if (jacobians) {
        // 对pw的雅克比
        vector<Eigen::RowVector3d> jaco_pw(points_size_);
        for (size_t k = 0; k < points_size_; k++) {
            double dis = normal_vector.transpose() * (points_world[k] - center);
            jaco_pw[k] = num_n_inverse_ * 2.0 * dis * normal_vector.transpose();
        }

        // Jacobians pose
        for (size_t k = 0; k < points_size_; k++) {
            if (jacobians[k]) {
                Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jaco(jacobians[k]);
                jaco.setZero();

                Matrix3d skew_body = Rotation::skewSymmetric(R_b_bb[k] * points_body[k]);

                jaco.block<1, 3>(0, 0) = jaco_pw[k];
                jaco.block<1, 3>(0, 3) = -jaco_pw[k] * R_w_b[k] * skew_body;
                jaco.block<1, 6>(0, 0) *= sqrt_info_;
            }
        }

        // Jacobian extrinsic
        if (jacobians[points_size_]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jaco(jacobians[points_size_]);
            jaco.setZero();

            for (size_t k = 0; k < points_size_; k++) {
                Eigen::RowVector3d jaco_pb = jaco_pw[k] * R_w_bb[k];
                jaco.block<1, 3>(0, 0) += jaco_pb;
                jaco.block<1, 3>(0, 3) -= jaco_pb * R_b_l * Rotation::skewSymmetric(points_lidar_[k]);
            }
            jaco.block<1, 6>(0, 0) *= sqrt_info_;
        }

        // Jacobian td
        if (jacobians[points_size_ + 1]) {
            double jaco_vel   = 0;
            double jaco_omega = 0;
            for (size_t k = 0; k < points_size_; k++) {
                jaco_vel += jaco_pw[k] * vels_[k];
                jaco_omega += jaco_pw[k] * R_w_b[k] * Rotation::skewSymmetric(omegas_[k]) * points_body[k];
            }
            jacobians[points_size_ + 1][0] = sqrt_info_ * (jaco_vel + jaco_omega);
        }
    }

    return true;
}

LidarFactor::LidarFactor(const PointType &point_lidar, Vector3d normal_vector, Vector3d center, double std,
                         Vector3d vel0, Vector3d omega0, double td0, Vector3d vel1, Vector3d omega1, double td1)
    : normal_vector_(std::move(normal_vector))
    , center_(std::move(center))
    , vel0_(std::move(vel0))
    , vel1_(std::move(vel1))
    , omega0_(std::move(omega0))
    , omega1_(std::move(omega1))
    , td0_(td0)
    , td1_(td1) {
    point_lidar_ = point_lidar.getVector3fMap().cast<double>();
    sqrt_info_   = 1.0 / std;
}

bool LidarFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
    // Pose_world_imu0
    Vector3d t_w_b0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond q_w_b0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
    // Pose_world_imu1
    Vector3d t_w_b1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond q_w_b1(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
    // Pose_imu_lidar
    Vector3d t_b_l(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond q_b_l(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    // td_imu_lidar
    double td = parameters[3][0];

    // 残差

    // Pose_world_imu0
    double dt0        = td - td0_;
    Matrix3d R_w_b0   = q_w_b0.toRotationMatrix();
    Matrix3d R_b0_bb0 = Matrix3d::Identity() + Rotation::skewSymmetric(omega0_ * dt0);
    Matrix3d R_w_bb0  = R_w_b0 * R_b0_bb0;
    Vector3d t_w_bb0  = t_w_b0 + vel0_ * dt0;
    // Pose_world_imu1
    double dt1        = td - td1_;
    Matrix3d R_w_b1   = q_w_b1.toRotationMatrix();
    Matrix3d R_b1_bb1 = Matrix3d::Identity() + Rotation::skewSymmetric(omega1_ * dt1);
    Matrix3d R_w_bb1  = R_w_b1 * R_b1_bb1;
    Vector3d t_w_bb1  = t_w_b1 + vel1_ * dt1;

    Matrix3d R_b_l = q_b_l.toRotationMatrix();
    Vector3d pb1   = R_b_l * point_lidar_ + t_b_l;
    Vector3d pw    = R_w_bb1 * pb1 + t_w_bb1;
    Vector3d pb0   = R_w_bb0.transpose() * (pw - t_w_bb0);
    Vector3d pl0   = R_b_l.transpose() * (pb0 - t_b_l);

    residuals[0] = sqrt_info_ * normal_vector_.transpose() * (pl0 - center_);

    if (jacobians) {
        Eigen::Matrix<double, 1, 3> sqrt_common = sqrt_info_ * normal_vector_.transpose() * R_b_l.transpose();

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jaco(jacobians[0]);
            jaco.setZero();

            jaco.block<1, 3>(0, 0) = -sqrt_common * R_w_bb0.transpose();
            jaco.block<1, 3>(0, 3) =
                sqrt_common * R_b0_bb0.transpose() * Rotation::skewSymmetric(R_w_b0.transpose() * (pw - t_w_bb0));
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jaco(jacobians[1]);
            jaco.setZero();

            jaco.block<1, 3>(0, 0) = sqrt_common * R_w_bb0.transpose();
            jaco.block<1, 3>(0, 3) =
                -sqrt_common * R_w_bb0.transpose() * R_w_b1 * Rotation::skewSymmetric(R_b1_bb1 * pb1);
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jaco(jacobians[2]);
            jaco.setZero();

            Matrix3d common                              = R_b_l.transpose() * R_w_bb0.transpose() * R_w_bb1;
            Eigen::Matrix<double, 1, 3> sqrt_unit_common = sqrt_info_ * normal_vector_.transpose();

            jaco.block<1, 3>(0, 0) = sqrt_unit_common * (common - R_b_l.transpose());
            jaco.block<1, 3>(0, 3) = sqrt_unit_common * (Rotation::skewSymmetric(pl0) -
                                                         common * R_b_l * Rotation::skewSymmetric(point_lidar_));
        }

        if (jacobians[3]) {
            jacobians[3][0] =
                sqrt_common * ((R_w_b0 * Rotation::skewSymmetric(omega0_)).transpose() * (pw - t_w_bb0) +
                               R_w_bb0.transpose() * (R_w_b1 * Rotation::skewSymmetric(omega1_) * pb1 + vel1_ - vel0_));
        }
    }

    return true;
}

LidarFrameToMapFactor::LidarFrameToMapFactor(const PointType &point_lidar, Vector3d normal_vector, Vector3d center,
                                             double std)
    : normal_vector_(std::move(normal_vector))
    , center_(std::move(center)) {
    point_lidar_ = point_lidar.getVector3fMap().cast<double>();
    sqrt_info_   = 1.0 / std;
}

bool LidarFrameToMapFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
    // Pose_world_lidar
    Vector3d t_w_l(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond q_w_l(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    // 残差
    Matrix3d R_w_l = q_w_l.toRotationMatrix();
    Vector3d pw    = R_w_l * point_lidar_ + t_w_l;
    residuals[0]   = sqrt_info_ * normal_vector_.transpose() * (pw - center_);

    if (jacobians) {
        Eigen::Matrix<double, 1, 3> sqrt_common = sqrt_info_ * normal_vector_.transpose();

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> jaco(jacobians[0]);
            jaco.setZero();

            jaco.block<1, 3>(0, 0) = sqrt_common;
            jaco.block<1, 3>(0, 3) = -sqrt_common * R_w_l * Rotation::skewSymmetric(point_lidar_);
        }
    }

    return true;
}
