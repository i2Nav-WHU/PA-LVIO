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

#include "factors/reprojection_factor.h"

ReprojectionFactor::ReprojectionFactor(Vector3d pts0, Vector3d pts1, Vector3d vel0, Vector3d vel1, double td0,
                                       double td1, double std)
    : pts0_(std::move(pts0))
    , pts1_(std::move(pts1))
    , vel0_(std::move(vel0))
    , vel1_(std::move(vel1))
    , td0_(td0)
    , td1_(td1) {

    sqrt_info_.setZero();
    sqrt_info_(0, 0) = 1.0 / std;
    sqrt_info_(1, 1) = 1.0 / std;
}

bool ReprojectionFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
    // 参考帧位姿
    Vector3d p0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond q0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d p1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond q1(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // 相机外参
    Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    // 逆深度
    double id0 = parameters[3][0];

    // 时间延时
    double td0 = parameters[4][0];
    double td1 = parameters[5][0];

    Vector3d pts_0_td = pts0_ - (td0 - td0_) * vel0_;
    Vector3d pts_1_td = pts1_ - (td1 - td1_) * vel1_;

    Vector3d pts_c_0 = pts_0_td / id0;
    Vector3d pts_b_0 = qic * pts_c_0 + tic;
    Vector3d pts_n   = q0 * pts_b_0 + p0;
    Vector3d pts_b_1 = q1.inverse() * (pts_n - p1);
    Vector3d pts_1   = qic.inverse() * (pts_b_1 - tic);

    double d1 = pts_1.z();

    // 残差, 没有考虑参考帧的残差
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = (pts_1 / d1).head(2) - pts_1_td.head(2);
    residual = sqrt_info_ * residual;

    if (jacobians) {
        Matrix3d cb0n = q0.toRotationMatrix();
        Matrix3d cnb1 = q1.toRotationMatrix().transpose();
        Matrix3d cbc  = qic.toRotationMatrix().transpose();
        Eigen::Matrix<double, 2, 3> reduce;
        reduce << 1.0 / d1, 0, -pts_1(0) / (d1 * d1), 0, 1.0 / d1, -pts_1(1) / (d1 * d1);

        reduce = sqrt_info_ * reduce;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco_i;
            jaco_i.leftCols<3>()  = cbc * cnb1;
            jaco_i.rightCols<3>() = -cbc * cnb1 * cb0n * Rotation::skewSymmetric(pts_b_0);

            jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
            jacobian_pose_i.rightCols<1>().setZero();
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco_j;
            jaco_j.leftCols<3>()  = -cbc * cnb1;
            jaco_j.rightCols<3>() = cbc * Rotation::skewSymmetric(pts_b_1);

            jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
            jacobian_pose_j.rightCols<1>().setZero();
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);

            Eigen::Matrix<double, 3, 6> jaco_ex;
            jaco_ex.leftCols<3>() = cbc * (cnb1 * cb0n - Matrix3d::Identity());
            Matrix3d tmp_r        = cbc * cnb1 * cb0n * cbc.transpose();

            jaco_ex.rightCols<3>() = -tmp_r * Rotation::skewSymmetric(pts_c_0) +
                                     Rotation::skewSymmetric(tmp_r * pts_c_0) +
                                     Rotation::skewSymmetric(cbc * (cnb1 * (cb0n * tic + p0 - p1) - tic));

            jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
            jacobian_ex_pose.rightCols<1>().setZero();
        }

        if (jacobians[3]) {
            Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
            jacobian_feature = -reduce * cbc * cnb1 * cb0n * cbc.transpose() * pts_0_td / (id0 * id0);
        }

        if (jacobians[4]) {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);
            jacobian_td = -reduce * cbc * cnb1 * cb0n * cbc.transpose() * vel0_ / id0;
        }

        if (jacobians[5]) {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[5]);
            jacobian_td = sqrt_info_ * vel1_.head(2);
        }
    }

    return true;
}

PoseOnlyReprojectionFactor::PoseOnlyReprojectionFactor(Vector3d pts0, Vector3d pts1, Vector3d pts2, Vector3d vel0,
                                                       Vector3d vel1, Vector3d vel2, double td0, double td1, double td2,
                                                       double std)
    : pts0_(std::move(pts0))
    , pts1_(std::move(pts1))
    , pts2_(std::move(pts2))
    , vel0_(std::move(vel0))
    , vel1_(std::move(vel1))
    , vel2_(std::move(vel2))
    , td0_(td0)
    , td1_(td1)
    , td2_(td2) {

    sqrt_info_.setZero();
    sqrt_info_(0, 0) = 1.0 / std;
    sqrt_info_(1, 1) = 1.0 / std;
}

bool PoseOnlyReprojectionFactor::Evaluate(const double *const *parameters, double *residuals,
                                          double **jacobians) const {
    // 位姿
    Vector3d t_w_b0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond q_w_b0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d t_w_b1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond q_w_b1(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Vector3d t_w_b2(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond q_w_b2(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    // 外参
    Vector3d t_b_c(parameters[3][0], parameters[3][1], parameters[3][2]);
    Quaterniond q_b_c(parameters[3][6], parameters[3][3], parameters[3][4], parameters[3][5]);

    // 延时
    double td0 = parameters[4][0];
    double td1 = parameters[5][0];
    double td2 = parameters[6][0];

    // 相机坐标系的位姿
    Matrix3d R_w_c0 = Rotation::quaternion2matrix(q_w_b0 * q_b_c);
    Vector3d t_w_c0 = q_w_b0 * t_b_c + t_w_b0;
    Matrix3d R_w_c1 = Rotation::quaternion2matrix(q_w_b1 * q_b_c);
    Vector3d t_w_c1 = q_w_b1 * t_b_c + t_w_b1;
    Matrix3d R_w_c2 = Rotation::quaternion2matrix(q_w_b2 * q_b_c);
    Vector3d t_w_c2 = q_w_b2 * t_b_c + t_w_b2;

    // 观测延时
    Vector3d pts_0_td = pts0_ - (td0 - td0_) * vel0_;
    Vector3d pts_1_td = pts1_ - (td1 - td1_) * vel1_;
    Vector3d pts_2_td = pts2_ - (td2 - td2_) * vel2_;

    // 相对位姿
    Matrix3d R_c1_c0 = R_w_c1.transpose() * R_w_c0;
    Vector3d t_c1_c0 = R_w_c1.transpose() * (t_w_c0 - t_w_c1);

    // 计算深度
    Matrix3d R_c2_c0   = R_w_c2.transpose() * R_w_c0;
    Vector3d t_c2_c0   = R_w_c2.transpose() * (t_w_c0 - t_w_c2);
    Vector3d pts_1_0   = R_c1_c0 * pts_0_td;
    Vector3d pts_2_0   = R_c2_c0 * pts_0_td;
    Matrix3d skew_p2   = Rotation::skewSymmetric(pts_2_td);
    Vector3d norm_vec  = skew_p2 * t_c2_c0;
    Vector3d theta_vec = skew_p2 * pts_2_0;

    double norm          = (skew_p2 * t_c2_c0).norm();
    double theta         = (theta_vec).norm();
    double norm_inverse  = 1.0 / norm;
    double theta_inverse = 1.0 / theta;

    // 计算的坐标
    Vector3d pts_1 = norm * pts_1_0 + theta * t_c1_c0;
    double d1      = pts_1.z();

    // 残差
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = (pts_1 / d1).head(2) - pts_1_td.head(2);
    residual = sqrt_info_ * residual;

    if (jacobians) {
        Eigen::Matrix<double, 2, 3> reduce;
        reduce << 1.0 / d1, 0, -pts_1(0) / (d1 * d1), 0, 1.0 / d1, -pts_1(1) / (d1 * d1);
        reduce = sqrt_info_ * reduce;

        Matrix3d R_c1_w  = R_w_c1.transpose();
        Matrix3d R_b_c   = q_b_c.toRotationMatrix();
        Matrix3d R_c_b   = q_b_c.toRotationMatrix().transpose();
        Matrix3d R_b1_w  = q_w_b1.toRotationMatrix().transpose();
        Matrix3d R_c1_b0 = R_c1_w * q_w_b0.toRotationMatrix();
        Matrix3d R_c2_w  = R_w_c2.transpose();
        Matrix3d R_c2_b0 = R_c2_w * q_w_b0.toRotationMatrix();
        Matrix3d R_b2_w  = q_w_b2.toRotationMatrix().transpose();

        Matrix3d skew_t_b_c        = Rotation::skewSymmetric(t_b_c);
        Matrix3d skew_t_c2_c0      = Rotation::skewSymmetric(t_c2_c0);
        Matrix3d skew_theta_prefix = Rotation::skewSymmetric(pts_2_0);
        Matrix3d norm_prefix       = (skew_p2 * R_c2_w).transpose();

        Eigen::RowVector3d jaco_norm_t = norm_inverse * (norm_prefix * norm_vec).transpose();
        Matrix3d jaco_t_c2_c0          = pts_1_0 * jaco_norm_t;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_0(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco;
            Matrix3d norm_r0_prefix  = -(skew_p2 * R_c2_b0 * skew_t_b_c).transpose();
            Matrix3d theta_r0_prefix = -(skew_p2 * R_c2_b0 * Rotation::skewSymmetric(R_b_c * pts_0_td)).transpose();

            Eigen::RowVector3d jaco_norm_r  = norm_inverse * (norm_r0_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r = theta_inverse * (theta_r0_prefix * theta_vec).transpose();

            jaco.leftCols<3>()  = jaco_t_c2_c0 + theta * R_c1_w;
            jaco.rightCols<3>() = pts_1_0 * jaco_norm_r - norm * R_c1_b0 * Rotation::skewSymmetric(q_b_c * pts_0_td) -
                                  theta * R_c1_b0 * skew_t_b_c + t_c1_c0 * jaco_theta_r;

            jacobian_pose_0.leftCols<6>() = reduce * jaco;
            jacobian_pose_0.rightCols<1>().setZero();
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_1(jacobians[1]);

            Eigen::Matrix<double, 3, 6> jaco;
            jaco.leftCols<3>()  = -theta * R_c1_w;
            jaco.rightCols<3>() = norm * R_c_b * Rotation::skewSymmetric(R_b1_w * R_w_c0 * pts_0_td) +
                                  theta * R_c_b * Rotation::skewSymmetric(R_b1_w * (t_w_c0 - t_w_c1)) +
                                  theta * R_c_b * skew_t_b_c;

            jacobian_pose_1.leftCols<6>() = reduce * jaco;
            jacobian_pose_1.rightCols<1>().setZero();
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_2(jacobians[2]);

            Eigen::Matrix<double, 3, 6> jaco;
            Matrix3d norm_r2_prefix =
                (skew_p2 * R_c_b * skew_t_b_c + skew_p2 * R_c_b * Rotation::skewSymmetric(R_b2_w * (t_w_c0 - t_w_c2)))
                    .transpose();
            Matrix3d theta_r2_prefix =
                (skew_p2 * R_c_b * Rotation::skewSymmetric(R_b2_w * R_w_c0 * pts_0_td)).transpose();

            Eigen::RowVector3d jaco_norm_r  = norm_inverse * (norm_r2_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r = theta_inverse * (theta_r2_prefix * theta_vec).transpose();

            jaco.leftCols<3>()            = -jaco_t_c2_c0;
            jaco.rightCols<3>()           = pts_1_0 * jaco_norm_r + t_c1_c0 * jaco_theta_r;
            jacobian_pose_2.leftCols<6>() = reduce * jaco;
            jacobian_pose_2.rightCols<1>().setZero();
        }

        if (jacobians[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_extrinsic(jacobians[3]);

            Eigen::Matrix<double, 3, 6> jaco;
            Matrix3d norm_t_prefix = (skew_p2 * (R_c2_b0 - R_c_b)).transpose();
            Matrix3d norm_r_prefix = (skew_p2 * skew_t_c2_c0).transpose();
            Matrix3d r_r0_prefix   = -R_c1_c0 * Rotation::skewSymmetric(pts_0_td);
            Matrix3d r_r1_prefix   = Rotation::skewSymmetric(pts_1_0);

            Matrix3d theta_r0_prefix = -(skew_p2 * R_c2_c0 * Rotation::skewSymmetric(pts_0_td)).transpose();
            Matrix3d theta_r2_prefix = (skew_p2 * skew_theta_prefix).transpose();
            Matrix3d t_prefix        = Rotation::skewSymmetric(t_c1_c0);

            Eigen::RowVector3d jaco_norm_t = norm_inverse * (norm_t_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_norm_r = norm_inverse * (norm_r_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r =
                theta_inverse * ((theta_r0_prefix + theta_r2_prefix) * theta_vec).transpose();

            jaco.leftCols<3>()  = theta * (R_c1_b0 - R_c_b) + pts_1_0 * jaco_norm_t;
            jaco.rightCols<3>() = pts_1_0 * jaco_norm_r + norm * r_r0_prefix + norm * r_r1_prefix +
                                  t_c1_c0 * jaco_theta_r + theta * t_prefix;
            jacobian_extrinsic.leftCols<6>() = reduce * jaco;
            jacobian_extrinsic.rightCols<1>().setZero();
        }

        if (jacobians[4]) {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);

            double jaco_theta_p0 = -theta_inverse * ((skew_p2 * R_c2_c0) * vel0_).transpose() * theta_vec;

            Vector3d jaco = -norm * R_c1_c0 * vel0_ + jaco_theta_p0 * t_c1_c0;
            jacobian_td   = reduce * jaco;
        }

        if (jacobians[5]) {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[5]);

            jacobian_td = sqrt_info_ * vel1_.head(2);
        }

        if (jacobians[6]) {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[6]);

            double jaco_norm_p2 = -norm_inverse * (skew_t_c2_c0 * vel2_).transpose() * skew_t_c2_c0 * pts_2_td;
            double jaco_theta_p2 =
                -theta_inverse * (skew_theta_prefix * vel2_).transpose() * skew_theta_prefix * pts_2_td;

            Vector3d jaco = jaco_norm_p2 * pts_1_0 + jaco_theta_p2 * t_c1_c0;
            jacobian_td   = reduce * jaco;
        }
    }

    return true;
}

PoseOnlyTwoViewReprojectionFactor::PoseOnlyTwoViewReprojectionFactor(Vector3d pts0, Vector3d pts1, Vector3d vel0,
                                                                     Vector3d vel1, double td0, double td1, double std)
    : pts0_(std::move(pts0))
    , pts1_(std::move(pts1))
    , vel0_(std::move(vel0))
    , vel1_(std::move(vel1))
    , td0_(td0)
    , td1_(td1) {

    sqrt_info_.setZero();
    sqrt_info_(0, 0) = 1.0 / std;
    sqrt_info_(1, 1) = 1.0 / std;
}

bool PoseOnlyTwoViewReprojectionFactor::Evaluate(const double *const *parameters, double *residuals,
                                                 double **jacobians) const {
    // 位姿
    Vector3d t_w_b0(parameters[0][0], parameters[0][1], parameters[0][2]);
    Quaterniond q_w_b0(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Vector3d t_w_b1(parameters[1][0], parameters[1][1], parameters[1][2]);
    Quaterniond q_w_b1(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    // 外参
    Vector3d t_b_c(parameters[2][0], parameters[2][1], parameters[2][2]);
    Quaterniond q_b_c(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    // 延时
    double td0 = parameters[3][0];
    double td1 = parameters[4][0];

    // 相机坐标系的位姿
    Matrix3d R_w_c0 = Rotation::quaternion2matrix(q_w_b0 * q_b_c);
    Vector3d t_w_c0 = q_w_b0 * t_b_c + t_w_b0;
    Matrix3d R_w_c1 = Rotation::quaternion2matrix(q_w_b1 * q_b_c);
    Vector3d t_w_c1 = q_w_b1 * t_b_c + t_w_b1;

    // 观测延时
    Vector3d pts_0_td = pts0_ - (td0 - td0_) * vel0_;
    Vector3d pts_1_td = pts1_ - (td1 - td1_) * vel1_;

    // 相对位姿
    Matrix3d R_c1_c0 = R_w_c1.transpose() * R_w_c0;
    Vector3d t_c1_c0 = R_w_c1.transpose() * (t_w_c0 - t_w_c1);

    Matrix3d R_w_b2 = q_w_b1.toRotationMatrix();
    Matrix3d R_w_c2 = R_w_c1;
    Vector3d t_w_c2 = t_w_c1;

    // 计算深度
    Matrix3d R_c2_c0   = R_w_c2.transpose() * R_w_c0;
    Vector3d t_c2_c0   = R_w_c2.transpose() * (t_w_c0 - t_w_c2);
    Vector3d pts_1_0   = R_c1_c0 * pts_0_td;
    Vector3d pts_2_0   = R_c2_c0 * pts_0_td;
    Matrix3d skew_p2   = Rotation::skewSymmetric(pts_1_td);
    Vector3d norm_vec  = skew_p2 * t_c2_c0;
    Vector3d theta_vec = skew_p2 * pts_2_0;

    double norm          = (skew_p2 * t_c2_c0).norm();
    double theta         = (theta_vec).norm();
    double norm_inverse  = 1.0 / norm;
    double theta_inverse = 1.0 / theta;

    // 计算的坐标
    Vector3d pts_1 = norm * pts_1_0 + theta * t_c1_c0;
    double d1      = pts_1.z();

    // 残差
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual = (pts_1 / d1).head(2) - pts_1_td.head(2);
    residual = sqrt_info_ * residual;

    if (jacobians) {
        Eigen::Matrix<double, 2, 3> reduce;
        reduce << 1.0 / d1, 0, -pts_1(0) / (d1 * d1), 0, 1.0 / d1, -pts_1(1) / (d1 * d1);
        reduce = sqrt_info_ * reduce;

        Matrix3d R_c1_w  = R_w_c1.transpose();
        Matrix3d R_b_c   = q_b_c.toRotationMatrix();
        Matrix3d R_c_b   = q_b_c.toRotationMatrix().transpose();
        Matrix3d R_b1_w  = q_w_b1.toRotationMatrix().transpose();
        Matrix3d R_c1_b0 = R_c1_w * q_w_b0.toRotationMatrix();
        Matrix3d R_c2_w  = R_w_c2.transpose();
        Matrix3d R_c2_b0 = R_c2_w * q_w_b0.toRotationMatrix();
        Matrix3d R_b2_w  = R_w_b2.transpose();

        Matrix3d skew_t_b_c        = Rotation::skewSymmetric(t_b_c);
        Matrix3d skew_t_c2_c0      = Rotation::skewSymmetric(t_c2_c0);
        Matrix3d skew_theta_prefix = Rotation::skewSymmetric(pts_2_0);
        Matrix3d norm_prefix       = (skew_p2 * R_c2_w).transpose();

        Eigen::RowVector3d jaco_norm_t = norm_inverse * (norm_prefix * norm_vec).transpose();
        Matrix3d jaco_t_c2_c0          = pts_1_0 * jaco_norm_t;

        if (jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_0(jacobians[0]);

            Eigen::Matrix<double, 3, 6> jaco;
            Matrix3d norm_r0_prefix  = -(skew_p2 * R_c2_b0 * skew_t_b_c).transpose();
            Matrix3d theta_r0_prefix = -(skew_p2 * R_c2_b0 * Rotation::skewSymmetric(R_b_c * pts_0_td)).transpose();

            Eigen::RowVector3d jaco_norm_r  = norm_inverse * (norm_r0_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r = theta_inverse * (theta_r0_prefix * theta_vec).transpose();

            jaco.leftCols<3>()  = jaco_t_c2_c0 + theta * R_c1_w;
            jaco.rightCols<3>() = pts_1_0 * jaco_norm_r - norm * R_c1_b0 * Rotation::skewSymmetric(q_b_c * pts_0_td) -
                                  theta * R_c1_b0 * skew_t_b_c + t_c1_c0 * jaco_theta_r;

            jacobian_pose_0.leftCols<6>() = reduce * jaco;
            jacobian_pose_0.rightCols<1>().setZero();
        }

        if (jacobians[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_1(jacobians[1]);

            Matrix3d norm_r2_prefix =
                (skew_p2 * R_c_b * skew_t_b_c + skew_p2 * R_c_b * Rotation::skewSymmetric(R_b2_w * (t_w_c0 - t_w_c2)))
                    .transpose();
            Matrix3d theta_r2_prefix =
                (skew_p2 * R_c_b * Rotation::skewSymmetric(R_b2_w * R_w_c0 * pts_0_td)).transpose();
            Eigen::RowVector3d jaco_norm_r  = norm_inverse * (norm_r2_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r = theta_inverse * (theta_r2_prefix * theta_vec).transpose();

            Eigen::Matrix<double, 3, 6> jaco;
            jaco.leftCols<3>()  = -theta * R_c1_w - jaco_t_c2_c0;
            jaco.rightCols<3>() = norm * R_c_b * Rotation::skewSymmetric(R_b1_w * R_w_c0 * pts_0_td) +
                                  theta * R_c_b * Rotation::skewSymmetric(R_b1_w * (t_w_c0 - t_w_c1)) +
                                  theta * R_c_b * skew_t_b_c + pts_1_0 * jaco_norm_r + t_c1_c0 * jaco_theta_r;

            jacobian_pose_1.leftCols<6>() = reduce * jaco;
            jacobian_pose_1.rightCols<1>().setZero();
        }

        if (jacobians[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_extrinsic(jacobians[2]);

            Eigen::Matrix<double, 3, 6> jaco;
            Matrix3d norm_t_prefix = (skew_p2 * (R_c2_b0 - R_c_b)).transpose();
            Matrix3d norm_r_prefix = (skew_p2 * skew_t_c2_c0).transpose();
            Matrix3d r_r0_prefix   = -R_c1_c0 * Rotation::skewSymmetric(pts_0_td);
            Matrix3d r_r1_prefix   = Rotation::skewSymmetric(pts_1_0);

            Matrix3d theta_r0_prefix = -(skew_p2 * R_c2_c0 * Rotation::skewSymmetric(pts_0_td)).transpose();
            Matrix3d theta_r2_prefix = (skew_p2 * skew_theta_prefix).transpose();
            Matrix3d t_prefix        = Rotation::skewSymmetric(t_c1_c0);

            Eigen::RowVector3d jaco_norm_t = norm_inverse * (norm_t_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_norm_r = norm_inverse * (norm_r_prefix * norm_vec).transpose();
            Eigen::RowVector3d jaco_theta_r =
                theta_inverse * ((theta_r0_prefix + theta_r2_prefix) * theta_vec).transpose();

            jaco.leftCols<3>()  = theta * (R_c1_b0 - R_c_b) + pts_1_0 * jaco_norm_t;
            jaco.rightCols<3>() = pts_1_0 * jaco_norm_r + norm * r_r0_prefix + norm * r_r1_prefix +
                                  t_c1_c0 * jaco_theta_r + theta * t_prefix;
            jacobian_extrinsic.leftCols<6>() = reduce * jaco;
            jacobian_extrinsic.rightCols<1>().setZero();
        }

        if (jacobians[3]) {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[3]);

            double jaco_theta_p0 = -theta_inverse * ((skew_p2 * R_c2_c0) * vel0_).transpose() * theta_vec;

            Vector3d jaco = -norm * R_c1_c0 * vel0_ + jaco_theta_p0 * t_c1_c0;
            jacobian_td   = reduce * jaco;
        }

        if (jacobians[4]) {
            Eigen::Map<Eigen::Vector2d> jacobian_td(jacobians[4]);

            double jaco_norm_p2 = -norm_inverse * (skew_t_c2_c0 * vel1_).transpose() * skew_t_c2_c0 * pts_1_td;
            double jaco_theta_p2 =
                -theta_inverse * (skew_theta_prefix * vel1_).transpose() * skew_theta_prefix * pts_1_td;

            Vector3d jaco = jaco_norm_p2 * pts_1_0 + jaco_theta_p2 * t_c1_c0;
            jacobian_td   = reduce * jaco + sqrt_info_ * vel1_.head(2);
        }
    }

    return true;
}
