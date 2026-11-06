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

#include "factors/lidar_depth_msc_factor.h"

#include "common/logging.h"

#include <fstream>

LiDARDepthMscFactor::LiDARDepthMscFactor(const vector<std::shared_ptr<ceres::CostFunction>> costs, int num_frames,
                                         double mappoint_depth)
    : costs_(costs)
    , num_frames_(num_frames)
    , mappoint_depth_(mappoint_depth) {

    // 参数块维度
    vector<int> parameter_block_sizes;
    for (int k = 0; k < num_frames_; k++) {
        // 位姿和时间偏移
        parameter_block_sizes.push_back(7);
        parameter_block_sizes.push_back(1);
    }
    // 外参
    parameter_block_sizes.push_back(7);
    *mutable_parameter_block_sizes() = parameter_block_sizes;

    // 残差维度
    int num_residuals = (num_frames_ - 1) * 2;
    set_num_residuals(num_residuals);
}

bool LiDARDepthMscFactor::Evaluate(const double *const *parameters, double *residuals, double **jacobians) const {
    vector<const double *> paras;
    vector<double *> jacos;
    double invdep = 1.0 / mappoint_depth_;

    // 重投影参数
    vector<Eigen::Vector2d> reproj_residuals(num_frames_ - 1);
    vector<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> reproj_jaco_pose0(num_frames_ - 1);
    vector<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> reproj_jaco_pose1(num_frames_ - 1);
    vector<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> reproj_jaco_ext(num_frames_ - 1);
    vector<Eigen::Vector2d> reproj_jaco_invdep(num_frames_ - 1);
    vector<Eigen::Vector2d> reproj_jaco_td0(num_frames_ - 1);
    vector<Eigen::Vector2d> reproj_jaco_td1(num_frames_ - 1);

    // 计算残差和雅可比
    for (int k = 1; k < num_frames_; k++) {
        paras.clear();
        jacos.clear();
        paras.push_back(parameters[0]);               // pose0
        paras.push_back(parameters[k * 2]);           // pose1
        paras.push_back(parameters[num_frames_ * 2]); // ext
        paras.push_back(&invdep);                     // invdep
        paras.push_back(parameters[1]);               // td0
        paras.push_back(parameters[k * 2 + 1]);       // td1

        jacos.push_back(reproj_jaco_pose0[k - 1].data());  // pose0
        jacos.push_back(reproj_jaco_pose1[k - 1].data());  // pose1
        jacos.push_back(reproj_jaco_ext[k - 1].data());    // ext
        jacos.push_back(reproj_jaco_invdep[k - 1].data()); // invdep
        jacos.push_back(reproj_jaco_td0[k - 1].data());    // td0
        jacos.push_back(reproj_jaco_td1[k - 1].data());    // td1

        costs_[k - 1]->Evaluate(paras.data(), reproj_residuals[k - 1].data(), jacos.data());
    }
    // LiDAR深度因子参数
    double depth_residual;
    double depth_jaco_invdep;
    paras.clear();
    jacos.clear();
    paras.push_back(&invdep);
    jacos.push_back(&depth_jaco_invdep);
    costs_.back()->Evaluate(paras.data(), &depth_residual, jacos.data());

    // MSC边缘化, 加上LiDAR深度残差和逆深度状态
    int num_residuals = (num_frames_ - 1) * 2 + 1;
    int num_paras     = std::accumulate(parameter_block_sizes().begin(), parameter_block_sizes().end(), 0);

    Eigen::MatrixXd msc_residuals, msc_jaco_reserved, msc_jaco_invdep;
    msc_residuals.setZero(num_residuals, 1);
    msc_jaco_reserved.setZero(num_residuals, num_paras);
    msc_jaco_invdep.setZero(num_residuals, 1);

    for (int k = 0; k < num_frames_ - 1; k++) {
        msc_jaco_reserved.block(k * 2, 0, 2, 7)               = reproj_jaco_pose0[k];
        msc_jaco_reserved.block(k * 2, 7, 2, 1)               = reproj_jaco_td0[k];
        msc_jaco_reserved.block(k * 2, k * 8 + 8, 2, 7)       = reproj_jaco_pose1[k];
        msc_jaco_reserved.block(k * 2, k * 8 + 15, 2, 1)      = reproj_jaco_td1[k];
        msc_jaco_reserved.block(k * 2, num_frames_ * 8, 2, 7) = reproj_jaco_ext[k];

        msc_jaco_invdep.block(k * 2, 0, 2, 1) = reproj_jaco_invdep[k];
        msc_residuals.block(k * 2, 0, 2, 1)   = reproj_residuals[k];
    }
    msc_jaco_invdep(num_residuals - 1, 0) = depth_jaco_invdep;
    msc_residuals(num_residuals - 1, 0)   = depth_residual;

    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(num_residuals, 1);
    qr.compute(msc_jaco_invdep);
    Eigen::MatrixXd Q  = qr.householderQ();
    Eigen::MatrixXd Q1 = Q.block(0, 0, Q.rows(), 1);
    Eigen::MatrixXd Q2 = Q.block(0, 1, Q.rows(), Q.cols() - 1);

    Eigen::Matrix<double, -1, -1> residual;
    residual = Q2.transpose() * msc_residuals;
    mempcpy(residuals, residual.data(), sizeof(double) * residual.size());

    if (jacobians) {
        Eigen::MatrixXd msc_jacobian = Q2.transpose() * msc_jaco_reserved;

        int num_residuals = (num_frames_ - 1) * 2;
        for (int k = 0; k < num_frames_; k++) {
            // Pose
            if (jacobians[k * 2]) {
                Eigen::Matrix<double, -1, -1, Eigen::RowMajor> jacobian;
                jacobian.setZero(num_residuals, 7);

                jacobian = msc_jacobian.block(0, k * 8, num_residuals, 7);
                memcpy(jacobians[k * 2], jacobian.data(), sizeof(double) * jacobian.size());
            }
            // Td
            if (jacobians[k * 2 + 1]) {
                Eigen::Matrix<double, -1, -1, Eigen::RowMajor> jacobian;
                jacobian.setZero(num_residuals, 1);

                jacobian = msc_jacobian.block(0, k * 8 + 7, num_residuals, 1);
                memcpy(jacobians[k * 2 + 1], jacobian.data(), sizeof(double) * jacobian.size());
            }
        }
        // Extrinsic
        if (jacobians[num_frames_ * 2]) {
            Eigen::Matrix<double, -1, -1, Eigen::RowMajor> jacobian;
            jacobian.setZero(num_residuals, 7);

            jacobian = msc_jacobian.block(0, num_frames_ * 8, num_residuals, 7);
            memcpy(jacobians[num_frames_ * 2], jacobian.data(), sizeof(double) * jacobian.size());
        }
    }

    return true;
}
