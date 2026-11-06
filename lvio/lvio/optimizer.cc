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

#include "lvio/optimizer.h"

#include "common/logging.h"
#include "common/misc.h"
#include "common/timecost.h"

#include "lidar/pointcloud.h"
#include "visual/tracking.h"

#include "factors/depth_factor.h"
#include "factors/lidar_depth_msc_factor.h"
#include "factors/lidar_factor.h"
#include "factors/lidar_pose_factor.h"
#include "factors/marginalization_factor.h"
#include "factors/marginalization_info.h"
#include "factors/pose_manifold.h"
#include "factors/reprojection_factor.h"
#include "factors/residual_block_info.h"
#include "factors/time_delay_factor.h"

#include "preintegration/imu_error_factor.h"
#include "preintegration/imu_mix_prior_factor.h"
#include "preintegration/imu_pose_prior_factor.h"
#include "preintegration/imu_zero_velocity_factor.h"
#include "preintegration/preintegration.h"
#include "preintegration/preintegration_factor.h"

#include <tbb/tbb.h>
#include <yaml-cpp/yaml.h>

Optimizer::Optimizer(const string &configfile, const std::shared_ptr<const IntegrationParameters> parameters,
                     visual::VisualMap::Ptr visual_map, visual::Camera::Ptr camera, lidar::LidarMap::Ptr lidar_map)
    : integration_parameters_(std::move(parameters))
    , visual_map_(std::move(visual_map))
    , camera_(std::move(camera))
    , lidar_map_(std::move(lidar_map)) {

    // 配置参数
    YAML::Node config;
    vector<double> vecdata;
    config = YAML::LoadFile(configfile);

    // 优化参数
    optimize_reprojection_std_     = config["optimizer"]["optimize_reprojection_std"].as<double>();
    optimize_point_to_plane_std_   = config["optimizer"]["optimize_point_to_plane_std"].as<double>();
    optimize_windows_size_         = config["optimizer"]["optimize_window_size"].as<size_t>();
    optimize_calculate_covariance_ = config["optimizer"]["optimize_calculate_covariance"].as<bool>();
    optimize_half_windows_size_    = optimize_windows_size_ / 2;

    optimize_estimate_cam_extrinsic_ = config["optimizer"]["optimize_estimate_cam_extrinsic"].as<bool>();
    optimize_estimate_cam_td_        = config["optimizer"]["optimize_estimate_cam_td"].as<bool>();
    optimize_cam_extrinsic_accurate_ = config["optimizer"]["optimize_cam_extrinsic_accurate"].as<bool>();

    optimize_estimate_lidar_extrinsic_ = config["optimizer"]["optimize_estimate_lidar_extrinsic"].as<bool>();
    optimize_estimate_lidar_td_        = config["optimizer"]["optimize_estimate_lidar_td"].as<bool>();
    optimize_lidar_extrinsic_accurate_ = config["optimizer"]["optimize_lidar_extrinsic_accurate"].as<bool>();

    // 归一化相机坐标系下
    optimize_reprojection_factor_std_ = optimize_reprojection_std_ / camera_->focalLength();

    gb_prior_std_ = config["imu"]["gb_prior_std"].as<double>() * D2R / 3600.0; // deg/hr -> rad/s
    ab_prior_std_ = config["imu"]["ab_prior_std"].as<double>() * 1.0e-5;       // mGal -> m/s^2
    imudatarate_  = config["imu"]["imudatarate"].as<double>();
    if (config["imu"]["is_use_zupt"]) {
        is_use_zupt_ = config["imu"]["is_use_zupt"].as<bool>();
    }

    // 相机参数
    is_use_visual_      = config["visual"]["is_use_visual"].as<bool>();
    is_use_lidar_depth_ = config["visual"]["is_use_lidar_depth"].as<bool>();
    vecdata             = config["visual"]["q_b_c"].as<vector<double>>();
    Quaterniond q_b_c   = Eigen::Quaterniond(vecdata.data()).normalized();
    vecdata             = config["visual"]["t_b_c"].as<vector<double>>();
    Vector3d t_b_c      = Eigen::Vector3d(vecdata.data());
    double td_b_c       = config["visual"]["td_b_c"].as<double>();
    pose_b_c_.R         = q_b_c.toRotationMatrix();
    pose_b_c_.t         = t_b_c;

    memcpy(extrinsic_b_c_, t_b_c.data(), sizeof(double) * 3);
    memcpy(extrinsic_b_c_ + 3, q_b_c.coeffs().data(), sizeof(double) * 4); // x, y, z, w
    extrinsic_b_c_[7] = td_b_c;

    // 激光参数
    is_use_lidar_ = config["lidar"]["is_use_lidar"].as<bool>();
    if (is_use_lidar_) {
        if (config["lidar"]["is_use_frame_to_map"]) {
            is_use_frame_to_map_ = config["lidar"]["is_use_frame_to_map"].as<bool>();
        }
    }
    vecdata           = config["lidar"]["q_b_l"].as<std::vector<double>>();
    Quaterniond q_b_l = Eigen::Quaterniond(vecdata.data());
    vecdata           = config["lidar"]["t_b_l"].as<std::vector<double>>();
    Vector3d t_b_l    = Eigen::Vector3d(vecdata.data());
    double td_b_l     = config["lidar"]["td_b_l"].as<double>();
    pose_b_l_.R       = q_b_l.toRotationMatrix();
    pose_b_l_.t       = t_b_l;

    memcpy(extrinsic_b_l_, t_b_l.data(), sizeof(double) * 3);
    memcpy(extrinsic_b_l_ + 3, q_b_l.coeffs().data(), sizeof(double) * 4); // x, y, z, w
    extrinsic_b_l_[7] = td_b_l;

    // 统计参数
    iterations_.resize(2);
    timecosts_.resize(4);
    outliers_.resize(3);

    LOGI << "Optimizer is constructed";
}

bool Optimizer::optimization(const deque<double> &timelist, deque<StateData> &statedatalist,
                             deque<PreBasePtr> &preintegrationlist) {
    static int first_num_iterations  = 5;
    static int second_num_iterations = 10;

    TimeCost timecost;

    // 方便后续移除粗差因子
    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type         = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations         = first_num_iterations;
    options.num_threads                = 8;

    // 状态参数
    addStateParameters(problem, statedatalist);

    // 重投影参数
    if (is_use_visual_) {
        addVisualParameters(problem);
        LOGI << "Add " << visual_mappoints_.size() << " visual landmarks to optimizer";
    }

    // 激光参数
    if (is_use_lidar_) {
        addLidarParameters(problem);
    }

    // 边缘化残差
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {
        auto factor = new MarginalizationFactor(last_marginalization_info_);
        problem.AddResidualBlock(factor, nullptr, last_marginalization_parameter_blocks_);
    }

    // 预积分残差
    addImuFactors(problem, statedatalist, preintegrationlist);

    // 视觉重投影残差
    Residuals visual_residuals;
    if (is_use_visual_) {
        visual_residuals = addReprojectionFactors(problem, true, timelist, statedatalist);
    }

    // 激光点到面残差
    LidarResiduals lidar_residuals;
    if (is_use_lidar_) {
        lidar_residuals = addLidarFactors(problem, true, timelist, statedatalist);
    }

    string msg = absl::StrFormat("Add %llu preintegration", preintegrationlist.size());
    if (is_use_visual_ && !visual_residuals.empty()) {
        absl::StrAppendFormat(&msg, ", %llu visual", visual_residuals.size());
    }
    if (is_use_lidar_ && !lidar_residuals.empty()) {
        absl::StrAppendFormat(&msg, ", %llu lidar", lidar_residuals.size());
    }
    LOGI << msg << " factors";

    // 第一次优化
    {
        timecost.restart();

        // 求解最小二乘
        solver.Solve(options, &problem, &summary);
        // LOGI << summary.BriefReport();

        iterations_[0] = summary.num_successful_steps;
        timecosts_[0]  = timecost.costInMillisecond();
    }

    // 粗差检测和剔除
    {
        // Detect outlier lidar factors
        ResidualOutliers lidar_outliers;
        if (is_use_lidar_) {
            lidar_outliers = lidarOutlierCullingByChi2(problem, lidar_residuals);
        }

        // Detect outlier visual factors
        ResidualOutliers visual_outliers;
        if (is_use_visual_) {
            visual_outliers = visualOutlierCullingByChi2(problem, visual_residuals);
        }

        // Remove factors in the final

        if (is_use_lidar_) {
            int num_outliers = 0;
            // Remove factors only
            for (const auto &outlier : lidar_outliers) {
                if (outlier.second) {
                    problem.RemoveResidualBlock(outlier.first);
                    num_outliers++;
                }
            }
            outliers_[2] = num_outliers;
            LOGI << "Remove " << num_outliers << " lidar outliers factors";
        }
        if (is_use_visual_) {
            int num_outliers = 0;
            // Remove outliers only
            for (const auto &outlier : visual_outliers) {
                if (outlier.second) {
                    problem.RemoveResidualBlock(outlier.first);
                    num_outliers++;
                }
            }
            LOGI << "Remove " << num_outliers << " reprojection outlier factors";
        }
    }

    // 第二次优化
    {
        // 第二次迭代次数
        options.max_num_iterations = second_num_iterations;

        timecost.restart();

        // 求解最小二乘
        solver.Solve(options, &problem, &summary);
        // LOGI << summary.BriefReport();

        iterations_[1] = summary.num_successful_steps;
        timecosts_[1]  = timecost.costInMillisecond();

        if ((is_use_visual_ && !visual_map_->isWindowFull()) || (is_use_lidar_ && !lidar_map_->isWindowFull())) {
            // 进行必要的重积分
            doReintegration(preintegrationlist, statedatalist);
        }
    }

    if (optimize_calculate_covariance_) {
        // 获取最新状态标准差
        getLatestStateStd(problem, statedatalist.back(), latest_state_std_);
    }

    if (is_use_visual_ && is_use_lidar_depth_ && !depth_mappoints_.empty()) {
        // 使用重投影优化深度
        depthOnlyOptimization(timelist, statedatalist);
    }

    // 更新参数, 必须的
    updateParametersFromOptimizer(timelist, statedatalist);

    // 移除粗差路标点
    if (is_use_visual_) {
        visualLandmarkOutlierCulling();
    }

    LOGI << "Optimization costs " << timecosts_[0] << " ms and " << timecosts_[1] << ", with iteration "
         << iterations_[0] << " and " << iterations_[1];
    return true;
}

void Optimizer::lidarFrameToMapOptimization(lidar::LidarFrame::Ptr frame) {
    TimeCost timecost;

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = true;

    ceres::Problem problem(problem_options);
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type         = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations         = 5;
    options.num_threads                = 8;

    LidarResiduals residual_block;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

    ceres::Manifold *manifold = new PoseManifold();
    problem.AddParameterBlock(frame->frameToMapPose(), 7, manifold);
    // problem.AddParameterBlock(extrinsic_b_l_, 7, manifold);
    // problem.AddParameterBlock(&extrinsic_b_l_[7], 1);
    // problem.SetParameterBlockConstant(extrinsic_b_l_);
    // problem.SetParameterBlockConstant(&extrinsic_b_l_[7]);

    // 先验位姿参数
    double pos_std           = 0.25;  // 0.25 m
    double att_std           = 0.002; // 0.1 deg
    double prior_pose_std[6] = {pos_std, pos_std, pos_std, att_std, att_std, att_std};
    auto prior_factor        = new ImuPosePriorFactor(frame->frameToMapPose(), prior_pose_std);
    problem.AddResidualBlock(prior_factor, nullptr, frame->frameToMapPose());

    const auto &features     = frame->globalMapFeatures();
    const auto &lidar_points = frame->pointCloudLidar()->points;
    for (size_t k = 0; k < features.size(); k++) {
        const auto &feature = features[k];
        if (!feature || (feature->featureType() != lidar::FEATURE_PLANE) || feature->isOutlier()) {
            continue;
        }

        auto factor =
            new LidarFrameToMapFactor(lidar_points[k], feature->normalVector(), feature->center(), feature->pointStd());
        auto residual_block_id = problem.AddResidualBlock(factor, loss_function, frame->frameToMapPose());
        residual_block.emplace_back(residual_block_id, feature);
    }

    solver.Solve(options, &problem, &summary);
    ResidualOutliers lidar_outliers = lidarOutlierCullingByChi2(problem, residual_block);
    int outliers_count              = 0;
    for (const auto &outlier : lidar_outliers) {
        if (outlier.second) {
            problem.RemoveResidualBlock(outlier.first);
            outliers_count++;
        }
    }
    options.max_num_iterations = 10;
    solver.Solve(options, &problem, &summary);

    // 计算最新状态的协方差
    ceres::Covariance::Options cov_options;
    cov_options.apply_loss_function = true;
    cov_options.algorithm_type      = ceres::SPARSE_QR;
    cov_options.num_threads         = 8;

    ceres::Covariance covariance(cov_options);

    vector<std::pair<const double *, const double *>> covariance_blocks;
    covariance_blocks.push_back(std::make_pair(frame->frameToMapPose(), frame->frameToMapPose()));

    if (covariance.Compute(covariance_blocks, &problem)) {
        double covariance_pose[6 * 6];
        covariance.GetCovarianceBlockInTangentSpace(frame->frameToMapPose(), frame->frameToMapPose(), covariance_pose);
        for (int k = 0; k < 6; k++) {
            frame->frameToMapPoseStd()[k] = sqrt(covariance_pose[k * 6 + k]);
        }
    } else {
        LOGW << "Failed to calculate frame-to-map pose covaraince";
        memcpy(frame->frameToMapPoseStd(), prior_pose_std, sizeof(double) * 6);
    }
    LOGI << timecost.costInMillisecond("LiDAR frame to map optimization") << " with " << residual_block.size()
         << " features and " << outliers_count << " outliers";
}


void Optimizer::getLatestStateStd(ceres::Problem &problem, const StateData &statedata, vector<double> &std) {
    TimeCost timecost;

    ceres::Covariance::Options options;
    options.apply_loss_function = true;
    options.algorithm_type      = ceres::SPARSE_QR;
    options.num_threads         = 8;

    ceres::Covariance covariance(options);

    // 计算最新状态的协方差
    vector<std::pair<const double *, const double *>> covariance_blocks;
    covariance_blocks.push_back(std::make_pair(statedata.pose, statedata.pose));
    covariance_blocks.push_back(std::make_pair(statedata.mix, statedata.mix));
    covariance.Compute(covariance_blocks, &problem);

    // 获取协方差
    constexpr int num_mix = 9;
    double covariance_pose[6 * 6];
    double covariance_mix[num_mix * num_mix];

    covariance.GetCovarianceBlockInTangentSpace(statedata.pose, statedata.pose, covariance_pose);
    covariance.GetCovarianceBlock(statedata.mix, statedata.mix, covariance_mix);

    std.resize(16, 0);
    std[0] = statedata.time;
    for (size_t k = 0; k < 3; k++) {
        std[k + 1]  = sqrt(covariance_pose[6 * k + k]);                // 位置
        std[k + 4]  = sqrt(covariance_mix[num_mix * k + k]);           // 速度
        std[k + 7]  = sqrt(covariance_pose[6 * (k + 3) + k + 3]);      // 姿态
        std[k + 10] = sqrt(covariance_mix[num_mix * (k + 3) + k + 3]); // 陀螺零偏
        std[k + 13] = sqrt(covariance_mix[num_mix * (k + 6) + k + 6]); // 加表零偏
    }

    timecosts_[3] = timecost.costInMillisecond();
    LOGI << "Calculate covariance costs " << timecosts_[3] << " ms";
}

void Optimizer::updateParametersFromOptimizer(const deque<double> &timelist, const deque<StateData> &statedatalist) {
    // 视觉处理参数
    if (is_use_visual_ && !(visual_map_->keyframes().size() < 2)) {
        // 先更新外参, 更新位姿需要外参
        pose_b_c_.t[0] = extrinsic_b_c_[0];
        pose_b_c_.t[1] = extrinsic_b_c_[1];
        pose_b_c_.t[2] = extrinsic_b_c_[2];

        Quaterniond q_b_c = Quaterniond(extrinsic_b_c_[6], extrinsic_b_c_[3], extrinsic_b_c_[4], extrinsic_b_c_[5]);
        pose_b_c_.R       = q_b_c.toRotationMatrix();

        // 保证系统稳定
        if (optimize_estimate_cam_td_ && (fabs(visual_map_->latestKeyFrame()->timeDelayEst()) > 0.5)) {
            LOGW << "Estimate large td_b_c " << Logging::doubleData(visual_map_->latestKeyFrame()->timeDelayEst())
                 << ", and td_b_c is reset";
        } else {
            extrinsic_b_c_[7] = visual_map_->latestKeyFrame()->timeDelayEst();
        }

        // 更新关键帧的位姿
        for (auto &keyframe : visual_map_->keyframes()) {
            auto &frame = keyframe.second;
            int index   = getStateDataIndex(timelist, frame->stamp());
            if (index < 0) {
                continue;
            }

            IntegrationState state = Preintegration::stateFromData(statedatalist[index]);
            frame->setPose(MISC::stateToPoseWithExtrinsic(state, pose_b_c_));
        }

        // 更新路标点的深度和位置
        for (const auto &landmark : visual_map_->landmarks()) {
            const auto &mappoint = landmark.second;
            if (!mappoint || mappoint->isOutlier()) {
                continue;
            }

            // 未参与优化的无效路标点
            if (visual_mappoints_.find(mappoint->id()) == visual_mappoints_.end()) {
                continue;
            }

            const auto &ref_frame = mappoint->referenceFrame();
            if (!ref_frame) {
                continue;
            }

            // 使用参考帧和最新帧计算路标点深度
            auto farthest_frame = mappoint->farthestKeyFrame();
            if (!farthest_frame) {
                continue;
            }
            auto farthest_feature = farthest_frame->features().find(mappoint->id())->second;

            Pose pose0    = ref_frame->pose();
            Pose pose1    = farthest_frame->pose();
            Vector3d pts0 = camera_->pixel2cam(mappoint->referenceKeypoint());
            Vector3d pts1 = camera_->pixel2cam(farthest_feature->keyPoint());

            Matrix3d R_c1_c0 = pose1.R.transpose() * pose0.R;
            Vector3d t_c1_c0 = pose1.R.transpose() * (pose0.t - pose1.t);

            double norm  = pts1.cross(t_c1_c0).norm();
            double theta = (Rotation::skewSymmetric(pts1) * R_c1_c0 * pts0).norm();
            double depth = norm / theta;

            // 使用优化的深度
            if (is_use_lidar_depth_ && (mappoint->mapPointType() == visual::MAPPOINT_DEPTH_ASSOCIATED)) {
                depth = 1.0 / visual_mappoints_[mappoint->id()];
            }

            auto pc         = depth * camera_->pixel2cam(mappoint->referenceKeypoint());
            mappoint->pos() = camera_->cam2world(pc, mappoint->referenceFrame()->pose());
            mappoint->updateDepth(depth);
        }
    }

    // 激光处理参数
    if (is_use_lidar_) {
        // 先更新外参
        pose_b_l_.t[0] = extrinsic_b_l_[0];
        pose_b_l_.t[1] = extrinsic_b_l_[1];
        pose_b_l_.t[2] = extrinsic_b_l_[2];

        Quaterniond q_b_l = Quaterniond(extrinsic_b_l_[6], extrinsic_b_l_[3], extrinsic_b_l_[4], extrinsic_b_l_[5]);
        pose_b_l_.R       = q_b_l.toRotationMatrix();

        // 保证系统稳定
        if (optimize_estimate_lidar_td_ && (fabs(extrinsic_b_l_[7]) > 0.5)) {
            LOGW << "Estimate large td_b_l " << Logging::doubleData(extrinsic_b_l_[7]) << ", and td_b_l is reset";
            extrinsic_b_l_[7] = 0;
        }

        for (const auto &frame : lidar_map_->keyframes()) {
            // 位姿状态索引
            int index = getStateDataIndex(timelist, frame->stamp());
            if (index < 0) {
                LOGE << "Wrong matched lidar frame at " << Logging::doubleData(frame->stamp());
                continue;
            }

            // 更新位姿
            IntegrationState state = Preintegration::stateFromData(statedatalist[index]);
            frame->setPose(MISC::stateToPoseWithExtrinsic(state, pose_b_l_));

            // 更新世界系点云
            lidar::PointCloudCommon::pointCloudProjection(frame->pose(), frame->pointCloudLidar(),
                                                          frame->pointCloudWorld());
        }
    }
}

bool Optimizer::marginalization(deque<double> &timelist, deque<StateData> &statedatalist,
                                deque<PreBasePtr> &preintegrationlist) {
    TimeCost timecost;

    size_t num_marg = 0;
    vector<ulong> keyframeids;
    if (is_use_visual_) {
        //  按时间先后排序的关键帧
        keyframeids          = visual_map_->orderedKeyFrames();
        auto latest_keyframe = visual_map_->latestKeyFrame();

        latest_keyframe->setKeyFrameState(visual::KEYFRAME_NORMAL);

        // 对齐到保留的最后一个关键帧, 可能移除多个预积分对象
        auto frame = visual_map_->keyframes().find(keyframeids[1])->second;
        num_marg   = static_cast<size_t>(getStateDataIndex(timelist, frame->stamp()));
    } else if (is_use_lidar_) {
        auto frame = lidar_map_->keyframes()[1];
        num_marg   = static_cast<size_t>(getStateDataIndex(timelist, frame->stamp()));
    }
    // 保留的时间节点
    double last_time = timelist[num_marg];

    // 边缘化信息
    std::shared_ptr<MarginalizationInfo> marginalization_info = std::make_shared<MarginalizationInfo>();

    // 指定每个参数块独立的ID, 用于索引参数
    std::unordered_map<long, long> parameters_ids;
    long parameters_id = 0;
    {
        // 边缘化参数
        for (auto &last_marginalization_parameter_block : last_marginalization_parameter_blocks_) {
            parameters_ids[reinterpret_cast<long>(last_marginalization_parameter_block)] = parameters_id++;
        }

        // 外参参数
        if (is_use_visual_) {
            parameters_ids[reinterpret_cast<long>(extrinsic_b_c_)] = parameters_id++;
            for (size_t k = 0; k < keyframeids.size(); k++) {
                auto frame = visual_map_->keyframes().at(keyframeids[k]);
                parameters_ids[reinterpret_cast<long>(&frame->timeDelayEst())] = parameters_id++;
            }
        }
        if (is_use_lidar_) {
            parameters_ids[reinterpret_cast<long>(extrinsic_b_l_)]     = parameters_id++;
            parameters_ids[reinterpret_cast<long>(extrinsic_b_l_ + 7)] = parameters_id++;
        }

        // 位姿参数
        for (const auto &statedata : statedatalist) {
            parameters_ids[reinterpret_cast<long>(statedata.pose)] = parameters_id++;
            parameters_ids[reinterpret_cast<long>(statedata.mix)]  = parameters_id++;
        }

        // 更新参数块的特定ID, 必要的
        marginalization_info->updateParamtersIds(parameters_ids);
    }

    // 边缘化因子
    if (last_marginalization_info_ && last_marginalization_info_->isValid()) {

        vector<int> marginalized_index;
        for (size_t i = 0; i < num_marg; i++) {
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++) {
                if (last_marginalization_parameter_blocks_[k] == statedatalist[i].pose ||
                    last_marginalization_parameter_blocks_[k] == statedatalist[i].mix) {
                    marginalized_index.push_back((int) k);
                }
            }
        }

        if (is_use_visual_) {
            for (size_t k = 0; k < last_marginalization_parameter_blocks_.size(); k++) {
                if (last_marginalization_parameter_blocks_[k] ==
                    &visual_map_->keyframes().at(keyframeids[0])->timeDelayEst()) {
                    marginalized_index.push_back((int) k);
                    break;
                }
            }
        }

        auto factor   = std::make_shared<MarginalizationFactor>(last_marginalization_info_);
        auto residual = std::make_shared<ResidualBlockInfo>(factor, nullptr, last_marginalization_parameter_blocks_,
                                                            marginalized_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // 预积分因子
    for (size_t k = 0; k < num_marg; k++) {
        // 由于会移除多个预积分, 会导致出现保留和移除同时出现, 判断索引以区分
        vector<int> marg_index;
        if (k == (num_marg - 1)) {
            marg_index = {0, 1};
        } else {
            marg_index = {0, 1, 2, 3};
        }

        auto factor = std::make_shared<PreintegrationFactor>(preintegrationlist[k]);
        auto residual =
            std::make_shared<ResidualBlockInfo>(factor, nullptr,
                                                vector<double *>{statedatalist[k].pose, statedatalist[k].mix,
                                                                 statedatalist[k + 1].pose, statedatalist[k + 1].mix},
                                                marg_index);
        marginalization_info->addResidualBlockInfo(residual);
    }

    // 先验约束因子
    if (is_use_prior_) {
        auto pose_factor   = std::make_shared<ImuPosePriorFactor>(pose_prior_, pose_prior_std_);
        auto pose_residual = std::make_shared<ResidualBlockInfo>(
            pose_factor, nullptr, vector<double *>{statedatalist[0].pose}, vector<int>{0});
        marginalization_info->addResidualBlockInfo(pose_residual);

        auto mix_factor   = std::make_shared<ImuMixPriorFactor>(mix_prior_, mix_prior_std_);
        auto mix_residual = std::make_shared<ResidualBlockInfo>(mix_factor, nullptr,
                                                                vector<double *>{statedatalist[0].mix}, vector<int>{0});
        marginalization_info->addResidualBlockInfo(mix_residual);

        // 相机IMU外参先验
        if (is_use_visual_ && optimize_estimate_cam_extrinsic_) {
            auto extrinsic_cam_factor =
                std::make_shared<ImuPosePriorFactor>(extrinsic_cam_prior_, extrinsic_cam_prior_std_);
            auto extrinsic_cam_residual = std::make_shared<ResidualBlockInfo>(
                extrinsic_cam_factor, nullptr, vector<double *>{extrinsic_b_c_}, vector<int>{});
            marginalization_info->addResidualBlockInfo(extrinsic_cam_residual);
        }

        // 雷达IMU外参先验
        if (is_use_lidar_ && optimize_estimate_lidar_extrinsic_) {
            auto extrinsic_lidar_factor =
                std::make_shared<ImuPosePriorFactor>(extrinsic_lidar_prior_, extrinsic_lidar_prior_std_);
            auto extrinsic_lidar_residual = std::make_shared<ResidualBlockInfo>(
                extrinsic_lidar_factor, nullptr, vector<double *>{extrinsic_b_l_}, vector<int>{});
            marginalization_info->addResidualBlockInfo(extrinsic_lidar_residual);
        }

        is_use_prior_ = false;
    }

    if (is_use_visual_) {

        // 重投影因子, 最老的关键帧
        auto frame           = visual_map_->keyframes().at(keyframeids[0]);
        const auto &features = frame->features();

        // 对重投影误差添加loss函数
        auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);

        // MSC因子
        if (is_use_lidar_depth_ && !depth_mappoints_.empty()) {
            for (const auto &mappoint : depth_mappoints_) {
                if (!mappoint || mappoint->isOutlier()) {
                    continue;
                }
                if (visual_mappoints_.find(mappoint->id()) == visual_mappoints_.end()) {
                    continue;
                }

                // 参考帧
                auto ref_frame = mappoint->referenceFrame();
                if (!ref_frame || ref_frame != frame) {
                    continue;
                }
                auto ref_frame_pc   = camera_->pixel2cam(mappoint->referenceKeypoint());
                int ref_frame_index = getStateDataIndex(timelist, ref_frame->stamp());
                if (ref_frame_index < 0) {
                    continue;
                }
                auto ref_feature = ref_frame->features().find(mappoint->id())->second;

                // 需要保存的信息
                int num_frames = 1;

                vector<double *> msc_factor_paras;
                msc_factor_paras.push_back(statedatalist[ref_frame_index].pose);
                msc_factor_paras.push_back(&ref_frame->timeDelayEst());

                vector<std::shared_ptr<ceres::CostFunction>> msc_factor_costs;

                // 所有的观测
                for (const auto &observation : mappoint->observations()) {
                    auto obs_feature = observation.lock();
                    // 无效特征
                    if (!obs_feature || obs_feature->isOutlier()) {
                        continue;
                    }
                    auto obs_frame = obs_feature->getFrame();
                    // 非关键帧特征, 参考帧
                    if (!obs_frame || !obs_frame->isKeyFrame() || (obs_frame == ref_frame)) {
                        continue;
                    }

                    // 观测帧
                    auto obs_frame_pc   = camera_->pixel2cam(obs_feature->keyPoint());
                    int obs_frame_index = getStateDataIndex(timelist, obs_frame->stamp());
                    if (obs_frame_index < 0) {
                        continue;
                    }

                    // 参考帧
                    num_frames++;
                    msc_factor_paras.push_back(statedatalist[obs_frame_index].pose);
                    msc_factor_paras.push_back(&obs_frame->timeDelayEst());

                    auto factor = std::make_shared<ReprojectionFactor>(
                        ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(), obs_feature->velocityInPixel(),
                        ref_frame->timeDelay(), obs_frame->timeDelay(), optimize_reprojection_factor_std_);
                    msc_factor_costs.push_back(factor);
                }
                // 外参
                msc_factor_paras.push_back(extrinsic_b_c_);

                // LiDAR深度因子
                auto factor = std::make_shared<ceres::AutoDiffCostFunction<DepthFactor, 1, 1>>(
                    new DepthFactor(mappoint->lidarDepth(), VISUAL_LANDMARK_LIDAR_DEPTH_STD));
                msc_factor_costs.push_back(factor);

                // 添加MSC因子
                auto msc_factor =
                    std::make_shared<LiDARDepthMscFactor>(msc_factor_costs, num_frames, mappoint->depth());
                auto residual =
                    std::make_shared<ResidualBlockInfo>(msc_factor, loss_function, msc_factor_paras, vector<int>{0, 1});
                marginalization_info->addResidualBlockInfo(residual);
            }
        }

        for (auto const &feature : features) {
            auto mappoint = feature.second->getMapPoint();
            if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier()) {
                continue;
            }

            // LiDAR深度关联的路标点
            if (is_use_lidar_depth_ && (mappoint->mapPointType() == visual::MAPPOINT_DEPTH_ASSOCIATED)) {
                continue;
            }

            // 参考帧
            auto ref_frame = mappoint->referenceFrame();
            // 参考帧非边缘化帧, 不边缘化
            if (ref_frame != frame) {
                continue;
            }

            auto ref_frame_pc   = camera_->pixel2cam(mappoint->referenceKeypoint());
            int ref_frame_index = getStateDataIndex(timelist, ref_frame->stamp());
            if (ref_frame_index < 0) {
                continue;
            }

            // 最新帧观测
            auto farthest_frame = mappoint->farthestKeyFrame();
            if (!farthest_frame) {
                continue;
            }
            auto farthest_feature = farthest_frame->features().find(mappoint->id())->second;
            if (farthest_feature->isOutlier()) {
                continue;
            }
            auto farthest_frame_pc   = camera_->pixel2cam(farthest_feature->keyPoint());
            int farthest_frame_index = getStateDataIndex(timelist, farthest_frame->stamp());
            auto ref_feature         = ref_frame->features().find(mappoint->id())->second;

            // 仅两视图的Pose-Only重投影因子
            auto factor = std::make_shared<PoseOnlyTwoViewReprojectionFactor>(
                ref_frame_pc, farthest_frame_pc, ref_feature->velocityInPixel(), farthest_feature->velocityInPixel(),
                ref_frame->timeDelay(), farthest_frame->timeDelay(), optimize_reprojection_factor_std_);
            auto residual = std::make_shared<ResidualBlockInfo>(
                factor, loss_function,
                vector<double *>{statedatalist[ref_frame_index].pose, statedatalist[farthest_frame_index].pose,
                                 extrinsic_b_c_, &ref_frame->timeDelayEst(), &farthest_frame->timeDelayEst()},
                vector<int>{0, 3});
            marginalization_info->addResidualBlockInfo(residual);

            // 所有的观测
            for (const auto &observation : mappoint->observations()) {
                auto obs_feature = observation.lock();
                // 无效特征
                if (!obs_feature || obs_feature->isOutlier()) {
                    continue;
                }
                auto obs_frame = obs_feature->getFrame();
                // 非关键帧特征, 参考帧
                if (!obs_frame || !obs_frame->isKeyFrame() || (obs_frame == ref_frame) ||
                    (obs_frame == farthest_frame)) {
                    continue;
                }

                // 观测帧
                auto obs_frame_pc   = camera_->pixel2cam(obs_feature->keyPoint());
                int obs_frame_index = getStateDataIndex(timelist, obs_frame->stamp());
                if (obs_frame_index < 0) {
                    continue;
                }

                // Pose-Only重投影因子
                auto factor = std::make_shared<PoseOnlyReprojectionFactor>(
                    ref_frame_pc, obs_frame_pc, farthest_frame_pc, ref_feature->velocityInPixel(),
                    obs_feature->velocityInPixel(), farthest_feature->velocityInPixel(), ref_frame->timeDelay(),
                    obs_frame->timeDelay(), farthest_frame->timeDelay(), optimize_reprojection_factor_std_);
                auto residual = std::make_shared<ResidualBlockInfo>(
                    factor, loss_function,
                    vector<double *>{statedatalist[ref_frame_index].pose, statedatalist[obs_frame_index].pose,
                                     statedatalist[farthest_frame_index].pose, extrinsic_b_c_,
                                     &ref_frame->timeDelayEst(), &obs_frame->timeDelayEst(),
                                     &farthest_frame->timeDelayEst()},
                    vector<int>{0, 4});
                marginalization_info->addResidualBlockInfo(residual);
            }
        }

        if (optimize_estimate_cam_td_) {
            auto frame_pre = visual_map_->keyframes().at(keyframeids[0]);
            auto frame_cur = visual_map_->keyframes().at(keyframeids[1]);
            auto factor    = std::make_shared<TimeDelayFactor>(frame_cur->stamp() - frame_pre->stamp());
            auto residual  = std::make_shared<ResidualBlockInfo>(
                factor, nullptr, vector<double *>{&frame_pre->timeDelayEst(), &frame_cur->timeDelayEst()},
                vector<int>{0});
            marginalization_info->addResidualBlockInfo(residual);
        }
    }

    if (is_use_lidar_) {
        // 可能移除多个雷达节点
        const auto &keyframes        = lidar_map_->keyframes();
        const auto &cur_frame        = keyframes.back();
        size_t frames_size           = keyframes.size();
        const auto &cur_points_lidar = cur_frame->pointCloudLidar()->points;

        // 获取关键帧时间节点索引
        vector<int> frames_index(keyframes.size());
        for (size_t k = 0; k < frames_size; k++) {
            int index       = getStateDataIndex(timelist, keyframes[k]->stamp());
            frames_index[k] = index;
            if (index < 0) {
                LOGE << "Wrong matched lidar frame at " << Logging::doubleData(keyframes[k]->stamp());
                continue;
            }
        }

        auto loss_function = std::make_shared<ceres::HuberLoss>(1.0);
        for (size_t i = 0; i < lidar_map_->keyframes().size() - 1; i++) {
            auto ref_frame = lidar_map_->keyframes()[i];
            if (MISC::isTheSameTimeNode(ref_frame->stamp(), last_time, MISC::MINIMUM_TIME_INTERVAL) ||
                (ref_frame->stamp() > last_time)) {
                break;
            }

            const auto &features = ref_frame->features();
            for (size_t k = 0; k < features.size(); k++) {
                const auto &feature = features[k];
                // 无效特征
                if (!feature || (feature->featureType() == lidar::FEATURE_NONE) || feature->isOutlier()) {
                    continue;
                }

                if (feature->featureType() == lidar::FEATURE_PLANE_BA) {
                    vector<Vector3d> points_lidar;
                    vector<Vector3d> vels;
                    vector<Vector3d> omegas;
                    vector<double> tds;
                    vector<double *> parameter_block;
                    lidar::LidarFeature::Ptr first_feature = nullptr;
                    for (size_t i = 0; i < frames_size - 1; i++) {
                        const auto &frame = keyframes[i];

                        // 无效特征
                        const auto &feature = frame->features()[k];
                        if (!feature || (feature->featureType() != lidar::FEATURE_PLANE_BA) || feature->isOutlier()) {
                            continue;
                        }
                        if (!first_feature) {
                            first_feature = feature;
                        }
                        const auto &nearest_point = feature->nearestPoints().front();

                        // 特征所在帧
                        points_lidar.push_back(nearest_point);
                        vels.push_back(frame->velocity());
                        omegas.push_back(frame->angularVelocity());
                        tds.push_back(frame->timeDelay());
                        int index = frames_index[i];
                        parameter_block.push_back(statedatalist[index].pose);
                    }
                    // 当前帧
                    points_lidar.push_back(cur_points_lidar[k].getVector3fMap().cast<double>());
                    vels.push_back(cur_frame->velocity());
                    omegas.push_back(cur_frame->angularVelocity());
                    tds.push_back(cur_frame->timeDelay());
                    int index = frames_index.back();
                    parameter_block.push_back(statedatalist[index].pose);

                    // 外参和时间偏移
                    parameter_block.push_back(extrinsic_b_l_);
                    parameter_block.push_back(&extrinsic_b_l_[7]);

                    auto factor =
                        std::make_shared<LidarBAFactor>(points_lidar, vels, omegas, tds, first_feature->pointStd());
                    auto residual = std::make_shared<ResidualBlockInfo>(factor, loss_function, parameter_block,
                                                                        std::vector<int>{0});
                    marginalization_info->addResidualBlockInfo(residual);
                }
            }
        }
    }

    // 边缘化处理
    marginalization_info->marginalization();

    // 保留的数据, 使用独立ID
    std::unordered_map<long, double *> address;
    for (size_t k = num_marg; k < statedatalist.size(); k++) {
        address[parameters_ids[reinterpret_cast<long>(statedatalist[k].pose)]] = statedatalist[k].pose;
        address[parameters_ids[reinterpret_cast<long>(statedatalist[k].mix)]]  = statedatalist[k].mix;
    }
    if (is_use_visual_) {
        address[parameters_ids[reinterpret_cast<long>(extrinsic_b_c_)]] = extrinsic_b_c_;
        for (size_t k = 1; k < keyframeids.size(); k++) {
            auto frame = visual_map_->keyframes().at(keyframeids[k]);
            address[parameters_ids[reinterpret_cast<long>(&frame->timeDelayEst())]] = &frame->timeDelayEst();
        }
    }
    if (is_use_lidar_) {
        address[parameters_ids[reinterpret_cast<long>(extrinsic_b_l_)]]     = extrinsic_b_l_;
        address[parameters_ids[reinterpret_cast<long>(extrinsic_b_l_ + 7)]] = &extrinsic_b_l_[7];
    }
    last_marginalization_parameter_blocks_ = marginalization_info->getParamterBlocks(address);
    last_marginalization_info_             = std::move(marginalization_info);

    // 边缘化时间统计至此
    timecosts_[2] = timecost.costInMillisecond();

    // 移除边缘化的数据

    if (is_use_visual_) {
        // 保存移除的路标点
        auto frame           = visual_map_->keyframes().at(keyframeids[0]);
        const auto &features = frame->features();

        fixed_visual_mappoints_.clear();
        for (const auto &feature : features) {
            auto mappoint = feature.second->getMapPoint();
            // 避免加入参考帧非当前帧的路标点
            if (feature.second->isOutlier() || !mappoint || mappoint->isOutlier() ||
                (mappoint->referenceFrame() != frame)) {
                continue;
            }
            fixed_visual_mappoints_.push_back(mappoint->pos());
        }

        // 关键帧
        visual_map_->removeKeyFrame(frame, true);
    }

    if (is_use_lidar_) {
        for (auto frame : lidar_map_->keyframes()) {
            if (MISC::isTheSameTimeNode(frame->stamp(), last_time, MISC::MINIMUM_TIME_INTERVAL) ||
                (frame->stamp() > last_time)) {
                break;
            }

            int index = getStateDataIndex(timelist, frame->stamp());
            if (index >= 0) {
                lidar_map_->removeOldestLidarFrame();
            }
        }
    }

    // 预积分观测及时间状态, 放最后, 需要关键帧索引需要timelist
    for (size_t k = 0; k < num_marg; k++) {
        timelist.pop_front();
        statedatalist.pop_front();
        preintegrationlist.pop_front();
    }

    LOGI << "Marginalize " << num_marg << " states, last time " << Logging::doubleData(last_time) << ", costs "
         << timecosts_[2];

    return true;
}

void Optimizer::addStateParameters(ceres::Problem &problem, deque<StateData> &statedatalist) {
    LOGI << "Total " << statedatalist.size() << " pose states from " << Logging::doubleData(statedatalist.begin()->time)
         << " to " << Logging::doubleData(statedatalist.back().time);

    for (auto &statedata : statedatalist) {
        // 位姿
        ceres::Manifold *manifold = new PoseManifold();
        problem.AddParameterBlock(statedata.pose, Preintegration::numPoseParameter(), manifold);
        problem.AddParameterBlock(statedata.mix, Preintegration::numMixParameter());
    }
}

void Optimizer::visualMappointDepthStd(visual::VisualMapPoint::Ptr mappoint, double &depth, double &depth_std) {
    auto ref_frame        = mappoint->referenceFrame();
    auto farthest_frame   = mappoint->farthestKeyFrame();
    auto farthest_feature = farthest_frame->features().find(mappoint->id())->second;

    // 计算深度
    Matrix3d R_w_c0 = ref_frame->pose().R;
    Vector3d t_w_c0 = ref_frame->pose().t;
    Matrix3d R_w_c2 = farthest_frame->pose().R;
    Vector3d t_w_c2 = farthest_frame->pose().t;
    Vector3d pts_0  = camera_->pixel2cam(mappoint->referenceKeypoint());
    Vector3d pts_2  = camera_->pixel2cam(farthest_feature->keyPoint());

    Matrix3d R_c2_c0   = R_w_c2.transpose() * R_w_c0;
    Vector3d t_c2_c0   = R_w_c2.transpose() * (t_w_c0 - t_w_c2);
    Vector3d pts_2_0   = R_c2_c0 * pts_0;
    Matrix3d skew_p2   = Rotation::skewSymmetric(pts_2);
    Vector3d norm_vec  = skew_p2 * t_c2_c0;
    Vector3d theta_vec = skew_p2 * pts_2_0;

    double norm          = (skew_p2 * t_c2_c0).norm();
    double theta         = (theta_vec).norm();
    double norm_inverse  = 1.0 / norm;
    double theta_inverse = 1.0 / theta;
    depth                = norm * theta_inverse;

    // 假定位姿准确计算深度的不确定度
    double norm_theta_inverse2 = -norm / (theta * theta);
    Vector3d jaco_norm_p2 =
        norm_inverse * (-Rotation::skewSymmetric(t_c2_c0).transpose() * norm_vec).transpose() * theta_inverse;
    Vector3d jaco_theta_p0 =
        theta_inverse * ((skew_p2 * R_c2_c0).transpose() * theta_vec).transpose() * norm_theta_inverse2;
    Vector3d jaco_theta_p2 =
        theta_inverse * (-Rotation::skewSymmetric(pts_2_0).transpose() * theta_vec).transpose() * norm_theta_inverse2;

    Vector2d jaco_p0 = jaco_theta_p0.topRows(2);
    Vector2d jaco_p2 = (jaco_norm_p2 + jaco_theta_p2).topRows(2);
    Eigen::Matrix2d cov_norm_pixel =
        Eigen::Matrix2d::Identity() * optimize_reprojection_factor_std_ * optimize_reprojection_factor_std_;
    double depth_cov = (jaco_p0.transpose() * cov_norm_pixel * jaco_p0).value() +
                       (jaco_p2.transpose() * cov_norm_pixel * jaco_p2).value();

    depth_std = sqrt(depth_cov);
}

void Optimizer::addVisualParameters(ceres::Problem &problem) {
    if (visual_map_->keyframes().size() < 2) {
        return;
    }

    // 逆深度
    visual_mappoints_.clear();
    depth_mappoints_.clear();
    for (const auto &landmark : visual_map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier() || (mappoint->usedTimes() < 2)) {
            continue;
        }

        // 是否为地图中的关键帧路标点
        auto ref_frame = mappoint->referenceFrame();
        if (!ref_frame) {
            continue;
        }

        // 参考帧
        Pose pose0    = ref_frame->pose();
        Vector3d pts0 = camera_->pixel2cam(mappoint->referenceKeypoint());

        // 计算视差最大的关键帧
        double max_theta                        = 0;
        visual::VisualFrame::Ptr farthest_frame = nullptr;
        for (const auto &observation : mappoint->observations()) {
            auto obs_feature = observation.lock();
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            if (!obs_frame || !obs_frame->isKeyFrame() || (obs_frame == ref_frame)) {
                continue;
            }

            // 观测帧
            Pose pose1    = obs_frame->pose();
            Vector3d pts1 = camera_->pixel2cam(obs_feature->keyPoint());

            // 计算视差
            Matrix3d R_c1_c0 = pose1.R.transpose() * pose0.R;
            double theta     = (Rotation::skewSymmetric(pts1) * R_c1_c0 * pts0).norm();
            if (theta > max_theta) {
                farthest_frame = obs_frame;
                max_theta      = theta;
            }
        }

        // 设置视差最大的关键帧
        if (!farthest_frame) {
            continue;
        }
        mappoint->setFarthestFrame(farthest_frame);

        if (visual_mappoints_.find(mappoint->id()) == visual_mappoints_.end()) {
            // 计算路标点深度STD
            double depth, depth_std;
            visualMappointDepthStd(mappoint, depth, depth_std);

            if (mappoint->mapPointType() == visual::MAPPOINT_DEPTH_ASSOCIATED) {
                // 判断激光深度有效性

                // 深度先验和激光深度检查
                double delta = fabs(depth - mappoint->lidarDepth());
                double delta_std =
                    sqrt(depth_std * depth_std + VISUAL_LANDMARK_LIDAR_DEPTH_STD * VISUAL_LANDMARK_LIDAR_DEPTH_STD);
                if (delta > 3.0 * delta_std) {
                    mappoint->mapPointType() = visual::MAPPOINT_TRIANGULATED;
                    mappoint->updateDepth(depth);
                } else {
                    depth_mappoints_.push_back(mappoint);
                }
            }

            // 加入当前路标点
            visual_mappoints_[mappoint->id()] = 1.0 / mappoint->depth();

            // 增加优化次数计数
            mappoint->addOptimizedTimes();
        }
    }
    // LOGE << "Find " << extrinsic_mappoints_.size() << " mappoints for extrinsic estimation.";

    // 外参参数化
    ceres::Manifold *manifold = new PoseManifold();
    problem.AddParameterBlock(extrinsic_b_c_, 7, manifold);

    // 固定外参
    if (!optimize_estimate_cam_extrinsic_ || !visual_map_->isWindowFull()) {
        problem.SetParameterBlockConstant(extrinsic_b_c_);
    }

    // 时间延时
    if (!optimize_estimate_cam_td_ || !visual_map_->isWindowFull()) {
        for (auto frame : visual_map_->keyframes()) {
            problem.AddParameterBlock(&frame.second->timeDelayEst(), 1);
            problem.SetParameterBlockConstant(&frame.second->timeDelayEst());
        }
    }
}

void Optimizer::addLidarParameters(ceres::Problem &problem) {
    // 外参参数化
    ceres::Manifold *manifold = new PoseManifold();
    problem.AddParameterBlock(extrinsic_b_l_, 7, manifold);
    if (!optimize_estimate_lidar_extrinsic_ || !lidar_map_->isWindowFull()) {
        problem.SetParameterBlockConstant(extrinsic_b_l_);
    }

    // 时间偏移
    problem.AddParameterBlock(&extrinsic_b_l_[7], 1);
    if (!optimize_estimate_lidar_td_ || !lidar_map_->isWindowFull()) {
        problem.SetParameterBlockConstant(&extrinsic_b_l_[7]);
    } else {
        if (lidar_map_->keyframes().back()->velocity().norm() < MINIMUM_SPEED_ESTIMATE_LIDAR_TD) {
            problem.SetParameterBlockConstant(&extrinsic_b_l_[7]);
        }
    }
}

void Optimizer::addDepthMappointMscFactors(ceres::Problem &problem, bool isusekernel, const deque<double> &timelist,
                                           deque<StateData> &statedatalist) {
    auto loss_function = new ceres::HuberLoss(1.0);
    for (const auto &mappoint : depth_mappoints_) {
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }
        if (visual_mappoints_.find(mappoint->id()) == visual_mappoints_.end()) {
            continue;
        }

        // 参考帧
        auto ref_frame = mappoint->referenceFrame();
        if (!ref_frame) {
            continue;
        }
        auto ref_frame_pc   = camera_->pixel2cam(mappoint->referenceKeypoint());
        int ref_frame_index = getStateDataIndex(timelist, ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }
        auto ref_feature = ref_frame->features().find(mappoint->id())->second;

        // 需要保存的信息
        int num_frames = 1;

        vector<double *> msc_factor_paras;
        msc_factor_paras.push_back(statedatalist[ref_frame_index].pose);
        msc_factor_paras.push_back(&ref_frame->timeDelayEst());

        vector<std::shared_ptr<ceres::CostFunction>> msc_factor_costs;

        // 所有的观测
        for (const auto &observation : mappoint->observations()) {
            auto obs_feature = observation.lock();
            // 无效特征
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            // 非关键帧特征, 参考帧
            if (!obs_frame || !obs_frame->isKeyFrame() || (obs_frame == ref_frame)) {
                continue;
            }

            // 观测帧
            auto obs_frame_pc   = camera_->pixel2cam(obs_feature->keyPoint());
            int obs_frame_index = getStateDataIndex(timelist, obs_frame->stamp());
            if (obs_frame_index < 0) {
                continue;
            }

            // 参考帧
            num_frames++;
            msc_factor_paras.push_back(statedatalist[obs_frame_index].pose);
            msc_factor_paras.push_back(&obs_frame->timeDelayEst());

            auto factor = std::make_shared<ReprojectionFactor>(
                ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(), obs_feature->velocityInPixel(),
                ref_frame->timeDelay(), obs_frame->timeDelay(), optimize_reprojection_factor_std_);
            msc_factor_costs.push_back(factor);
        }
        // 外参
        msc_factor_paras.push_back(extrinsic_b_c_);

        // LiDAR深度因子
        auto factor = std::make_shared<ceres::AutoDiffCostFunction<DepthFactor, 1, 1>>(
            new DepthFactor(mappoint->lidarDepth(), VISUAL_LANDMARK_LIDAR_DEPTH_STD));
        msc_factor_costs.push_back(factor);

        // 添加MSC因子
        auto msc_factor = new LiDARDepthMscFactor(msc_factor_costs, num_frames, mappoint->depth());
        problem.AddResidualBlock(msc_factor, loss_function, msc_factor_paras);
    }
}

void Optimizer::depthOnlyOptimization(const deque<double> &timelist, deque<StateData> &statedatalist) {
    TimeCost timecost;

    ceres::Problem problem;
    ceres::Solver solver;
    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.linear_solver_type         = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations         = 2;
    options.num_threads                = 8;

    // 参数信息

    // 位姿
    for (auto &statedata : statedatalist) {
        ceres::Manifold *manifold = new PoseManifold();
        problem.AddParameterBlock(statedata.pose, Preintegration::numPoseParameter(), manifold);
        problem.SetParameterBlockConstant(statedata.pose);
    }

    // 外参和时间偏移
    ceres::Manifold *manifold = new PoseManifold();
    problem.AddParameterBlock(extrinsic_b_c_, 7, manifold);
    problem.SetParameterBlockConstant(extrinsic_b_c_);
    for (auto frame : visual_map_->keyframes()) {
        problem.AddParameterBlock(&frame.second->timeDelayEst(), 1);
        problem.SetParameterBlockConstant(&frame.second->timeDelayEst());
    }

    // 逆深度数据
    for (auto const &mappoint : depth_mappoints_) {
        double *inverse_depth = &visual_mappoints_[mappoint->id()];
        problem.AddParameterBlock(inverse_depth, 1);
    }

    auto loss_function = new ceres::HuberLoss(1.0);
    for (const auto &mappoint : depth_mappoints_) {
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }
        if (visual_mappoints_.find(mappoint->id()) == visual_mappoints_.end()) {
            continue;
        }

        // 参考帧
        auto ref_frame = mappoint->referenceFrame();
        if (!ref_frame) {
            continue;
        }
        auto ref_frame_pc   = camera_->pixel2cam(mappoint->referenceKeypoint());
        int ref_frame_index = getStateDataIndex(timelist, ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }
        auto ref_feature = ref_frame->features().find(mappoint->id())->second;

        // 逆深度
        double *inverse_depth = &visual_mappoints_[mappoint->id()];

        auto factor = DepthFactor::create(mappoint->lidarDepth(), VISUAL_LANDMARK_LIDAR_DEPTH_STD);
        problem.AddResidualBlock(factor, loss_function, inverse_depth);

        // 所有的观测
        for (const auto &observation : mappoint->observations()) {
            auto obs_feature = observation.lock();
            // 无效特征
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            // 非关键帧特征, 参考帧
            if (!obs_frame || !obs_frame->isKeyFrame() || (obs_frame == ref_frame)) {
                continue;
            }

            // 观测帧
            auto obs_frame_pc   = camera_->pixel2cam(obs_feature->keyPoint());
            int obs_frame_index = getStateDataIndex(timelist, obs_frame->stamp());
            if (obs_frame_index < 0) {
                continue;
            }

            auto factor = new ReprojectionFactor(ref_frame_pc, obs_frame_pc, ref_feature->velocityInPixel(),
                                                 obs_feature->velocityInPixel(), ref_frame->timeDelay(),
                                                 obs_frame->timeDelay(), optimize_reprojection_factor_std_);
            problem.AddResidualBlock(factor, loss_function, statedatalist[ref_frame_index].pose,
                                     statedatalist[obs_frame_index].pose, extrinsic_b_c_, inverse_depth,
                                     &ref_frame->timeDelayEst(), &obs_frame->timeDelayEst());
        }
    }

    solver.Solve(options, &problem, &summary);
    // timecosts_[3] = timecost.costInMillisecond();
    // LOGI << summary.BriefReport();
    LOGI << timecost.costInMillisecond("Depth only optimization");
}

Residuals Optimizer::addReprojectionFactors(ceres::Problem &problem, bool isusekernel, const deque<double> &timelist,
                                            deque<StateData> &statedatalist) {
    Residuals residual_ids;

    if (visual_map_->keyframes().size() < 2) {
        return residual_ids;
    }

    // 鲁棒核函数
    ceres::LossFunction *loss_function = nullptr;
    if (isusekernel) {
        loss_function = new ceres::HuberLoss(1.0);
    }

    if (is_use_lidar_depth_ && !depth_mappoints_.empty()) {
        // 带LiDAR深度的路标点添加MSC因子
        addDepthMappointMscFactors(problem, isusekernel, timelist, statedatalist);
    }

    // 所有路标点的观测
    residual_ids.clear();
    for (const auto &landmark : visual_map_->landmarks()) {
        const auto &mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        if (visual_mappoints_.find(mappoint->id()) == visual_mappoints_.end()) {
            continue;
        }

        // LiDAR深度关联的路标点
        if (is_use_lidar_depth_ && (mappoint->mapPointType() == visual::MAPPOINT_DEPTH_ASSOCIATED)) {
            continue;
        }

        // 参考帧
        auto ref_frame = mappoint->referenceFrame();
        if (!ref_frame) {
            continue;
        }

        auto ref_frame_pc   = camera_->pixel2cam(mappoint->referenceKeypoint());
        int ref_frame_index = getStateDataIndex(timelist, ref_frame->stamp());
        if (ref_frame_index < 0) {
            continue;
        }

        // 最新帧观测
        auto farthest_frame = mappoint->farthestKeyFrame();
        if (!farthest_frame) {
            continue;
        }
        auto farthest_feature = farthest_frame->features().find(mappoint->id())->second;
        if (farthest_feature->isOutlier()) {
            continue;
        }
        auto farthest_frame_pc   = camera_->pixel2cam(farthest_feature->keyPoint());
        int farthest_frame_index = getStateDataIndex(timelist, farthest_frame->stamp());
        auto ref_feature         = ref_frame->features().find(mappoint->id())->second;

        // 仅两视图的Pose-Only重投影因子
        auto factor = new PoseOnlyTwoViewReprojectionFactor(
            ref_frame_pc, farthest_frame_pc, ref_feature->velocityInPixel(), farthest_feature->velocityInPixel(),
            ref_frame->timeDelay(), farthest_frame->timeDelay(), optimize_reprojection_factor_std_);

        auto residual_block_id = problem.AddResidualBlock(factor, loss_function, statedatalist[ref_frame_index].pose,
                                                          statedatalist[farthest_frame_index].pose, extrinsic_b_c_,
                                                          &ref_frame->timeDelayEst(), &farthest_frame->timeDelayEst());
        residual_ids.push_back(residual_block_id);

        // 所有的观测
        for (const auto &observation : mappoint->observations()) {
            auto obs_feature = observation.lock();
            // 无效特征
            if (!obs_feature || obs_feature->isOutlier()) {
                continue;
            }
            auto obs_frame = obs_feature->getFrame();
            // 非关键帧特征, 参考帧
            if (!obs_frame || !obs_frame->isKeyFrame() || (obs_frame == ref_frame) || (obs_frame == farthest_frame)) {
                continue;
            }

            // 观测帧
            auto obs_frame_pc   = camera_->pixel2cam(obs_feature->keyPoint());
            int obs_frame_index = getStateDataIndex(timelist, obs_frame->stamp());
            if (obs_frame_index < 0) {
                continue;
            }

            // Pose-Only重投影因子
            auto factor = new PoseOnlyReprojectionFactor(
                ref_frame_pc, obs_frame_pc, farthest_frame_pc, ref_feature->velocityInPixel(),
                obs_feature->velocityInPixel(), farthest_feature->velocityInPixel(), ref_frame->timeDelay(),
                obs_frame->timeDelay(), farthest_frame->timeDelay(), optimize_reprojection_factor_std_);

            auto residual_block_id = problem.AddResidualBlock(
                factor, loss_function, statedatalist[ref_frame_index].pose, statedatalist[obs_frame_index].pose,
                statedatalist[farthest_frame_index].pose, extrinsic_b_c_, &ref_frame->timeDelayEst(),
                &obs_frame->timeDelayEst(), &farthest_frame->timeDelayEst());
            residual_ids.push_back(residual_block_id);
        }
    }

    if (is_use_prior_ && optimize_estimate_cam_extrinsic_) {
        // 对外参添加先验
        auto pose_factor = new ImuPosePriorFactor(extrinsic_cam_prior_, extrinsic_cam_prior_std_);
        problem.AddResidualBlock(pose_factor, nullptr, extrinsic_b_c_);
    }

    if (optimize_estimate_cam_td_) {
        auto keyframeids = visual_map_->orderedKeyFrames();
        for (size_t k = 0; k < keyframeids.size() - 1; k++) {
            auto frame_pre = visual_map_->keyframes().at(keyframeids[k]);
            auto frame_cur = visual_map_->keyframes().at(keyframeids[k + 1]);
            auto factor    = new TimeDelayFactor(frame_cur->stamp() - frame_pre->stamp());
            problem.AddResidualBlock(factor, nullptr, &frame_pre->timeDelayEst(), &frame_cur->timeDelayEst());
        }
    }

    return residual_ids;
}

LidarResiduals Optimizer::addLidarFactors(ceres::Problem &problem, bool isusekernel, const deque<double> &timelist,
                                          deque<StateData> &statedatalist) {
    LidarResiduals residual_block;
    residual_block.clear();

    ceres::LossFunction *loss_function = nullptr;
    if (isusekernel) {
        loss_function = new ceres::HuberLoss(1.0);
    }

    const auto &keyframes        = lidar_map_->keyframes();
    const auto &cur_frame        = keyframes.back();
    size_t frames_size           = keyframes.size();
    size_t points_size           = cur_frame->pointCloudLidar()->size();
    const auto &cur_points_lidar = cur_frame->pointCloudLidar()->points;
    const auto &features_type    = cur_frame->featuresType();

    // 获取关键帧时间节点索引
    vector<int> frames_index(frames_size);
    for (size_t k = 0; k < frames_size; k++) {
        int index       = getStateDataIndex(timelist, keyframes[k]->stamp());
        frames_index[k] = index;
        if (index < 0) {
            continue;
        }
    }

    for (size_t k = 0; k < points_size; k++) {
        // 初始关键帧不足以进行BA时使用帧间点到面约束
        if (frames_size < static_cast<size_t>(lidar::LidarMap::MINIMUM_PLANE_BA_FEATURES)) {
            for (size_t i = 0; i < frames_size - 1; i++) {
                // 位姿状态索引
                int ref_index = frames_index[i];
                int cur_index = frames_index.back();

                const auto &ref_frame = keyframes[i];
                const auto &feature   = ref_frame->features()[k];
                if (!feature || (feature->featureType() == lidar::FEATURE_NONE) || feature->isOutlier()) {
                    continue;
                }

                auto factor = new LidarFactor(
                    cur_points_lidar[k], feature->normalVector(), feature->center(), optimize_point_to_plane_std_,
                    ref_frame->velocity(), ref_frame->angularVelocity(), ref_frame->timeDelay(), cur_frame->velocity(),
                    cur_frame->angularVelocity(), cur_frame->timeDelay());
                auto residual_block_id =
                    problem.AddResidualBlock(factor, loss_function, statedatalist[ref_index].pose,
                                             statedatalist[cur_index].pose, extrinsic_b_l_, &extrinsic_b_l_[7]);
                residual_block.emplace_back(residual_block_id, feature);
            }
            continue;
        }

        // 平面BA约束
        if (!features_type.empty() && (features_type[k] == lidar::FEATURE_PLANE_BA)) {
            vector<Vector3d> points_lidar;
            vector<Vector3d> vels;
            vector<Vector3d> omegas;
            vector<double> tds;
            vector<double *> parameter_block;
            lidar::LidarFeature::Ptr first_feature = nullptr;
            for (size_t i = 0; i < frames_size - 1; i++) {
                const auto &frame = keyframes[i];

                // 无效特征
                const auto &feature = frame->features()[k];
                if (!feature || (feature->featureType() != lidar::FEATURE_PLANE_BA) || feature->isOutlier()) {
                    continue;
                }
                if (!first_feature) {
                    first_feature = feature;
                }
                const auto &nearest_point = feature->nearestPoints().front();

                // 特征所在帧
                points_lidar.push_back(nearest_point);
                vels.push_back(frame->velocity());
                omegas.push_back(frame->angularVelocity());
                tds.push_back(frame->timeDelay());
                int index = frames_index[i];
                parameter_block.push_back(statedatalist[index].pose);
            }
            // 当前帧
            points_lidar.push_back(cur_points_lidar[k].getVector3fMap().cast<double>());
            vels.push_back(cur_frame->velocity());
            omegas.push_back(cur_frame->angularVelocity());
            tds.push_back(cur_frame->timeDelay());
            int index = frames_index.back();
            parameter_block.push_back(statedatalist[index].pose);

            // 外参和时间偏移
            parameter_block.push_back(extrinsic_b_l_);
            parameter_block.push_back(&extrinsic_b_l_[7]);

            auto factor            = new LidarBAFactor(points_lidar, vels, omegas, tds, first_feature->pointStd());
            auto residual_block_id = problem.AddResidualBlock(factor, loss_function, parameter_block);
            residual_block.emplace_back(residual_block_id, first_feature);
        }
    }

    if (is_use_frame_to_map_) {
        vector<lidar::LidarFrame::Ptr> frames;
        auto last_frame = lidar_map_->keyframes().front();
        auto cur_frame  = lidar_map_->keyframes().back();
        int last_index  = frames_index.front();
        int cur_index   = frames_index.back();

        // 拷贝先验位姿
        if (last_frame->globalMapFeatures().size()) {
            memcpy(last_frame->frameToMapPose(), last_frame->pose().t.data(), sizeof(double) * 3);
            Quaterniond q = Quaterniond(last_frame->pose().R);
            memcpy(last_frame->frameToMapPose() + 3, q.coeffs().data(), sizeof(double) * 4);

            frames.push_back(last_frame);
        }
        if (cur_frame->globalMapFeatures().size()) {
            memcpy(cur_frame->frameToMapPose(), cur_frame->pose().t.data(), sizeof(double) * 3);
            Quaterniond q = Quaterniond(cur_frame->pose().R);
            memcpy(cur_frame->frameToMapPose() + 3, q.coeffs().data(), sizeof(double) * 4);
            frames.push_back(cur_frame);
        }

        // 位姿优化
        TimeCost timecost;
        tbb::parallel_for(size_t(0), frames.size(), [&](const size_t &k) { lidarFrameToMapOptimization(frames[k]); });
        timecosts_[3] = timecost.costInMillisecond();

        // 添加位姿约束
        if (last_frame->globalMapFeatures().size()) {
            auto factor = new LiDARPoseFactor(last_frame->frameToMapPose(), last_frame->frameToMapPoseStd());
            problem.AddResidualBlock(factor, nullptr, statedatalist[last_index].pose, extrinsic_b_l_);
        }
        if (cur_frame->globalMapFeatures().size()) {
            auto factor = new LiDARPoseFactor(cur_frame->frameToMapPose(), cur_frame->frameToMapPoseStd());
            problem.AddResidualBlock(factor, nullptr, statedatalist[cur_index].pose, extrinsic_b_l_);
        }
    }

    if (is_use_prior_ && optimize_estimate_lidar_extrinsic_) {
        // 对外参添加先验
        auto pose_factor = new ImuPosePriorFactor(extrinsic_lidar_prior_, extrinsic_lidar_prior_std_);
        problem.AddResidualBlock(pose_factor, nullptr, extrinsic_b_l_);
    }

    return residual_block;
}

void Optimizer::addImuFactors(ceres::Problem &problem, deque<StateData> &statedatalist,
                              deque<PreBasePtr> &preintegrationlist) {
    for (size_t k = 0; k < preintegrationlist.size(); k++) {
        // 预积分因子
        auto factor = new PreintegrationFactor(preintegrationlist[k]);
        problem.AddResidualBlock(factor, nullptr, statedatalist[k].pose, statedatalist[k].mix,
                                 statedatalist[k + 1].pose, statedatalist[k + 1].mix);
    }

    // 添加IMU误差约束, 限制过大的误差估计
    auto factor = new ImuErrorFactor();
    problem.AddResidualBlock(factor, nullptr, statedatalist.back().mix);

    // IMU初始先验因子, 仅限于初始化
    if (is_use_prior_) {
        auto pose_factor = new ImuPosePriorFactor(pose_prior_, pose_prior_std_);
        problem.AddResidualBlock(pose_factor, nullptr, statedatalist[0].pose);

        auto mix_factor = new ImuMixPriorFactor(mix_prior_, mix_prior_std_);
        problem.AddResidualBlock(mix_factor, nullptr, statedatalist[0].mix);
    }

    // 零速约束
    if (is_use_zupt_) {
        vector<double> average;
        if (MISC::detectZeroVelocity(preintegrationlist.back()->imuBuffer(), imudatarate_, average)) {
            LOGI << "Zero velocity is detected at " << Logging::doubleData(statedatalist.back().time);
            auto factor = new ImuZeroVelocityFactor(ZERO_VELOCITY_STD);
            problem.AddResidualBlock(factor, nullptr, statedatalist.back().mix);
        }
    }
}

ResidualOutliers Optimizer::visualOutlierCullingByChi2(ceres::Problem &problem, vector<ResidualId> &residual_ids) {
    static double chi2_threshold2 = 5.991; // 0.05
    static double chi2_threshold1 = 3.841; // 0.05

    size_t residual_size = residual_ids.size();
    ResidualOutliers outlier_ids(residual_size);

    tbb::parallel_for(size_t(0), residual_size, [&](const size_t &k) {
        const auto &residual_id = residual_ids[k];

        // 获取参数块和CostFunction
        vector<double *> parameter_blocks;
        problem.GetParameterBlocksForResidualBlock(residual_id, &parameter_blocks);
        const ceres::CostFunction *cost_function = problem.GetCostFunctionForResidualBlock(residual_id);

        // 重投影残差
        bool is_outlier = false;
        double cost;
        double threshold;
        if (cost_function->num_residuals() == 2) {
            // 计算残差
            Vector2d residual;
            cost_function->Evaluate(parameter_blocks.data(), residual.data(), nullptr);

            cost      = residual.transpose() * residual;
            threshold = chi2_threshold2;
        } else {
            // 计算残差
            double residual;
            cost_function->Evaluate(parameter_blocks.data(), &residual, nullptr);

            cost      = residual * residual;
            threshold = chi2_threshold1;
        }

        if (cost > threshold) {
            is_outlier = true;
        }
        outlier_ids[k] = std::make_pair(residual_id, is_outlier);
    });

    return outlier_ids;
}

ResidualOutliers Optimizer::lidarOutlierCullingByChi2(ceres::Problem &problem, LidarResiduals &residual_ids) {
    static double chi2_threshold = 3.841; // 0.05

    size_t residual_size = residual_ids.size();
    ResidualOutliers outlier_ids(residual_size);

    tbb::parallel_for(size_t(0), residual_size, [&](const size_t &k) {
        const auto &residual_id = residual_ids[k].first;
        const auto &feature     = residual_ids[k].second;

        // 获取参数块和CostFunction
        vector<double *> parameter_blocks;
        problem.GetParameterBlocksForResidualBlock(residual_id, &parameter_blocks);
        const ceres::CostFunction *cost_function = problem.GetCostFunctionForResidualBlock(residual_id);

        // 计算残差
        double residual;
        cost_function->Evaluate(parameter_blocks.data(), &residual, nullptr);

        // 判断粗差
        bool is_outlier = false;
        double cost     = residual * residual;
        if (cost > chi2_threshold) {
            feature->setOutlier();
            is_outlier = true;
        }
        outlier_ids[k] = std::make_pair(residual_id, is_outlier);
    });

    return outlier_ids;
}

bool Optimizer::visualLandmarkOutlierCulling() {
    if (visual_map_->keyframes().size() < 2) {
        return false;
    }

    // 移除非关键帧中的路标点, 不能在遍历中直接移除, 否则破坏了遍历
    vector<visual::VisualMapPoint::Ptr> mappoints;
    int num_outliers_mappoint = 0;
    int num_outliers_feature  = 0;
    int num1 = 0, num2 = 0, num3 = 0;
    for (auto &landmark : visual_map_->landmarks()) {
        auto mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        // 未参与优化的无效路标点
        if (visual_mappoints_.find(mappoint->id()) == visual_mappoints_.end()) {
            continue;
        }

        // 路标点在滑动窗口内的所有观测
        vector<double> reprojection_errors;
        for (const auto &observation : mappoint->observations()) {
            auto feat = observation.lock();
            // 无效特征
            if (!feat || feat->isOutlier()) {
                continue;
            }
            auto frame = feat->getFrame();
            // 非关键帧特征
            if (!frame || !frame->isKeyFrame()) {
                continue;
            }

            auto pp = feat->keyPoint();

            // 计算重投影误差
            double error = camera_->reprojectionError(frame->pose(), mappoint->pos(), pp).norm();

            // 大于3倍阈值, 则禁用当前观测
            if (!visual::VisualMapPoint::isGoodToTrack(camera_, pp, frame->pose(), mappoint->pos(),
                                                       3.0 * optimize_reprojection_std_, 1.0)) {
                feat->setOutlier(true);
                mappoint->decreaseUsedTimes();

                // 如果当前观测帧是路标点的参考帧, 直接设置为outlier
                if (frame == mappoint->referenceFrame()) {
                    mappoint->setOutlier(true);
                    mappoints.push_back(mappoint);
                    num_outliers_mappoint++;
                    num1++;
                    break;
                }
                num_outliers_feature++;
            } else {
                reprojection_errors.push_back(error);
            }
        }

        // 有效观测不足, 平均重投影误差较大, 则为粗差
        if (reprojection_errors.size() < 2) {
            mappoint->setOutlier(true);
            mappoints.push_back(mappoint);
            num_outliers_mappoint++;
            num2++;
        } else {
            double avg_error = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) /
                               static_cast<double>(reprojection_errors.size());
            if (avg_error > optimize_reprojection_std_) {
                mappoint->setOutlier(true);
                mappoints.push_back(mappoint);
                num_outliers_mappoint++;
                num3++;
            }
        }
    }

    // 移除outliers
    for (auto &mappoint : mappoints) {
        visual_map_->removeMappoint(mappoint);
    }

    LOGI << "Culled " << num_outliers_mappoint << " visual mappoint with " << num_outliers_feature
         << " bad observed features " << num1 << ", " << num2 << ", " << num3;

    outliers_[0] = num_outliers_mappoint;
    outliers_[1] = num_outliers_feature;

    return true;
}

void Optimizer::doReintegration(deque<PreBasePtr> &preintegrationlist, deque<StateData> &statedatalist) {
    int cnt = 0;
    for (size_t k = 0; k < preintegrationlist.size(); k++) {
        IntegrationState state = Preintegration::stateFromData(statedatalist[k]);
        Vector3d dbg           = preintegrationlist[k]->deltaState().bg - state.bg;
        Vector3d dba           = preintegrationlist[k]->deltaState().ba - state.ba;
        if ((dbg.norm() > 6 * integration_parameters_->gyr_bias_std) ||
            (dba.norm() > 6 * integration_parameters_->acc_bias_std)) {
            preintegrationlist[k]->reintegration(state);
            cnt++;
        }
    }
    if (cnt) {
        LOGW << "Reintegration " << cnt << " preintegration";
    }
}

int Optimizer::getStateDataIndex(const deque<double> &timelist, double time) {

    size_t index = MISC::getStateDataIndex(timelist, time, MISC::MINIMUM_TIME_INTERVAL);
    if (!MISC::isTheSameTimeNode(timelist[index], time, MISC::MINIMUM_TIME_INTERVAL)) {
        LOGW << "Wrong matching time node " << Logging::doubleData(timelist[index]) << " to "
             << Logging::doubleData(time);
        return -1;
    }
    return static_cast<int>(index);
}

void Optimizer::constructPrior(const StateData &statedata, bool is_zero_velocity) {
    // 初始先验
    double pos_prior_std = 0.001;                                      // 局部系原点为定义值
    double att_prior_std = 0.5 * D2R;                                  // 0.5 deg, 水平姿态角可能估计不准确
    double vel_prior_std = 0.1;                                        // 0.1 m/s
    double bg_prior_std  = integration_parameters_->gyr_bias_std * 3;  // Bias std * 3
    double ba_prior_std  = ab_prior_std_;

    if (!is_zero_velocity) {
        // 运动中调平受加速度污染
        att_prior_std = 1.5 * D2R; // 1.5 deg
        bg_prior_std  = gb_prior_std_;
        vel_prior_std = 0.5;
    }

    memcpy(pose_prior_, statedata.pose, sizeof(double) * 7);
    memcpy(mix_prior_, statedata.mix, sizeof(double) * 9);
    for (size_t k = 0; k < 3; k++) {
        pose_prior_std_[k + 0] = pos_prior_std;
        pose_prior_std_[k + 3] = att_prior_std;

        mix_prior_std_[k + 0] = vel_prior_std;
        mix_prior_std_[k + 3] = bg_prior_std;
        mix_prior_std_[k + 6] = ba_prior_std;
    }
    // 位置和航向是局部坐标系的定义值
    pose_prior_std_[0] = 0.001;
    pose_prior_std_[1] = 0.001;
    pose_prior_std_[2] = 0.001;
    pose_prior_std_[5] = 0.001 * D2R;

    if (is_use_visual_ && optimize_estimate_cam_extrinsic_) {
        // 外参先验
        if (optimize_cam_extrinsic_accurate_) {
            pos_prior_std = 0.05;      // 0.05 m
            att_prior_std = 0.1 * D2R; // 0.1 deg
        } else {
            pos_prior_std = 0.1;       // 0.1 m
            att_prior_std = 1.0 * D2R; // 1.0 deg
        }

        memcpy(extrinsic_cam_prior_, extrinsic_b_c_, sizeof(double) * 7);
        for (size_t k = 0; k < 3; k++) {
            extrinsic_cam_prior_std_[k + 0] = pos_prior_std;
            extrinsic_cam_prior_std_[k + 3] = att_prior_std;
        }
    }

    if (is_use_lidar_ && optimize_estimate_lidar_extrinsic_) {
        // 外参先验
        if (optimize_lidar_extrinsic_accurate_) {
            pos_prior_std = 0.05;      // 0.05 m
            att_prior_std = 0.1 * D2R; // 0.1 deg
        } else {
            pos_prior_std = 0.1;       // 0.1 m
            att_prior_std = 1.0 * D2R; // 1.0 deg
        }

        memcpy(extrinsic_lidar_prior_, extrinsic_b_l_, sizeof(double) * 7);
        for (size_t k = 0; k < 3; k++) {
            extrinsic_lidar_prior_std_[k + 0] = pos_prior_std;
            extrinsic_lidar_prior_std_[k + 3] = att_prior_std;
        }
    }

    is_use_prior_ = true;
}

const vector<double> &Optimizer::reprojectionError() {
    // 路标点重投影误差统计
    vector<double> reprojection_errors;
    for (auto &landmark : visual_map_->landmarks()) {
        auto mappoint = landmark.second;
        if (!mappoint || mappoint->isOutlier()) {
            continue;
        }

        // 未参与优化的无效路标点
        if (visual_mappoints_.find(mappoint->id()) == visual_mappoints_.end()) {
            continue;
        }

        // 所有的观测
        vector<double> errors;
        for (const auto &observation : mappoint->observations()) {
            auto feat = observation.lock();
            // 无效特征
            if (!feat || feat->isOutlier()) {
                continue;
            }
            auto frame = feat->getFrame();
            // 非关键帧特征
            if (!frame || !frame->isKeyFrame()) {
                continue;
            }

            double error = camera_->reprojectionError(frame->pose(), mappoint->pos(), feat->keyPoint()).norm();
            errors.push_back(error);
        }
        if (errors.empty()) {
            LOGE << "Mappoint " << mappoint->id() << " with zero observation";
            continue;
        }
        double avg_error = std::accumulate(errors.begin(), errors.end(), 0.0) / static_cast<double>(errors.size());
        reprojection_errors.emplace_back(avg_error);
    }

    // 避免为空造成内存错误
    if (reprojection_errors.empty()) {
        reprojection_errors.push_back(0);
    }
    reprojection_.clear();

    // 最小误差
    double min_error = *std::min_element(reprojection_errors.begin(), reprojection_errors.end());
    reprojection_.push_back(min_error);
    // 最大误差
    double max_error = *std::max_element(reprojection_errors.begin(), reprojection_errors.end());
    reprojection_.push_back(max_error);
    // 平均误差
    double avg_error = std::accumulate(reprojection_errors.begin(), reprojection_errors.end(), 0.0) /
                       static_cast<double>(reprojection_errors.size());
    reprojection_.push_back(avg_error);
    // RMSE
    double sq_sum =
        std::inner_product(reprojection_errors.begin(), reprojection_errors.end(), reprojection_errors.begin(), 0.0);
    double rms_error = std::sqrt(sq_sum / static_cast<double>(reprojection_errors.size()));
    reprojection_.push_back(rms_error);

    return reprojection_;
}

void Optimizer::updateCameraExtrinsic(Pose &pose_b_c, double &td_b_c) const {
    pose_b_c = pose_b_c_;
    td_b_c   = extrinsic_b_c_[7];
}

void Optimizer::updateLidarExtrinsic(Pose &pose_b_l, double &td_b_l) const {
    pose_b_l = pose_b_l_;
    td_b_l   = extrinsic_b_l_[7];
}

void Optimizer::updateLidarToCameraExtrinsic(Pose &pose_c_l) const {
    // 使用估计的外参链式计算
    pose_c_l.R = pose_b_c_.R.transpose() * pose_b_l_.R;
    pose_c_l.t = pose_b_c_.R.transpose() * (pose_b_l_.t - pose_b_c_.t);
}