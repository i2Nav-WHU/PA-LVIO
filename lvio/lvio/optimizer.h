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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "lidar/lidar_feature.h"
#include "lidar/lidar_map.h"
#include "visual/visual_map.h"

#include "factors/marginalization_info.h"
#include "preintegration/preintegration.h"

#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <deque>
#include <memory>

using std::deque;
using std::string;
using std::vector;

typedef IntegrationStateData StateData;
typedef std::shared_ptr<PreintegrationBase> PreBasePtr;
typedef ceres::ResidualBlockId ResidualId;

typedef vector<ResidualId> Residuals;
typedef vector<std::pair<ResidualId, bool>> ResidualOutliers;
typedef vector<std::pair<ResidualId, lidar::LidarFeature::Ptr>> LidarResiduals;

class Optimizer {

public:
    typedef std::shared_ptr<Optimizer> Ptr;

    Optimizer() = delete;
    Optimizer(const string &configfile, const std::shared_ptr<const IntegrationParameters> parameters,
              visual::VisualMap::Ptr visual_map, visual::Camera::Ptr camera, lidar::LidarMap::Ptr lidar_map);

    bool marginalization(deque<double> &timelist, deque<StateData> &statedatalist,
                         deque<PreBasePtr> &preintegrationlist);
    bool optimization(const deque<double> &timelist, deque<StateData> &statedatalist,
                      deque<PreBasePtr> &preintegrationlist);

    const vector<double> &reprojectionError();

    void constructPrior(const StateData &statedata, bool is_zero_velocity);

    static int getStateDataIndex(const deque<double> &timelist, double time);

private:
    void updateParametersFromOptimizer(const deque<double> &timelist, const deque<StateData> &statedatalist);

    bool visualLandmarkOutlierCulling();

    void addStateParameters(ceres::Problem &problem, deque<StateData> &statedatalist);
    void addVisualParameters(ceres::Problem &problem);
    void addLidarParameters(ceres::Problem &problem);
    void doReintegration(std::deque<PreBasePtr> &preintegrationlist, std::deque<StateData> &statedatalist);
    void depthOnlyOptimization(const deque<double> &timelist, deque<StateData> &statedatalist);

    void addDepthMappointMscFactors(ceres::Problem &problem, bool isusekernel, const deque<double> &timelist,
                                    deque<StateData> &statedatalist);
    Residuals addReprojectionFactors(ceres::Problem &problem, bool isusekernel, const deque<double> &timelist,
                                     deque<StateData> &statedatalist);
    void addImuFactors(ceres::Problem &problem, deque<StateData> &statedatalist, deque<PreBasePtr> &preintegrationlist);
    LidarResiduals addLidarFactors(ceres::Problem &problem, bool isusekernel, const deque<double> &timelist,
                                   deque<StateData> &statedatalist);

    static ResidualOutliers lidarOutlierCullingByChi2(ceres::Problem &problem, LidarResiduals &residual_ids);
    static ResidualOutliers visualOutlierCullingByChi2(ceres::Problem &problem, Residuals &residual_ids);

    void getLatestStateStd(ceres::Problem &problem, const StateData &statedata, vector<double> &std);
    void visualMappointDepthStd(visual::VisualMapPoint::Ptr mappoint, double &depth, double &depth_std);

    void lidarFrameToMapOptimization(lidar::LidarFrame::Ptr frame);

public:
    void updateCameraExtrinsic(Pose &pose_b_c, double &td_b_c) const;
    void updateLidarExtrinsic(Pose &pose_b_l, double &td_b_l) const;
    void updateLidarToCameraExtrinsic(Pose &pose_c_l) const;

    const vector<Vector3d> &newFixedVisualMappoints() {
        return fixed_visual_mappoints_;
    }

    const vector<int> &iterations() {
        return iterations_;
    }

    const vector<double> &timecosts() {
        return timecosts_;
    }

    const vector<double> &outliers() {
        return outliers_;
    }

    const vector<double> &latestStateStd() {
        return latest_state_std_;
    }

private:
    struct PointCloudMap {
        double stamp{0};
        double distance{0};

        Pose pose;
        double pose_state[7];
        double pose_std[6];

        PointCloudPtr point_cloud;
        pcl::KdTreeFLANN<PointType>::Ptr kdtree;

        vector<lidar::LidarFeature::Ptr> oldest_features;
        vector<lidar::LidarFeature::Ptr> latest_features;

        bool is_using{false};
        int num_features{0};
        int oldest_num_features{0};
        int latest_num_features{0};
        int num_frames{0};

        lidar::LidarFrame::Ptr reference_frame;
    };

    // 估计雷达时间偏移需要的最低速度
    const double MINIMUM_SPEED_ESTIMATE_LIDAR_TD = 1.0; // 1.0 m/s

    // 视觉路标激光深度观测标准差
    const double VISUAL_LANDMARK_LIDAR_DEPTH_STD = 0.1; // 0.1 m

    const double ZERO_VELOCITY_STD = 0.03; // 0.03 m/s
    double imudatarate_{200};

    std::shared_ptr<const IntegrationParameters> integration_parameters_;

    visual::VisualMap::Ptr visual_map_;
    visual::Camera::Ptr camera_;
    lidar::LidarMap::Ptr lidar_map_;

    // 传感器使用
    bool is_use_visual_{true};
    bool is_use_lidar_{false};
    bool is_use_lidar_depth_{false};
    bool is_use_frame_to_map_{false};
    bool is_use_zupt_{true};

    // 边缘化
    std::shared_ptr<MarginalizationInfo> last_marginalization_info_{nullptr};
    vector<double *> last_marginalization_parameter_blocks_;
    vector<Vector3d> fixed_visual_mappoints_;

    vector<visual::VisualMapPoint::Ptr> depth_mappoints_;

    // 估计参数
    std::unordered_map<ulong, double> visual_mappoints_;
    double extrinsic_b_c_[8]{0};
    double extrinsic_b_l_[8]{0};
    Pose pose_b_c_, pose_b_l_;

    // 先验
    bool is_use_prior_{false};
    double mix_prior_[9];
    double mix_prior_std_[9];
    double pose_prior_[7];
    double pose_prior_std_[6];

    double extrinsic_cam_prior_[7];
    double extrinsic_cam_prior_std_[6];
    double extrinsic_lidar_prior_[7];
    double extrinsic_lidar_prior_std_[6];
    double gb_prior_std_;
    double ab_prior_std_;

    // 优化选项
    bool optimize_estimate_cam_extrinsic_;
    bool optimize_estimate_cam_td_;
    bool optimize_cam_extrinsic_accurate_;

    bool optimize_estimate_lidar_extrinsic_;
    bool optimize_estimate_lidar_td_;
    bool optimize_lidar_extrinsic_accurate_;

    double optimize_reprojection_factor_std_;
    double optimize_point_to_plane_std_;
    size_t optimize_windows_size_;
    size_t optimize_half_windows_size_;
    bool optimize_calculate_covariance_;

    double optimize_reprojection_std_;

    // 统计参数
    vector<int> iterations_;
    vector<double> timecosts_;
    vector<double> outliers_;
    vector<double> reprojection_;
    vector<double> latest_state_std_;
};

#endif // OPTIMIZER_H
