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

#include "lidar/lidar_map.h"

#include "common/logging.h"
#include "common/misc.h"
#include "common/timecost.h"

#include <tbb/tbb.h>
#include <yaml-cpp/yaml.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <filesystem>

namespace lidar {

LidarMap::LidarMap(const string &configfile, const string &outputpath) {
    // 加载配置
    YAML::Node config;
    std::vector<double> vecdata;
    config = YAML::LoadFile(configfile);

    point_to_plane_std_ = config["optimizer"]["optimize_point_to_plane_std"].as<double>();
    window_size_        = config["optimizer"]["optimize_window_size"].as<size_t>();
    half_window_size_   = window_size_ / 2;

    // 点云处理参数

    downsample_size_            = config["lidar"]["downsample_size"].as<double>();
    lidar_frame_rate_           = config["lidar"]["frame_rate"].as<double>();
    plane_estimation_threshold_ = config["lidar"]["plane_estimation_threshold"].as<double>();
    point_filter_num_           = config["lidar"]["point_filter_num"].as<int>();

    // 保存点云地图
    is_save_pointcloud_ = config["lidar"]["is_save_pointcloud"].as<bool>();
    if (is_save_pointcloud_) {
        pcd_file_ = outputpath + "/lvio_map.pcd";

        saved_pointcloud_ = PointCloudPtr(new PointCloud);
    }

    // 保存单帧点云
    if (config["lidar"]["is_save_frame_pointcloud"]) {
        is_save_frame_pointcloud_ = config["lidar"]["is_save_frame_pointcloud"].as<bool>();
        if (is_save_frame_pointcloud_) {
            frame_pcd_path_ = outputpath + "/PointClouds";
            std::filesystem::create_directory(frame_pcd_path_);
            lidar_pose_saver_ =
                std::make_shared<FileSaver>(outputpath + "/PointClouds/lidar_pose.csv", 8, FileSaver::TEXT);
        }
    }

    // 体素滤波降采样
    downsample_filter_.setLeafSize(downsample_size_, downsample_size_, downsample_size_);
    lidar_frame_dt_ = 1.0 / lidar_frame_rate_;

    // frame-to-map模式, 全局地图由被边缘化帧在线增量构建
    if (config["lidar"]["is_use_frame_to_map"]) {
        is_use_frame_to_map_ = config["lidar"]["is_use_frame_to_map"].as<bool>();
        if (is_use_frame_to_map_) {
            global_map_ikd_tree_ = std::make_shared<ikd_Tree::KD_TREE<PointType>>();
            global_map_ikd_tree_->setDownsampleParam(downsample_size_);
        }
    }

    LOGI << "Lidar mapping is constructed";
}

bool LidarMap::keyframeSelection(LidarFrame::Ptr frame) {
    auto last_keyframe = keyframes_.back();

    // 关键帧选择: 时间, 平移, 旋转, 覆盖率
    stat_keyframe_interval_    = frame->stamp() - last_keyframe->stamp();
    stat_keyframe_translation_ = (frame->pose().t - last_keyframe->pose().t).norm();
    Eigen::AngleAxisd angleaxis(frame->pose().R.transpose() * last_keyframe->pose().R);
    stat_keyframe_rotation_ = fabs(angleaxis.angle());

    if (((stat_keyframe_translation_ > MINIMUM_KEYFRAME_TRANSLATION) &&
         (stat_keyframe_interval_ > DEFAULT_KEYFRAME_INTERVAL)) ||
        (stat_keyframe_rotation_ > MINIMUM_KEYFRAME_ROTATION) ||
        (stat_keyframe_interval_ > MAXIMUM_KEYFRAME_INTERVAL)) {
        // 设置为关键帧, 附加帧清零
        frame->setKeyFrame();
        LOGI << "New lidar keyframe at " << Logging::doubleData(frame->stamp()) << " with interval "
             << stat_keyframe_interval_ << " s, translation " << stat_keyframe_translation_ << ", rotation "
             << stat_keyframe_rotation_;
    }

    return frame->isKeyFrame();
}

void LidarMap::addNewKeyFrame(LidarFrame::Ptr frame) {
    if (!keyframes_.empty()) {
        auto last_keyframe         = keyframes_.back();
        stat_keyframe_interval_    = frame->stamp() - last_keyframe->stamp();
        stat_keyframe_translation_ = (frame->pose().t - last_keyframe->pose().t).norm();
        Eigen::AngleAxisd angleaxis(frame->pose().R.transpose() * last_keyframe->pose().R);
        stat_keyframe_rotation_ = fabs(angleaxis.angle());
    }
    keyframes_.push_back(frame);

    // 构建新的ikdtree, 加入雷达系的点云
    new_ikd_tree_mutex_.lock();
    if (!new_ikd_tree_) {
        new_ikd_tree_ = std::make_shared<ikd_Tree::KD_TREE<PointType>>();
    }
    new_ikd_tree_->setDownsampleParam(downsample_size_);
    lidar_frame_map_[frame->id()] = std::move(new_ikd_tree_);
    new_ikd_tree_mutex_.unlock();

    // 分配下一个ikd-Tree内存
    task_group_.run([this]() {
        new_ikd_tree_mutex_.lock();
        new_ikd_tree_ = std::make_shared<ikd_Tree::KD_TREE<PointType>>();
        new_ikd_tree_mutex_.unlock();
    });
}

void LidarMap::removeOldestLidarFrame() {
    marginalized_frame_ = keyframes_.front();
    keyframes_.pop_front();

    if (lidar_frame_map_.find(marginalized_frame_->id()) != lidar_frame_map_.end()) {
        // 将最老帧加入全局地图, 从而保证最新帧能够有效匹配
        if (is_use_frame_to_map_) {
            task_group_.run([this]() { addNewPointsToGlobalMap(marginalized_frame_); });
        }
        auto ikd_tree = lidar_frame_map_[marginalized_frame_->id()];
        lidar_frame_map_.erase(marginalized_frame_->id());

        // Release the ikd-Tree
        task_group_.run([ikd_tree]() { ikd_tree->release(); });
    }

    if (is_save_frame_pointcloud_) {
        // 保存关键帧点云
        static uint64_t counts = 0;
        string frame_file =
            frame_pcd_path_ + absl::StrFormat("/%0.6lf_%06llu.pcd", marginalized_frame_->stamp(), ++counts);

        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinaryCompressed(frame_file, *marginalized_frame_->pointCloudMapFull());

        Pose pose     = marginalized_frame_->pose();
        Quaterniond q = Quaterniond(pose.R);
        vector<double> data;
        data.push_back(marginalized_frame_->stamp());
        data.push_back(pose.t.x());
        data.push_back(pose.t.y());
        data.push_back(pose.t.z());
        data.push_back(q.x());
        data.push_back(q.y());
        data.push_back(q.z());
        data.push_back(q.w());
        lidar_pose_saver_->dump(data);
    }

    LOGI << "Remove lidar keyframe at " << Logging::doubleData(marginalized_frame_->stamp());
}

void LidarMap::addNewPointsToGlobalMap(LidarFrame::Ptr frame) {
    frame_to_map_mutex_.lock();

    // 将被边缘化帧的地图点云加入全局地图
    auto pointcloud_lidar = frame->pointCloudMap();
    auto pointcloud_world = PointCloudPtr(new PointCloud());
    PointCloudCommon::pointCloudProjection(frame->pose(), frame->pointCloudMap(), pointcloud_world);
    if (global_map_ikd_tree_->size()) {
        global_map_ikd_tree_->addPoints(pointcloud_world->points, true);
    } else {
        global_map_ikd_tree_->build(pointcloud_world->points);
    }

    frame_to_map_mutex_.unlock();
}

void LidarMap::removeNewestLidarFrame() {
    auto newest_frame = keyframes_.back();
    keyframes_.pop_back();

    if (lidar_frame_map_.find(newest_frame->id()) != lidar_frame_map_.end()) {
        auto ikd_tree = lidar_frame_map_[newest_frame->id()];
        lidar_frame_map_.erase(newest_frame->id());

        // Release the ikd-Tree
        task_group_.run([ikd_tree]() { ikd_tree->release(); });
    }

    LOGI << "Remove lidar frame at " << Logging::doubleData(newest_frame->stamp());
}

void LidarMap::mergeNonKeyframes(const vector<LidarFrame::Ptr> &nonkeyframes, LidarFrame::Ptr keyframe) {
    Pose keyframe_pose = keyframe->pose();
    Pose nonkeyframe_pose, relative_pose;

    for (auto nonkeyframe : nonkeyframes) {
        nonkeyframe_pose = nonkeyframe->pose();

        // From nonkeyframe to keyframe
        relative_pose.R = keyframe_pose.R.transpose() * nonkeyframe_pose.R;
        relative_pose.t = keyframe_pose.R.transpose() * (nonkeyframe_pose.t - keyframe_pose.t);

        PointCloudCommon::pointCloudProjection(relative_pose, nonkeyframe->pointCloudMapFull(),
                                               nonkeyframe->pointCloudMapFull());

        // 点云累积
        *keyframe->pointCloudMapFull() += *nonkeyframe->pointCloudMapFull();
    }
    keyframe->setNumMergedFrames(nonkeyframes.size());

    // 降采样
    downsample_filter_.setInputCloud(keyframe->pointCloudMapFull());
    downsample_filter_.filter(*keyframe->pointCloudMap());

    // 构建ikd-tree点云
    lidar_frame_map_.at(keyframe->id())->build(keyframe->pointCloudMap()->points);
}

bool LidarMap::lidarFrameProcessing(const std::deque<std::pair<IMU, IntegrationState>> &ins_window,
                                    const Pose &pose_b_l, LidarFrame::Ptr frame) {

    // 用于建立雷达帧的地图
    size_t points_size                = frame->rawPointCloud()->size();
    PointCloudPtr map_pointcloud_full = PointCloudPtr(new PointCloud);
    PointCloudPtr map_pointcloud      = PointCloudPtr(new PointCloud);
    map_pointcloud_full->resize(points_size);

    Pose lidar_pose;
    MISC::getPoseFromInsWindow(ins_window, pose_b_l, frame->stamp(), lidar_pose);
    double td = frame->timeDelay();

    tbb::parallel_for(size_t(0), points_size, [&](const size_t &k) {
        auto &raw = frame->rawPointCloud()->points[k];

        // 数据转换
        PointType point;
        point.intensity        = raw.intensity;
        point.getVector3fMap() = raw.getVector3fMap();

        // 当前点的位姿, 考虑每个点的时间偏移
        Pose curr_pose;
        MISC::getPoseFromInsWindow(ins_window, pose_b_l, raw.time + td, curr_pose);

        // 投影到第一个点的位姿上
        map_pointcloud_full->points[k] = PointCloudCommon::relativeProjection(lidar_pose, curr_pose, point);
    });

    // 运动畸变补偿降采样的点云
    PointCloudPtr undis_pointcloud = PointCloudPtr(new PointCloud);
    undis_pointcloud->reserve(points_size);

    // 抽取降采样
    int num_points = 0;
    for (size_t k = 0; k < points_size; k++) {
        const auto &point = map_pointcloud_full->points[k];
        if (num_points++ % point_filter_num_ == 0) {
            undis_pointcloud->points.push_back(point);
        }
    }

    // 降采样
    PointCloudPtr down_pointcloud = PointCloudPtr(new PointCloud);
    pcl::VoxelGrid<PointType> downsample_filter;
    downsample_filter.setLeafSize(downsample_size_, downsample_size_, downsample_size_);
    downsample_filter.setInputCloud(undis_pointcloud);
    downsample_filter.filter(*down_pointcloud);

    // 投影到世界系
    frame->setPose(lidar_pose);
    PointCloudPtr world_pointcloud = PointCloudPtr(new PointCloud);
    PointCloudCommon::pointCloudProjection(lidar_pose, down_pointcloud, world_pointcloud);

    // 去畸变且降采样的点云
    frame->setUndistortedPointCloud(undis_pointcloud, down_pointcloud, world_pointcloud, map_pointcloud_full,
                                    map_pointcloud);

    // LOGI << "New downsampled " << world_pointcloud->size() << " points from " << undis_pointcloud->size()
    //      << " raw points at lidar frame " << Logging::doubleData(frame->stamp());

    return true;
}

bool LidarMap::lidarFrameProcessing(const IntegrationState &state, const Pose &pose_b_l, LidarFrame::Ptr frame) {
    // 直接累积静态点云, 无需去畸变

    // 用于建立雷达帧的地图
    size_t points_size                = frame->rawPointCloud()->size();
    PointCloudPtr map_pointcloud_full = PointCloudPtr(new PointCloud);
    PointCloudPtr map_pointcloud      = PointCloudPtr(new PointCloud);
    map_pointcloud_full->resize(points_size);

    Pose lidar_pose = MISC::stateToPoseWithExtrinsic(state, pose_b_l);
    for (size_t k = 0; k < points_size; k++) {
        auto &raw = frame->rawPointCloud()->points[k];

        // 数据转换
        PointType point;
        point.intensity        = raw.intensity;
        point.getVector3fMap() = raw.getVector3fMap();

        map_pointcloud_full->points[k] = point;
    }

    // 运动畸变补偿降采样的点云
    PointCloudPtr undis_pointcloud = PointCloudPtr(new PointCloud);
    undis_pointcloud->reserve(points_size);

    // 抽取降采样
    int num_points = 0;
    for (size_t k = 0; k < points_size; k++) {
        const auto &point = map_pointcloud_full->points[k];
        if (num_points++ % point_filter_num_ == 0) {
            undis_pointcloud->points.push_back(point);
        }
    }

    // 降采样
    PointCloudPtr down_pointcloud = PointCloudPtr(new PointCloud);
    downsample_filter_.setInputCloud(undis_pointcloud);
    downsample_filter_.filter(*down_pointcloud);

    // 投影到世界系
    frame->setPose(lidar_pose);
    PointCloudPtr world_pointcloud = PointCloudPtr(new PointCloud);
    PointCloudCommon::pointCloudProjection(lidar_pose, down_pointcloud, world_pointcloud);

    // 去畸变且降采样的点云
    frame->setUndistortedPointCloud(undis_pointcloud, down_pointcloud, world_pointcloud, map_pointcloud_full,
                                    map_pointcloud);

    // LOGI << "New downsampled " << world_pointcloud->size() << " points from " << undis_pointcloud->size()
    //      << " raw points at lidar frame " << Logging::doubleData(frame->stamp());

    return true;
}

int LidarMap::findFeaturesInMap(ikd_Tree::KD_TREE<PointType>::Ptr ikdtree, LidarFrame::Ptr ref_frame,
                                LidarFrame::Ptr cur_frame) {

    auto pointcloud_lidar = cur_frame->pointCloudLidar();
    auto pointcloud_world = cur_frame->pointCloudWorld();
    size_t points_size    = pointcloud_lidar->size();

    // 特征保存在参考帧
    auto &features = ref_frame->features();

    // 分配内存
    features.resize(points_size);
    for (size_t k = 0; k < points_size; k++) {
        features[k] = std::make_shared<LidarFeature>();
    }

    // 世界系到参考雷达系的位姿变换
    Pose pose_ref_world;
    pose_ref_world.R = ref_frame->pose().R.transpose();
    pose_ref_world.t = -pose_ref_world.R * ref_frame->pose().t;

    vector<int> feature_types(points_size, 0);
    tbb::parallel_for(size_t(0), points_size, [&](const size_t &k) {
        const PointType &world = pointcloud_world->points[k];
        const PointType &lidar = pointcloud_lidar->points[k];

        // 投影到参考雷达帧坐标系
        PointType reference;
        reference.getVector3fMap() =
            (pose_ref_world.R * world.getVector3fMap().cast<double>() + pose_ref_world.t).cast<float>();

        vector<float> point_square_distance;
        vector<double> point_to_plane_dis;
        PointVector nearest;
        Vector3d normal_vector;
        Vector3d center;
        vector<Vector3d> nearest_points;
        double varaince   = 0.0;
        auto feature_type = FEATURE_NONE;

        // 最近邻点搜索
        ikdtree->nearestSearch(reference, PLANE_ESTIMATION_POINT_SZIE, nearest, point_square_distance,
                               MAXIMUM_SEARCH_SQUARE_DISTANCE);
        PointCloudCommon::pointVectorConversion(nearest, nearest_points);

        if (nearest_points.size() == static_cast<size_t>(PLANE_ESTIMATION_POINT_SZIE)) {
            if (point_square_distance.back() < MAXIMUM_SEARCH_SQUARE_DISTANCE) {
                // 拟合平面
                bool is_plane = PointCloudCommon::planeFittingWithDistance(
                    nearest_points, plane_estimation_threshold_, normal_vector, center, varaince, point_to_plane_dis);
                // 平面检查
                if (is_plane) {
                    // s = dis / sqrt(norm)
                    double dis = fabs(normal_vector.transpose() * (reference.getVector3fMap().cast<double>() - center));
                    double scalar = dis / sqrt(lidar.getVector3fMap().norm());
                    if ((scalar < 0.1) && (dis < 4.5 * point_to_plane_std_)) {
                        feature_type     = FEATURE_PLANE;
                        feature_types[k] = 1;
                    }
                }
            }
        }

        // 更新特征
        features[k]->updateFeature(nearest_points, point_to_plane_dis, normal_vector, center, varaince, feature_type,
                                   false);
    });

    // 统计有效特征
    int num_features = std::accumulate(feature_types.begin(), feature_types.end(), 0);

    return num_features;
}

int LidarMap::findFeaturesInGlobalMap(ikd_Tree::KD_TREE<PointType>::Ptr ikdtree, LidarFrame::Ptr cur_frame) {
    auto pointcloud_lidar = cur_frame->pointCloudLidar();
    auto pointcloud_world = cur_frame->pointCloudWorld();
    size_t points_size    = pointcloud_lidar->size();
    auto &features        = cur_frame->globalMapFeatures();

    // 分配内存
    features.resize(points_size);
    for (size_t k = 0; k < points_size; k++) {
        features[k] = std::make_shared<LidarFeature>();
    }

    vector<int> feature_types(points_size, 0);
    tbb::parallel_for(size_t(0), points_size, [&](const size_t &k) {
        const PointType &world = pointcloud_world->points[k];
        const PointType &lidar = pointcloud_lidar->points[k];

        vector<float> point_square_distance;
        vector<double> point_to_plane_dis;
        PointVector nearest;
        Vector3d normal_vector;
        Vector3d center;
        vector<Vector3d> nearest_points;
        double varaince   = 0.0;
        auto feature_type = FEATURE_NONE;

        // 最近邻点搜索
        ikdtree->nearestSearch(world, PLANE_ESTIMATION_POINT_SZIE, nearest, point_square_distance,
                               MAXIMUM_SEARCH_SQUARE_DISTANCE);
        PointCloudCommon::pointVectorConversion(nearest, nearest_points);

        if (nearest_points.size() == static_cast<size_t>(PLANE_ESTIMATION_POINT_SZIE)) {
            if (point_square_distance.back() < MAXIMUM_SEARCH_SQUARE_DISTANCE) {
                // 拟合平面
                bool is_plane =
                    PointCloudCommon::planeFittingWithDistance(nearest_points, plane_estimation_threshold_ * 1.5,
                                                               normal_vector, center, varaince, point_to_plane_dis);
                // 平面检查
                if (is_plane) {
                    // s = dis / sqrt(norm)
                    double dis    = fabs(normal_vector.transpose() * (world.getVector3fMap().cast<double>() - center));
                    double scalar = dis / sqrt(lidar.getVector3fMap().norm());
                    if ((scalar < 0.1) && (dis < 4.5 * point_to_plane_std_)) {
                        feature_type     = FEATURE_PLANE;
                        feature_types[k] = 1;

                        double std = PointCloudCommon::pointToPlaneStd(nearest, world, 0.03);
                        features[k]->setPointStd(std);
                    }
                }
            }
        }

        // 更新特征
        features[k]->updateFeature(nearest_points, point_to_plane_dis, normal_vector, center, varaince, feature_type,
                                   false);
    });

    // 统计有效特征
    int num_features = std::accumulate(feature_types.begin(), feature_types.end(), 0);

    return num_features;
}

bool LidarMap::addNewPointCloudToMap(PointCloudPtr map_pointcloud) {
    if (!marginalized_frame_) {
        return false;
    }

    // 可视化点云
    PointCloudCommon::pointCloudProjection(marginalized_frame_->pose(), marginalized_frame_->pointCloudMapFull(),
                                           map_pointcloud);

    // 保存点云地图
    if (is_save_pointcloud_ && !map_pointcloud->empty()) {
        *saved_pointcloud_ += *map_pointcloud;
    }

    return true;
}

bool LidarMap::updateLidarFeaturesInMap() {
    TimeCost timecost;

    // 搜索特征点
    vector<int> frame_features(keyframes_.size());
    size_t frame_size = keyframes_.size();
    auto cur_frame    = keyframes_.back();

    tbb::parallel_for(size_t(0), frame_size - 1, [&](const size_t &k) {
        auto ref_frame = keyframes_[k];

        auto ikdtree      = lidar_frame_map_[ref_frame->id()];
        frame_features[k] = findFeaturesInMap(ikdtree, ref_frame, cur_frame);
    });
    cur_frame->setNumFeatures(frame_features.back());

    // 拷贝与最近的关键帧的特征用于可视化
    cur_frame->features() = keyframes_[keyframes_.size() - 2]->features();
    frame_features.back() = cur_frame->numFeatures();

    stat_newest_features_  = frame_features.back();
    stat_average_features_ = std::accumulate(frame_features.begin(), frame_features.end(), 0.0) / keyframes_.size();

    // 平面BA关联
    stat_ba_features_ = checkPlaneCorresponding();

    // 关联全局地图
    if (is_use_frame_to_map_ && global_map_ikd_tree_->size()) {
        frame_to_map_mutex_.lock();
        int last_num_features = findFeaturesInGlobalMap(global_map_ikd_tree_, keyframes_.front());
        int cur_num_features  = findFeaturesInGlobalMap(global_map_ikd_tree_, cur_frame);
        frame_to_map_mutex_.unlock();

        LOGI << "Find " << last_num_features << " and " << cur_num_features << " features in global map";
    }

    stat_update_feature_timecost_ = timecost.costInMillisecond();
    LOGI << "Find the newest " << stat_newest_features_ << " features and " << stat_ba_features_
         << " plane BA features at " << Logging::doubleData(keyframes_.back()->stamp()) << ", "
         << "update map cost " << stat_update_feature_timecost_ << " ms";

    return true;
}

int LidarMap::checkPlaneCorresponding() {
    size_t frame_size      = keyframes_.size();
    const auto &cur_frame  = keyframes_.back();
    const auto &cur_points = cur_frame->pointCloudLidar()->points;
    auto points_size       = cur_points.size();
    auto &features_type    = cur_frame->featuresType();
    features_type.resize(points_size, FEATURE_NONE);

    // 计算当前点云关联的关键帧数量
    std::vector<int> features_counts(points_size, 0);
    for (size_t k = 0; k < points_size; k++) {
        int counts = 0;
        for (size_t i = 0; i < frame_size - 1; i++) {
            const auto &feature = keyframes_[i]->features()[k];
            if (!feature || (feature->featureType() == FEATURE_NONE) || feature->isOutlier()) {
                continue;
            }
            counts++;
        }
        // 包括当前帧的点
        features_counts[k] = counts + 1;
    }

    // 校验平面点是否可以进行BA, 否则退化为点到面约束
    int ba_count = 0;
    tbb::concurrent_vector<int> is_ba_points(points_size, 0);
    tbb::concurrent_vector<double> plane_std(points_size);
    tbb::parallel_for(size_t(0), points_size, [&](const size_t &k) {
        if (features_counts[k] >= MINIMUM_PLANE_BA_FEATURES) {
            vector<Vector3d> points_world;
            vector<LidarFeature::Ptr> features;
            vector<double> variances;
            for (size_t i = 0; i < frame_size - 1; i++) {
                const auto &frame   = keyframes_[i];
                const auto &feature = frame->features()[k];
                if (!feature || (feature->featureType() == FEATURE_NONE) || feature->isOutlier()) {
                    continue;
                }

                // 选用最近点
                const auto &nearest_point = feature->nearestPoints().front();
                Vector3d pw               = frame->pose().R * nearest_point + frame->pose().t;

                points_world.push_back(pw);
                features.push_back(feature);
                variances.push_back(feature->variance());
            }
            // 当前帧的点
            Vector3d pw = cur_frame->pose().R * cur_points[k].getVector3fMap().cast<double>() + cur_frame->pose().t;
            points_world.push_back(pw);

            // 计算每个平面协方差的RMS
            double rms = 0;
            for (const auto &var : variances) {
                rms += var * var;
            }
            rms        = sqrt(rms / static_cast<double>(variances.size()));
            double std = sqrt(rms) * 0.840896415; // var = sqrt(2) * std^2

            // 平面BA观测噪声
            plane_std[k] = std::max(std, MINIMUM_PLANE_BA_STD);

            // 平面校验阈值, 综合考虑协方差及设置平面阈值
            double threshold = 3.0 * std::min(std, plane_estimation_threshold_);

            // 平面拟合校验
            vector<double> point_to_plane_dis;
            Vector3d normal_vector;
            Vector3d center;
            double varaince;
            bool is_plane = PointCloudCommon::planeFittingWithDistance(points_world, threshold, normal_vector, center,
                                                                       varaince, point_to_plane_dis);
            if (is_plane) {
                for (const auto &feature : features) {
                    feature->setFeatureType(FEATURE_PLANE_BA);
                    feature->setPointStd(plane_std[k]);
                }
                features_type[k] = FEATURE_PLANE_BA;
                is_ba_points[k]  = 1;
            }
        }

        // 剩余点类型
        if (features_type[k] == FEATURE_NONE) {
            const auto &feature = keyframes_.front()->features()[k];
            if (!(!feature || (feature->featureType() == FEATURE_NONE) || feature->isOutlier())) {
                features_type[k] = FEATURE_PLANE;
            }
        }
    });

    // 有效平面BA点统计
    ba_count = std::accumulate(is_ba_points.begin(), is_ba_points.end(), 0);
    return ba_count;
}

bool LidarMap::saveMapPointCloud() {
    LOGW << "Start to save pcd file";

    // 随机降采样 1/10
    pcl::RandomSample<PointType> random_sample;
    random_sample.setSample(saved_pointcloud_->size() / 10);

    PointCloud downsampled;
    random_sample.setInputCloud(saved_pointcloud_);
    random_sample.filter(downsampled);

    LOGW << "Total " << downsampled.size() << " downsampled points";

    if (!downsampled.empty()) {
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinaryCompressed(pcd_file_, downsampled);
    }

    return is_save_pointcloud_;
}

const vector<double> &LidarMap::statisticParameters() {
    statistic_parameters_.clear();

    statistic_parameters_.push_back(keyframes_.back()->stamp());
    statistic_parameters_.push_back(stat_average_features_);
    statistic_parameters_.push_back(stat_newest_features_);
    statistic_parameters_.push_back(keyframes_.back()->pointCloudLidarFull()->size());
    statistic_parameters_.push_back(stat_ba_features_);
    statistic_parameters_.push_back(stat_keyframe_interval_);
    statistic_parameters_.push_back(stat_keyframe_translation_);
    statistic_parameters_.push_back(stat_keyframe_rotation_ * R2D);
    statistic_parameters_.push_back(stat_update_feature_timecost_);

    return statistic_parameters_;
}

} // namespace lidar
