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

#include "lvio/lvio.h"

#include "common/logging.h"
#include "common/misc.h"
#include "common/timecost.h"
#include "lidar/pointcloud.h"

#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/io/pcd_io.h>

#include <yaml-cpp/yaml.h>

LVIO::LVIO(const string &configfile, const string &outputpath, visual::VisualDrawer::Ptr visual_drawer,
           lidar::LidarViewer::Ptr lidar_viewer) {
    // 初始状态
    lviostate_ = LVIO_ERROR;

    // 加载配置
    YAML::Node config;
    std::vector<double> vecdata;
    config = YAML::LoadFile(configfile);

    // 文件IO
    visual_pts_filesaver_  = FileSaver::create(outputpath + "/visual_mappoint.txt", 3);
    visual_stat_filesaver_ = FileSaver::create(outputpath + "/visual_statistic.txt", 3);
    lidar_stat_filesaver_  = FileSaver::create(outputpath + "/lidar_statistic.txt", 3);
    cam_ext_filesaver_     = FileSaver::create(outputpath + "/visual_extrinsic.txt", 7);
    lidar_ext_filesaver_   = FileSaver::create(outputpath + "/lidar_extrinsic.txt", 7);
    imu_err_filesaver_     = FileSaver::create(outputpath + "/LVIO_IMU_ERR.bin", 7, FileSaver::BINARY);
    std_filesaver_         = FileSaver::create(outputpath + "/LVIO_STD.bin", 16, FileSaver::BINARY);
    traj_filesaver_        = FileSaver::create(outputpath + "/trajectory.csv", 8);

    // make a copy of configuration file
    std::ofstream ofconfig(outputpath + "/lvio.yaml");
    ofconfig << YAML::Dump(config);
    ofconfig.close();

    // 积分配置参数
    integration_parameters_ = std::make_shared<IntegrationParameters>();

    // IMU噪声参数
    integration_parameters_->gyr_arw      = config["imu"]["arw"].as<double>() * D2R / 60.0;
    integration_parameters_->gyr_bias_std = config["imu"]["gbstd"].as<double>() * D2R / 3600.0;
    integration_parameters_->acc_vrw      = config["imu"]["vrw"].as<double>() / 60.0;
    integration_parameters_->acc_bias_std = config["imu"]["abstd"].as<double>() * 1.0e-5;
    integration_parameters_->corr_time    = config["imu"]["corrtime"].as<double>() * 3600;
    if (config["imu"]["is_use_zupt"]) {
        is_use_zupt_ = config["imu"]["is_use_zupt"].as<bool>();
    }

    // IMU数据率
    imudatarate_ = config["imu"]["imudatarate"].as<double>();
    imudatadt_   = 1.0 / imudatarate_;

    // 固定正常重力
    integration_parameters_->gravity = {0, 0, NORMAL_GRAVITY};

    // 相机参数
    is_use_visual_            = config["visual"]["is_use_visual"].as<bool>();
    is_use_lidar_depth_       = config["visual"]["is_use_lidar_depth"].as<bool>();
    vector<double> intrinsic  = config["visual"]["intrinsic"].as<std::vector<double>>();
    vector<double> distortion = config["visual"]["distortion"].as<std::vector<double>>();
    vector<int> resolution    = config["visual"]["resolution"].as<std::vector<int>>();

    camera_ = visual::Camera::createCamera(intrinsic, distortion, resolution);

    // Camera外参
    vecdata           = config["visual"]["q_b_c"].as<std::vector<double>>();
    Quaterniond q_b_c = Eigen::Quaterniond(vecdata.data());
    vecdata           = config["visual"]["t_b_c"].as<std::vector<double>>();
    Vector3d t_b_c    = Eigen::Vector3d(vecdata.data());
    td_b_c_           = config["visual"]["td_b_c"].as<double>();

    pose_b_c_.R = q_b_c.toRotationMatrix();
    pose_b_c_.t = t_b_c;

    // 雷达参数
    is_use_lidar_     = config["lidar"]["is_use_lidar"].as<bool>();
    lidar_frame_rate_ = config["lidar"]["frame_rate"].as<double>();
    lidar_frame_dt_   = 1.0 / lidar_frame_rate_;

    is_save_pointcloud_ = config["lidar"]["is_save_pointcloud"].as<bool>();
    if (is_use_lidar_ && is_use_visual_ && is_save_pointcloud_) {
        rgb_pointcloud_map_ = RGBPointCloudPtr(new RGBPointCloud);
        rgb_pcd_file_       = outputpath + "/lvio_rgb_map.pcd";
    }

    // 保留的缓存帧数量
    reserved_buffer_counts_ = RESERVED_BUFFER_LENGTH / lidar_frame_dt_;

    // 雷达外参
    vecdata           = config["lidar"]["q_b_l"].as<std::vector<double>>();
    Quaterniond q_b_l = Eigen::Quaterniond(vecdata.data());
    vecdata           = config["lidar"]["t_b_l"].as<std::vector<double>>();
    Vector3d t_b_l    = Eigen::Vector3d(vecdata.data());
    td_b_l_           = config["lidar"]["td_b_l"].as<double>();

    pose_b_l_.R = q_b_l.toRotationMatrix();
    pose_b_l_.t = t_b_l;

    // 雷达相机外参
    pose_c_l_.R = pose_b_c_.R.transpose() * pose_b_l_.R;
    pose_c_l_.t = pose_b_c_.R.transpose() * (pose_b_l_.t - pose_b_c_.t);

    // 可视化
    is_use_visualization_ = config["is_use_visualization"].as<bool>();

    optimize_estimate_cam_extrinsic_   = config["optimizer"]["optimize_estimate_cam_extrinsic"].as<bool>();
    optimize_estimate_cam_td_          = config["optimizer"]["optimize_estimate_cam_td"].as<bool>();
    optimize_estimate_lidar_extrinsic_ = config["optimizer"]["optimize_estimate_lidar_extrinsic"].as<bool>();
    optimize_estimate_lidar_td_        = config["optimizer"]["optimize_estimate_lidar_td"].as<bool>();
    size_t optimize_window_size        = config["optimizer"]["optimize_window_size"].as<size_t>();
    optimize_calculate_covariance_     = config["optimizer"]["optimize_calculate_covariance"].as<bool>();

    // 清理容器
    preintegrationlist_.clear();
    statedatalist_.clear();
    timelist_.clear();

    // 视觉跟踪及可视化
    if (is_use_visual_) {
        visual_map_    = std::make_shared<visual::VisualMap>(optimize_window_size);
        visual_drawer_ = std::move(visual_drawer);
        if (is_use_visualization_) {
            visual_drawer_thread_ = std::thread(&visual::VisualDrawer::run, visual_drawer_);
        }
        tracking_ = std::make_shared<visual::Tracking>(camera_, visual_map_, visual_drawer_, configfile, outputpath);
    }

    // 激光处理及可视化
    if (is_use_lidar_) {
        lidar_map_    = std::make_shared<lidar::LidarMap>(configfile, outputpath);
        lidar_viewer_ = std::move(lidar_viewer);
        if (is_use_visualization_) {
            lidar_viewer_thread_ = std::thread(&lidar::LidarViewer::run, lidar_viewer_);
        }
    }

    // 优化对象
    optimizer_ = std::make_shared<Optimizer>(configfile, integration_parameters_, visual_map_, camera_, lidar_map_);

    // 融合处理三个线程
    fusion_thread_ = std::thread(&LVIO::runFusion, this);

    // 更新系统状态, 进入初始化状态
    lviostate_ = LVIO_INITIALIZING;
}

bool LVIO::addNewImu(const IMU &imu) {
    if (imu_buffer_mutex_.try_lock()) {
        imu_buffer_.push(imu);
        imu_buffer_mutex_.unlock();
        return true;
    }

    return false;
}

bool LVIO::addNewVisualFrame(const VisualFrame::Ptr &frame) {
    Lock lock(visual_frame_buffer_mutex_);
    visual_frame_buffer_.push(frame);

    return true;
}

bool LVIO::addNewLidarFrame(const LidarFrame::Ptr &frame) {
    // 需要构建初始地图
    Lock lock(lidar_frame_buffer_mutex_);
    lidar_frame_buffer_.push(frame);

    return true;
}

bool LVIO::addNewPointCloud(PointCloudCustomPtr pointcloud) {
    // 激光深度增强
    if (is_use_lidar_depth_) {
        Lock lock(lidar_point_buffer_mutex_);

        // 拷贝点云数据
        lidar_point_buffer_.insert(lidar_point_buffer_.end(), pointcloud->points.begin(), pointcloud->points.end());
        double reserved_time = RESERVED_BUFFER_LENGTH + LIDAR_DEPTH_POINT_STACK_INTERVAL * 2;
        while (fabs(lidar_point_buffer_.back().time - lidar_point_buffer_.front().time) > reserved_time) {
            lidar_point_buffer_.pop_front();
        }
    }

    return true;
}

bool LVIO::canAddData() {
    // Buffer容量为空则可以继续加入数据, 注意考虑IMU数据需要超前, 注意点云数据的累积处理
    return (visual_frame_buffer_.size() < reserved_buffer_counts_) &&
           (lidar_frame_buffer_.size() < reserved_buffer_counts_);
}

bool LVIO::isBufferEmpty() {
    return imu_buffer_.empty() || (is_use_visual_ && visual_frame_buffer_.empty()) ||
           (is_use_lidar_ && lidar_frame_buffer_.empty());
}

void LVIO::setFinished() {
    is_finished_ = true;

    if (is_use_visualization_) {
        // 结束绘图线程
        if (is_use_visual_) {
            visual_drawer_->setFinished();
            visual_drawer_thread_.join();
        }
        if (is_use_lidar_) {
            lidar_viewer_->setFinished();
            lidar_viewer_thread_.join();
        }
    }
    task_group_.wait();
    fusion_thread_.join();

    if (is_use_visual_ && (optimize_estimate_cam_extrinsic_ || optimize_estimate_cam_td_)) {
        Quaterniond q_b_c = Rotation::matrix2quaternion(pose_b_c_.R);
        Vector3d t_b_c    = pose_b_c_.t;
        LOGW << "Estimated camera extrinsics: "
             << absl::StrFormat("(%0.6lf, %0.6lf, %0.6lf, %0.6lf), (%0.3lf, %0.3lf, %0.3lf), %0.4lf", q_b_c.x(),
                                q_b_c.y(), q_b_c.z(), q_b_c.w(), t_b_c.x(), t_b_c.y(), t_b_c.z(), td_b_c_);
    }
    if (is_use_lidar_ && (optimize_estimate_lidar_extrinsic_ || optimize_estimate_lidar_td_)) {
        Quaterniond q_b_l = Rotation::matrix2quaternion(pose_b_l_.R);
        Vector3d t_b_l    = pose_b_l_.t;
        LOGW << "Estimated lidar extrinsics: "
             << absl::StrFormat("(%0.6lf, %0.6lf, %0.6lf, %0.6lf), (%0.3lf, %0.3lf, %0.3lf), %0.4lf", q_b_l.x(),
                                q_b_l.y(), q_b_l.z(), q_b_l.w(), t_b_l.x(), t_b_l.y(), t_b_l.z(), td_b_l_);
    }
    LOGW << "LVIO has finished processing";
}

void LVIO::runFusion() {
    VisualFrame::Ptr visual_frame = nullptr;
    LidarFrame::Ptr lidar_frame   = nullptr;
    bool is_need_optimization     = false;
    vector<LidarFrame::Ptr> lidar_nonkeyframes_;

    double last_update_time = 0;

    PointCloudPtr map_pointcloud           = PointCloudPtr(new PointCloud);
    PointCloudPtr last_map_pointcloud      = PointCloudPtr(new PointCloud);
    PointCloudPtr last_last_map_pointcloud = PointCloudPtr(new PointCloud);
    PointCloudPtr clean_pointcloud         = PointCloudPtr(new PointCloud);

    LOGI << "Fusion thread is started";
    while (!is_finished_) {
        if ((lviostate_ == LVIO_INITIALIZING)) {
            // 初始化需要IMU数据有效
            while (!is_finished_ && imu_buffer_.empty()) {
                usleep(WAITING_DELAY_IN_US);
            }
        } else {
            // 以视觉为主
            if (is_use_visual_) {
                // 等待图像数据
                while (!is_finished_ && visual_frame_buffer_.empty()) {
                    usleep(WAITING_DELAY_IN_US);
                }

                // 取最新的关键帧
                if (!is_finished_) {
                    Lock lock(visual_frame_buffer_mutex_);
                    visual_frame = visual_frame_buffer_.front();
                    visual_frame_buffer_.pop();
                }
            } else if (is_use_lidar_) {
                // 等待雷达数据
                while (!is_finished_ && lidar_frame_buffer_.empty()) {
                    usleep(WAITING_DELAY_IN_US);
                }

                // 取最新的关键帧
                if (!is_finished_) {
                    Lock lock(lidar_frame_buffer_mutex_);
                    lidar_frame = lidar_frame_buffer_.front();
                    lidar_frame_buffer_.pop();
                }
            }
        }
        if (is_finished_) {
            break;
        }
        if ((visual_frame_buffer_.size() > reserved_buffer_counts_) ||
            (lidar_frame_buffer_.size() > reserved_buffer_counts_)) {
            LOGI << "Buffer size: visual " << visual_frame_buffer_.size() << ", lidar " << lidar_frame_buffer_.size();
        }

        // 在改变状态节点前等待处理完成
        if (is_fusing_) {
            task_group_.wait();
        }
        is_fusing_ = true;

        // 融合状态
        if (lviostate_ == LVIO_INITIALIZING) {
            // 移除过多的INS状态
            while (ins_window_.size() > MAXIMUM_INS_NUMBER) {
                ins_window_.pop_front();
            }

            // 初始时刻
            if (ins_window_.empty()) {
                Lock lock(imu_buffer_mutex_);
                auto imu = imu_buffer_.front();

                // 向前取整, 取下一个有效时刻
                last_update_time = ceil(imu.time / INS_INITIALIZATION_INTERVAL) * INS_INITIALIZATION_INTERVAL;
            }

            // 等待IMU数据有效并加入INS窗口
            double current_time = last_update_time + INS_INITIALIZATION_INTERVAL;
            if (!waitImuAddToInsWindow(current_time)) {
                break;
            }

            // 初始化参数, 第一个节点
            if (lvioInitialization(last_update_time, current_time)) {
                if (is_use_visual_) {
                    lviostate_ = LVIO_INITIALIZING_VISUAL;
                } else if (is_use_lidar_) {
                    lviostate_ = LVIO_NORMAL;
                }
            }
            last_update_time = current_time;
        } else if (lviostate_ >= LVIO_INITIALIZING_VISUAL) {
            if (is_use_visual_) {
                // 视觉模式

                // 视觉跟踪处理
                if (lvioVisualTrackingProcessing(visual_frame) == 1) {

                    // 在改变状态节点前等待处理完成
                    if (is_fusing_) {
                        task_group_.wait();
                    }
                    is_fusing_ = true;

                    // 插入关键帧节点
                    addNewVisualFrameTimeNode(visual_frame);

                    // 初始帧不优化
                    if (lviostate_ == LVIO_INITIALIZING_VISUAL) {
                        is_need_optimization = false;

                        // 累积点云帧
                        if (is_use_lidar_) {
                            if (accumulateDynamicPointcloud(visual_frame->stamp())) {
                                is_need_optimization = true;
                            }
                        }

                        // VIO初始化完成
                        lviostate_ = LVIO_NORMAL;
                    } else if (lviostate_ == LVIO_NORMAL) {
                        // 添加有效雷达帧到当前相机时刻
                        if (is_use_lidar_) {
                            accumulateDynamicPointcloud(visual_frame->stamp());
                        }

                        is_need_optimization = true;
                    }
                }

            } else if (is_use_lidar_) {
                // 激光模式

                int result = lvioLidarProcessing(lidar_frame);
                if (result == 1) {
                    // 合并非关键帧到上一关键帧构建点云地图
                    auto last_keyframe = lidar_map_->keyframes().back();
                    lidar_map_->mergeNonKeyframes(lidar_nonkeyframes_, last_keyframe);
                    LOGI << "Accumulate " << lidar_nonkeyframes_.size() << " frames to "
                         << Logging::doubleData(last_keyframe->stamp()) << " with "
                         << last_keyframe->pointCloudMap()->size() << " points";
                    lidar_nonkeyframes_.clear();

                    // 在改变状态节点前等待处理完成
                    if (is_fusing_) {
                        task_group_.wait();
                    }
                    is_fusing_ = true;

                    // 加入新的雷达帧
                    addNewLidarFrameTimeNode(lidar_frame, true);

                    // 更新点云特征
                    lidar_map_->updateLidarFeaturesInMap();

                    is_need_optimization = true;
                } else if (result == 0) {
                    // 非关键帧累积
                    lidar_nonkeyframes_.push_back(lidar_frame);
                }
            }
        }

        if (is_need_optimization) {
            is_need_optimization = false;

            // 优化求解
            lvioOptimizationProcessing();

            // 更新窗口内状态
            auto state = Preintegration::stateFromData(statedatalist_.back());
            MISC::redoInsMechanization(integration_parameters_, state, RESERVED_INS_NUM, ins_window_);

            // 可视化
            if (is_use_visualization_) {
                task_group_.run(
                    [this, &map_pointcloud, &last_last_map_pointcloud, &last_map_pointcloud, &clean_pointcloud]() {
                        // 当雷达有效时, 仅显示点云地图
                        if (is_use_lidar_) {
                            // 更新点云地图
                            lidar_map_->addNewPointCloudToMap(map_pointcloud);

                            // 可视化
                            auto frame = lidar_map_->keyframes().back();
                            lvioLidarVisualization(frame, map_pointcloud);

                            if (is_use_visual_ && !map_pointcloud->empty()) {
                                // if (last_last_map_pointcloud->empty()) {
                                //     *last_last_map_pointcloud = *last_map_pointcloud;
                                //     *last_map_pointcloud      = *map_pointcloud;
                                //     *clean_pointcloud         = *map_pointcloud;
                                // } else {
                                //     // 剔除动态点云
                                //     auto kdtree = pcl::KdTreeFLANN<PointType>::Ptr(new
                                //     pcl::KdTreeFLANN<PointType>()); kdtree->setInputCloud(last_last_map_pointcloud);

                                //     vector<int> outliers;
                                //     for (size_t i = 0; i < map_pointcloud->size(); i++) {
                                //         PointType p = map_pointcloud->points[i];
                                //         std::vector<int> indices;
                                //         std::vector<float> sqr_distances;
                                //         kdtree->nearestKSearch(p, 1, indices, sqr_distances);
                                //         // 距离阈值
                                //         if (sqr_distances[0] > 0.15 * 0.15) {
                                //             outliers.push_back(i);
                                //         }
                                //     }

                                //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
                                //     inliers->indices = outliers;

                                //     pcl::ExtractIndices<PointType> extract;
                                //     extract.setInputCloud(map_pointcloud);
                                //     extract.setIndices(inliers);
                                //     extract.setNegative(true); // 设置为true表示移除这些点，而非提取
                                //     extract.filter(*clean_pointcloud);

                                //     *last_last_map_pointcloud = *last_map_pointcloud;
                                //     *last_map_pointcloud      = *map_pointcloud;
                                //     LOGW << "Remove " << outliers.size() << " outlier points from "
                                //          << map_pointcloud->size() << " points";
                                // }

                                // 可视化RGB点云
                                auto visual_frame = visual_map_->oldestKeyFrame();
                                // lvioRGBPointcloudVisualization(visual_frame, clean_pointcloud);
                                lvioRGBPointcloudVisualization(visual_frame, map_pointcloud);
                            }
                        } else if (is_use_visual_) {
                            lvioVisualVisualization();
                        }

                        // 完成所有处理继续
                        is_fusing_ = false;
                    });
            } else {
                if (is_use_lidar_) {
                    task_group_.run([this, &map_pointcloud]() {
                        // 更新点云地图
                        lidar_map_->addNewPointCloudToMap(map_pointcloud);

                        // 保存RGB点云
                        if (is_save_pointcloud_ && is_use_visual_ && !map_pointcloud->empty()) {
                            auto visual_frame = visual_map_->oldestKeyFrame();
                            lvioRGBPointcloudVisualization(visual_frame, map_pointcloud);
                        }

                        // 完成所有处理继续
                        is_fusing_ = false;
                    });
                } else {
                    is_fusing_ = false;
                }
            }
        } else {
            is_fusing_ = false;
        }

    } // while

    if (is_use_lidar_ && is_save_pointcloud_) {
        lidar_map_->saveMapPointCloud();
        if (is_use_visual_ && !rgb_pointcloud_map_->empty()) {
            pcl::io::savePCDFileBinary(rgb_pcd_file_, *rgb_pointcloud_map_);
            LOGW << "Save " << rgb_pointcloud_map_->size() << " RGB point cloud";
        }
    }

    LOGI << "Fusion thread is exited";
}

bool LVIO::lvioOptimizationProcessing() {

    // 两次非线性优化并进行粗差剔除
    optimizer_->optimization(timelist_, statedatalist_, preintegrationlist_);

    if (is_use_visual_ && visual_map_->isWindowFull()) {
        // 移除所有窗口中间插入的非关键帧
        removeNonKeyFrame();
    }

    // 边缘化, 移除旧的观测, 按时间对齐到保留的最后一个关键帧
    if (is_use_visual_) {
        if (visual_map_->isWindowFull()) {
            optimizer_->marginalization(timelist_, statedatalist_, preintegrationlist_);
        }
    } else if (is_use_lidar_) {
        if (lidar_map_->isWindowFull()) {
            optimizer_->marginalization(timelist_, statedatalist_, preintegrationlist_);
        }
    }

    // 更新外参参数
    updateParametersFromOptimizer();

    // 统计并输出视觉相关的参数
    parametersStatistic();

    return true;
}

void LVIO::lvioRGBPointcloudVisualization(const VisualFrame::Ptr frame, const PointCloudPtr pointcloud) {
    TimeCost timecost;

    const auto &image = frame->rawImage();
    const auto &pose  = frame->pose();

    // 将彩色图像在HSV颜色空间对亮度进行直方图均衡化
    cv::Mat hsv_image;
    cv::cvtColor(image, hsv_image, cv::COLOR_RGB2HSV);
    vector<cv::Mat> hsv_channels;
    cv::split(hsv_image, hsv_channels);
    cv::equalizeHist(hsv_channels[2], hsv_channels[2]);
    cv::merge(hsv_channels, hsv_image);
    static cv::Mat equalized_image;
    cv::cvtColor(hsv_image, equalized_image, cv::COLOR_HSV2RGB);

    size_t points_size = pointcloud->size();
    tbb::concurrent_vector<Vector3d> cam_points(points_size);
    tbb::concurrent_vector<cv::Point2f> pixel_points(points_size);
    tbb::concurrent_vector<bool> valid_points(points_size, false);
    tbb::parallel_for(size_t(0), points_size, [&](const size_t &k) {
        Vector3d world  = pointcloud->points[k].getVector3fMap().cast<double>();
        cam_points[k]   = camera_->world2cam(world, pose);
        pixel_points[k] = camera_->cam2pixel(cam_points[k]);
        camera_->distortPoint(pixel_points[k]);
        if (camera_->isPixelInImage(pixel_points[k]) && (cam_points[k].z() > 0) && ((cam_points[k].z() < 100))) {
            valid_points[k] = true;
        }
    });

    vector<size_t> final_indexs;
    for (size_t k = 0; k < points_size; k++) {
        if (valid_points[k]) {
            final_indexs.push_back(k);
        }
    }

    RGBPointCloudPtr rgb_pointcloud(new RGBPointCloud);
    rgb_pointcloud->resize(final_indexs.size());
    tbb::concurrent_vector<bool> rgb_valid_points(final_indexs.size(), false);
    tbb::parallel_for(size_t(0), final_indexs.size(), [&](const size_t &k) {
        size_t index = final_indexs[k];
        auto pixel   = pixel_points[index];
        vector<float> rgb(3);

        if (interpolatePixelColor(equalized_image, pixel, rgb)) {
            pcl::PointXYZRGB rgb_point;
            rgb_point.getVector3fMap() = pointcloud->points[index].getVector3fMap();

            rgb_point.r = rgb[0];
            rgb_point.g = rgb[1];
            rgb_point.b = rgb[2];

            rgb_pointcloud->points[k] = rgb_point;
            rgb_valid_points[k]       = true;
        }
    });

    RGBPointCloudPtr valid_rgb_pointcloud(new RGBPointCloud);
    for (size_t k = 0; k < final_indexs.size(); k++) {
        if (rgb_valid_points[k]) {
            valid_rgb_pointcloud->points.push_back(rgb_pointcloud->points[k]);
        }
    }

    // 可视化
    if (is_use_visualization_) {
        lidar_viewer_->updateRGBMapPointCloud(valid_rgb_pointcloud);
    }

    if (is_save_pointcloud_) {
        // 累积点云保存
        *rgb_pointcloud_map_ += *valid_rgb_pointcloud;
    }

    LOGI << "Colorize " << valid_rgb_pointcloud->size() << " points from " << pointcloud->size() << " points cost "
         << timecost.costInMillisecond() << " ms";
}

bool LVIO::lvioLidarVisualization(LidarFrame::Ptr frame, PointCloudPtr map_pointcloud) {
    // IMU的位姿
    auto state = Preintegration::stateFromData(statedatalist_.back());
    Pose pose;
    pose.R = state.q.toRotationMatrix();
    pose.t = state.p;
    lidar_viewer_->updateCurrentPose(pose);

    // 新的地图点云
    if (!map_pointcloud->empty() && !is_use_visual_) {
        lidar_viewer_->updateMapPointCloud(map_pointcloud);
    }

    // 当前帧点云
    lidar_viewer_->updateCurrentPointCloud(frame->pointCloudWorld());

    // 当前帧平面点
    PointCloudPtr plane_pointcloud = PointCloudPtr(new PointCloud);
    const auto &features           = frame->features();
    for (size_t k = 0; k < features.size(); k++) {
        if ((features[k]->featureType() == lidar::FEATURE_PLANE_BA) && !features[k]->isOutlier()) {
            plane_pointcloud->push_back(frame->pointCloudWorld()->points[k]);
        }
    }
    lidar_viewer_->updatePlanePointCloud(plane_pointcloud);

    return true;
}

bool LVIO::lvioVisualVisualization() {
    // 当前窗口内的路标点
    const auto &landmarks = visual_map_->landmarks();
    vector<Vector3d> current;
    for (const auto &landmark : landmarks) {
        const auto &mappoint = landmark.second;
        if (mappoint && !mappoint->isOutlier()) {
            current.push_back(mappoint->pos());
        }
    }
    if (!current.empty()) {
        visual_drawer_->updateCurrentMappoints(current);
    }

    // 新的固定的路标点
    const auto &fixed = optimizer_->newFixedVisualMappoints();
    if (!fixed.empty()) {
        visual_drawer_->updateNewFixedMappoints(fixed);
    }

    // 位姿可视化
    auto state = Preintegration::stateFromData(statedatalist_.back());
    Pose pose;
    pose.t = state.p;
    pose.R = state.q.toRotationMatrix();

    visual_drawer_->updateMap(pose);

    return true;
}

int LVIO::lvioVisualTrackingProcessing(VisualFrame::Ptr frame) {
    static PointCloud pointcloud;

    // 时标更新
    frame->setTimeDelay(td_b_c_);
    frame->setStamp(frame->stamp() + td_b_c_);

    // 进行机械编排处理
    if (!waitImuDoInsMechanization(frame->stamp())) {
        return -1;
    }

    TimeCost timecost;

    // 获取初始位姿
    Pose pose;
    MISC::getPoseFromInsWindow(ins_window_, pose_b_c_, frame->stamp(), pose);
    frame->setPose(pose);

    // 加锁, 和点云投影冲突
    auto trackstate = tracking_->track(frame, pointcloud);

    if (trackstate == visual::TRACK_LOST) {
        LOGE << "Tracking lost at " << Logging::doubleData(frame->stamp());
    }

    // 包括第一帧在内的所有关键帧, 跟踪失败时的当前帧也会成为新的关键帧
    if (tracking_->isNewKeyFrame() || (trackstate == visual::TRACK_FIRST_FRAME)) {
        if (is_use_lidar_depth_ && (frame->keyFrameState() != visual::KEYFRAME_REMOVE_SECOND_NEW)) {
            // 点云数据投影准备
            pointcloud.clear();

            // 点云数据投影到参考帧, 在上一个关键帧后的时间进行计算
            if (!lidar_point_buffer_.empty() && (lviostate_ == LVIO_NORMAL)) {
                pointcloudStackForVisualTracking(frame, pointcloud);
            }
        }

        stat_visual_tracking_cost_ = timecost.costInMillisecond();
        LOGI << "Tracking cost " << stat_visual_tracking_cost_ << " ms";
        return 1;
    }

    return 0;
}

bool LVIO::pointcloudStackForVisualTracking(const VisualFrame::Ptr frame, PointCloud &pointcloud) {
    TimeCost timecost;

    double last_stamp = frame->stamp() - LIDAR_DEPTH_POINT_STACK_INTERVAL;
    double curr_stamp = frame->stamp();

    while (!is_finished_) {
        lidar_point_buffer_mutex_.lock();
        auto point_stamp = lidar_point_buffer_.back().time;
        lidar_point_buffer_mutex_.unlock();
        if (point_stamp > curr_stamp) {
            break;
        }

        usleep(WAITING_DELAY_IN_US);
    }

    // 点云数据加锁
    Lock lock(lidar_point_buffer_mutex_);

    // 移除过期的点云
    while (true) {
        if (lidar_point_buffer_.empty()) {
            break;
        }

        auto &point = lidar_point_buffer_.front();
        if (point.time < last_stamp) {
            lidar_point_buffer_.pop_front();
        } else {
            break;
        }
    }

    if (lidar_point_buffer_.empty()) {
        LOGW << "No valid point cloud in current frame " << Logging::doubleData(curr_stamp);
        return false;
    }

    size_t num_projected = lidar_point_buffer_.size();
    for (size_t k = 0; k < lidar_point_buffer_.size(); k++) {
        if (lidar_point_buffer_[k].time > curr_stamp) {
            num_projected = k;
            break;
        }
    }
    pointcloud.points.resize(num_projected);

    // 使用连续的INS位姿
    Pose last_keyframe_pose;
    MISC::getPoseFromInsWindow(ins_window_, pose_b_c_, curr_stamp, last_keyframe_pose);

    // 有效点云数据投影
    tbb::parallel_for(size_t(0), num_projected, [&](const size_t &k) {
        auto &raw = lidar_point_buffer_[k];
        auto norm = raw.getVector3fMap().norm();
        if ((norm < visual::VisualMapPoint::FARTHEST_DEPTH) && (norm > visual::VisualMapPoint::NEAREST_DEPTH)) {
            // 获取点对应时刻的相机位姿, 考虑IMU数据延时
            Pose pose;
            MISC::getPoseFromInsWindow(ins_window_, pose_b_c_, raw.time + td_b_l_, pose);

            // 通过激光和相机外参转换到相机坐标系
            PointType point;
            point.getVector3fMap() = raw.getVector3fMap();
            pointcloud[k] =
                lidar::PointCloudCommon::relativeProjectionToCamera(pose_c_l_, last_keyframe_pose, pose, point);
        }
    });

    LOGI << "Projection to " << Logging::doubleData(curr_stamp) << " cost " << timecost.costInMillisecond()
         << " ms with size " << pointcloud.size();

    return !pointcloud.empty();
}

int LVIO::lvioLidarProcessing(LidarFrame::Ptr frame) {

    // 更新时标
    frame->setTimeDelay(td_b_l_);
    frame->setStamp(frame->stamp() + td_b_l_);

    // 进行机械编排处理
    if (!waitImuDoInsMechanization(frame->endTime())) {
        return -1;
    }

    // 去畸变降采样
    lidar_map_->lidarFrameProcessing(ins_window_, pose_b_l_, frame);

    // 关键帧选择
    bool is_keyframe = lidar_map_->keyframeSelection(frame);

    return is_keyframe ? 1 : 0;
}

bool LVIO::waitImuAddToInsWindow(double time) {
    double end_time = time + imudatadt_ * 2;

    // 等待IMU数据
    waitImuData(end_time);

    // IMU数据加入窗口
    Lock lock(imu_buffer_mutex_);
    while (true) {
        if (imu_buffer_.empty() || (imu_buffer_.front().time > end_time)) {
            break;
        }
        auto imu = imu_buffer_.front();
        imu_buffer_.pop();

        ins_window_.emplace_back(imu, IntegrationState());
    }

    return !is_finished_;
}

bool LVIO::waitImuDoInsMechanization(double time) {
    double end_time = time + imudatadt_ * 2;

    // 等待IMU数据
    waitImuData(end_time);

    // INS机械编排
    doInsMechanization(end_time);

    return !is_finished_;
}

bool LVIO::waitImuData(double time) {
    while (true) {
        imu_buffer_mutex_.lock();
        if ((!imu_buffer_.empty() && (imu_buffer_.back().time < time)) ||
            (imu_buffer_.empty() && !ins_window_.empty() && ins_window_.back().first.time < time)) {
            imu_buffer_mutex_.unlock();

            usleep(WAITING_DELAY_IN_US);
        } else {
            imu_buffer_mutex_.unlock();
            break;
        }

        if (is_finished_) {
            break;
        }
    }

    return true;
}

void LVIO::doInsMechanization(double time) {
    // INS机械编排到最新IMU数据
    auto imu_pre = ins_window_.back().first;
    auto state   = ins_window_.back().second;

    write_state_mutex_.lock();
    write_state_buffer_.clear();
    Lock lock(imu_buffer_mutex_);
    while (true) {
        if (imu_buffer_.empty()) {
            break;
        }

        auto imu_cur = imu_buffer_.front();
        imu_buffer_.pop();

        MISC::insMechanization(integration_parameters_, imu_pre, imu_cur, state);
        ins_window_.push_back({imu_cur, state});

        // 保存定位结果
        write_state_buffer_.push_back(state);

        imu_pre = imu_cur;

        // 总是积分到大于设定时刻
        if (imu_cur.time > time) {
            break;
        }
    }
    write_state_mutex_.unlock();

    // 写入文件
    task_group_.run([this]() {
        Lock lock(write_state_mutex_);
        for (const auto &state : write_state_buffer_) {
            writeStateResult(state);
        }
    });
}

void LVIO::addNewLidarFrameTimeNode(LidarFrame::Ptr frame, bool is_add_node) {

    // 添加时间节点
    IntegrationState state;
    if (is_add_node) {
        state = addNewTimeNode(frame->stamp(), NODE_LIDAR);
    } else {
        state = Preintegration::stateFromData(statedatalist_.back());
        LOGI << "Append lidar frame to node " << Logging::doubleData(frame->stamp());
    }

    // 加入关键帧队列
    lidar_map_->addNewKeyFrame(frame);

    // 时间偏移估计参数

    // 速度
    frame->setVelocity(state.v);

    // 角速度, 半个雷达帧间隔内的平均角速度
    Vector3d omega(0, 0, 0);
    auto preintegration    = preintegrationlist_.back();
    const auto &imu_buffer = preintegration->imuBuffer();
    size_t start_index     = static_cast<size_t>((preintegration->deltaTime() - lidar_frame_dt_ * 0.5) / imudatadt_);
    double dt              = 0;
    // 防止没有数据, 至少加入两个历元
    if (start_index >= imu_buffer.size()) {
        start_index = imu_buffer.size() - 2;
    }
    for (size_t k = start_index; k < imu_buffer.size(); k++) {
        const auto &imu = imu_buffer[k];
        omega += imu.dtheta;
        dt += imu.dt;
    }
    // 补偿零偏
    omega = omega / dt - state.bg;
    frame->setAngularVelocity(omega);
}

void LVIO::addNewVisualFrameTimeNode(VisualFrame::Ptr frame) {
    double frametime = frame->stamp();

    // 添加关键帧
    LOGI << "Insert keyframe " << frame->keyFrameId() << " at " << Logging::doubleData(frame->stamp()) << " with "
         << frame->unupdatedMappoints().size() << " new mappoints";
    visual_map_->insertKeyFrame(frame);

    addNewTimeNode(frametime, NODE_VISUAL);

    // 移除多余的预积分节点
    removeUnusedTimeNode();
}

bool LVIO::removeUnusedTimeNode() {
    if (unused_time_node_ == 0) {
        return false;
    }

    // 移除多余的节点
    int index = optimizer_->getStateDataIndex(timelist_, unused_time_node_);

    // Exception
    if (index < 0) {
        return false;
    }

    auto first_preintegration  = preintegrationlist_[index - 1];
    auto second_preintegration = preintegrationlist_[index];
    auto imu_buffer            = second_preintegration->imuBuffer();

    // 将后一个预积分的IMU数据合并到前一个, 不包括第一个IMU数据
    for (size_t k = 1; k < imu_buffer.size(); k++) {
        first_preintegration->addNewImu(imu_buffer[k]);
    }

    // 移除时间节点, 以及后一个预积分
    preintegrationlist_.erase(preintegrationlist_.begin() + index);
    timelist_.erase(timelist_.begin() + index);
    statedatalist_.erase(statedatalist_.begin() + index);

    LOGI << "Remove unused time node " << Logging::doubleData(unused_time_node_);
    unused_time_node_ = 0;

    return true;
}

IntegrationState LVIO::addNewTimeNode(double time, NodeType type) {

    vector<IMU> series;
    IntegrationState state;

    // 获取时段内用于预积分的IMU数据
    double start = timelist_.back();
    double end   = time;
    MISC::getImuSeriesFromTo(ins_window_, start, end, series);

    state = Preintegration::stateFromData(statedatalist_.back());

    // 新建立新的预积分
    {
        preintegrationlist_.emplace_back(
            Preintegration::createPreintegration(integration_parameters_, series[0], state));
    }

    // 预积分, 从第二个历元开始
    for (size_t k = 1; k < series.size(); k++) {
        preintegrationlist_.back()->addNewImu(series[k]);
    }

    // 当前状态加入到滑窗中
    state      = preintegrationlist_.back()->currentState();
    state.time = time;
    statedatalist_.emplace_back(Preintegration::stateToData(state));

    // 新的时间节点
    LOGI << "Add new " << NODE_NAME.at(type) << " time node at "
         << absl::StrFormat("%0.6lf with dt %0.3lf", time, time - timelist_.back());
    timelist_.push_back(time);

    return state;
}

void LVIO::parametersStatistic() {

    // 待输出的参数
    vector<double> parameters;

    if (is_use_visual_) {
        parameters.clear();

        // 所有关键帧
        const auto keyframeids = visual_map_->orderedKeyFrames();
        size_t size            = keyframeids.size();
        if (size < 2) {
            return;
        }
        const auto &keyframes = visual_map_->keyframes();

        // 最新的关键帧
        auto frame_cur = keyframes.at(keyframeids[size - 1]);
        auto frame_pre = keyframes.at(keyframeids[size - 2]);

        // 时间戳
        parameters.push_back(frame_cur->stamp());
        parameters.push_back(frame_cur->stamp() - frame_pre->stamp());

        // 当前关键帧与上一个关键帧的id差, 即最新关键帧的跟踪帧数
        auto frame_cnt = static_cast<double>(frame_cur->id() - frame_pre->id());
        parameters.push_back(frame_cnt);

        // 特征点数量
        parameters.push_back(static_cast<double>(frame_cur->numFeatures()));

        // 重投影误差统计
        const auto &reprojection = optimizer_->reprojectionError();
        parameters.insert(parameters.end(), reprojection.begin(), reprojection.end());

        // 迭代次数
        const auto &iteration = optimizer_->iterations();
        parameters.insert(parameters.end(), iteration.begin(), iteration.end());

        // 计算耗时
        const auto &timecost = optimizer_->timecosts();
        parameters.insert(parameters.end(), timecost.begin(), timecost.end());

        // 路标点粗差
        const auto &outliers = optimizer_->outliers();
        parameters.push_back(outliers[0]); // mappoints
        parameters.push_back(outliers[1]); // features

        // 激光深度特征统计
        int num_mappoint_total = 0;
        int num_mappoint_depth = 0;
        for (const auto &feature : frame_cur->features()) {
            auto mappoint = feature.second->getMapPoint();
            if (!mappoint || mappoint->isOutlier()) {
                continue;
            }

            num_mappoint_total++;
            if (mappoint->mapPointType() == visual::MAPPOINT_DEPTH_ASSOCIATED) {
                num_mappoint_depth++;
            }
        }

        parameters.push_back(num_mappoint_total);
        parameters.push_back(num_mappoint_depth);

        parameters.push_back(stat_visual_tracking_cost_);
        parameters.push_back(tracking_->trackingTimeCost());

        // 保存数据
        visual_stat_filesaver_->dump(parameters);
        visual_stat_filesaver_->flush();
    }

    if (is_use_lidar_) {
        parameters.clear();

        // 地图参数
        const auto &statistic = lidar_map_->statisticParameters();
        parameters.insert(parameters.end(), statistic.begin(), statistic.end());

        // 计算耗时
        const auto &timecost = optimizer_->timecosts();
        parameters.insert(parameters.end(), timecost.begin(), timecost.end());

        // 迭代次数
        const auto &iteration = optimizer_->iterations();
        parameters.insert(parameters.end(), iteration.begin(), iteration.end());

        // 粗差雷达因子
        const auto &outliers = optimizer_->outliers();
        parameters.push_back(outliers[2]);

        lidar_stat_filesaver_->dump(parameters);
        lidar_stat_filesaver_->flush();
    }
}

void LVIO::updateParametersFromOptimizer() {

    if (is_use_visual_) {
        // 更新外参
        if (optimize_estimate_cam_td_ || optimize_estimate_cam_extrinsic_) {
            optimizer_->updateCameraExtrinsic(pose_b_c_, td_b_c_);

            vector<double> extrinsic;
            Vector3d euler = Rotation::matrix2euler(pose_b_c_.R) * R2D;

            extrinsic.push_back(timelist_.back());
            extrinsic.push_back(pose_b_c_.t[0]);
            extrinsic.push_back(pose_b_c_.t[1]);
            extrinsic.push_back(pose_b_c_.t[2]);
            extrinsic.push_back(euler[0]);
            extrinsic.push_back(euler[1]);
            extrinsic.push_back(euler[2]);
            extrinsic.push_back(td_b_c_);

            cam_ext_filesaver_->dump(extrinsic);
            cam_ext_filesaver_->flush();
        }

        // 输出路标点
        const auto &points = optimizer_->newFixedVisualMappoints();
        for (const auto &point : points) {
            visual_pts_filesaver_->dump({point.x(), point.y(), point.z()});
        }
    }

    if (is_use_lidar_) {
        if (optimize_estimate_lidar_td_ || optimize_estimate_lidar_extrinsic_) {
            optimizer_->updateLidarExtrinsic(pose_b_l_, td_b_l_);

            vector<double> extrinsic;
            Vector3d euler = Rotation::matrix2euler(pose_b_l_.R) * R2D;

            extrinsic.push_back(timelist_.back());
            extrinsic.push_back(pose_b_l_.t[0]);
            extrinsic.push_back(pose_b_l_.t[1]);
            extrinsic.push_back(pose_b_l_.t[2]);
            extrinsic.push_back(euler[0]);
            extrinsic.push_back(euler[1]);
            extrinsic.push_back(euler[2]);
            extrinsic.push_back(td_b_l_);

            lidar_ext_filesaver_->dump(extrinsic);
            lidar_ext_filesaver_->flush();
        }
    }

    // 更新雷达相机外参
    if (is_use_visual_ && is_use_lidar_) {
        optimizer_->updateLidarToCameraExtrinsic(pose_c_l_);
    }

    // 输出协方差
    if (optimize_calculate_covariance_) {
        auto std = optimizer_->latestStateStd();
        if (!std.empty()) {
            for (size_t k = 0; k < 3; k++) {
                std[k + 7] *= R2D;
                std[k + 10] *= 3600 * R2D;
                std[k + 13] *= 1.0e5;
            }
            std_filesaver_->dump(std);
            std_filesaver_->flush();
        }
    }
}

bool LVIO::removeNonKeyFrame() {
    auto keyframeids = visual_map_->orderedKeyFrames();

    auto newest_id    = keyframeids.back();
    auto visual_frame = visual_map_->keyframes().find(newest_id)->second;

    // 移除非关键帧
    if (visual_frame->keyFrameState() == visual::KEYFRAME_REMOVE_SECOND_NEW) {
        LOGI << "Remove nonkeyframe at " << Logging::doubleData(visual_frame->stamp());

        unused_time_node_ = visual_frame->stamp();

        // 仅需要重置关键帧标志, 从地图中移除次新关键帧即可, 无需调整状态参数和路标点
        visual_frame->resetKeyFrame();
        visual_map_->removeKeyFrame(visual_frame, false);

        if (is_use_lidar_) {
            // 同时移除雷达帧

            auto lidar_frame = lidar_map_->keyframes().back();
            if (MISC::isTheSameTimeNode(visual_frame->stamp(), lidar_frame->stamp(), MISC::MINIMUM_TIME_INTERVAL)) {
                second_new_lidar_frame_ = lidar_frame;
                lidar_map_->removeNewestLidarFrame();
            }
        }
    }

    return true;
}

bool LVIO::lvioInitialization(double last_time, double current_time) {

    if (ins_window_.size() < imudatarate_) {
        return false;
    }

    // 缓存数据用于零速检测
    vector<IMU> imu_buff;
    for (const auto &ins : ins_window_) {
        auto &imu = ins.first;
        if ((imu.time > last_time) && (imu.time < current_time)) {
            imu_buff.push_back(imu);
        }
    }

    // 零速检测估计陀螺零偏和横滚俯仰角
    vector<double> average; // 速率形式的IMU均值
    static Vector3d bg{0, 0, 0};
    static Vector3d attitude{0, 0, 0};
    static bool is_has_zero_velocity = false;

    bool is_zero_velocity = false;
    if (is_use_zupt_) {
        is_zero_velocity = MISC::detectZeroVelocity(imu_buff, imudatarate_, average);
        if (is_zero_velocity) {
            // 陀螺零偏
            bg = Vector3d(average[0], average[1], average[2]);

            // 重力调平获取横滚俯仰角
            Vector3d fb(average[3], average[4], average[5]);

            attitude[0] = -asin(fb[1] / integration_parameters_->gravity.z());
            attitude[1] = asin(fb[0] / integration_parameters_->gravity.z());

            LOGI << "Zero velocity get gyroscope bias " << bg.transpose() * 3600 * R2D << ", roll " << attitude[0] * R2D
                 << ", pitch " << attitude[1] * R2D;
            is_has_zero_velocity = true;
        }
    }

    // 初始位置和速度, 局部坐标系原点为初始位置, 航向为0
    Vector3d position   = Vector3d::Zero();
    Vector3d velocity   = Vector3d::Zero();
    double initial_time = last_time;

    // 推算当前时刻速度, 避免错误初始化
    double current_velocity;
    {
        auto state = IntegrationState{
            .time = initial_time,
            .p    = position,
            .q    = Rotation::euler2quaternion(attitude),
            .v    = velocity,
            .bg   = bg,
            .ba   = {0, 0, 0},
        };
        MISC::redoInsMechanization(integration_parameters_, state, RESERVED_INS_NUM * 2, ins_window_);
        current_velocity = ins_window_.back().second.v.norm();
        LOGI << "Current velocity is " << current_velocity;
    }

    // 非零速且存在运动时完成初始化, 静止时无平移激励不初始化
    bool is_initialized = false;
    if (!is_zero_velocity && (current_velocity > 0.2)) {
        if (!is_has_zero_velocity) {
            // 直接重力调平获取横滚俯仰角
            Vector3d fb(average[3], average[4], average[5]);

            attitude[0] = -asin(fb[1] / integration_parameters_->gravity.z());
            attitude[1] = asin(fb[0] / integration_parameters_->gravity.z());

            LOGW << "Get roll " << attitude[0] * R2D << ", pitch " << attitude[1] * R2D << " without zero velocity";
        }

        is_initialized = true;
    }

    if (!is_initialized) {
        if (is_use_visual_) {
            // 移除无效的图像数据
            VisualFrame::Ptr frame;
            while (!is_finished_ && !visual_frame_buffer_.empty()) {
                visual_frame_buffer_mutex_.lock();
                frame = visual_frame_buffer_.front();
                visual_frame_buffer_mutex_.unlock();
                if (frame->stamp() > current_time) {
                    break;
                }
                visual_frame_buffer_.pop();
            }
        }

        if (is_use_lidar_) {
            // 初始位姿, 横滚俯仰来自重力调平
            auto state = IntegrationState{
                .p = position,
                .q = Rotation::euler2quaternion(attitude),
            };

            // 累积静态点云到当前时刻
            accumulateStationaryPointcloud(state, current_time);
        }

        // 零速状态返回
        return false;
    }

    if (is_use_lidar_ && lidar_frame_buffer_.empty()) {
        LOGW << "Lidar frame is not ready at " << Logging::doubleData(current_time);
        return false;
    }

    // 初始状态, 从上一秒开始
    auto state = IntegrationState{
        .time = initial_time,
        .p    = position,
        .q    = Rotation::euler2quaternion(attitude),
        .v    = velocity,
        .bg   = bg,
        .ba   = {0, 0, 0},
    };

    // 初始时间节点
    statedatalist_.emplace_back(Preintegration::stateToData(state));
    timelist_.push_back(initial_time);

    // 初始先验
    optimizer_->constructPrior(statedatalist_.front(), is_has_zero_velocity);

    // 计算第一秒的INS结果
    state = Preintegration::stateFromData(statedatalist_.back());
    MISC::redoInsMechanization(integration_parameters_, state, RESERVED_INS_NUM, ins_window_);
    {
        // 输出第一秒的结果
        write_state_mutex_.lock();
        write_state_buffer_.clear();
        for (size_t k = 0; k < ins_window_.size(); k++) {
            if (ins_window_[k].first.time >= initial_time) {
                write_state_buffer_.push_back(ins_window_[k].second);
            }
        }
        write_state_mutex_.unlock();

        // 写入文件
        task_group_.run([this]() {
            Lock lock(write_state_mutex_);
            for (const auto &state : write_state_buffer_) {
                writeStateResult(state);
            }
        });
    }

    start_time_ = current_time;
    LOGI << "Initialization at " << Logging::doubleData(last_time);

    // 当前时间节点
    addNewTimeNode(current_time, NODE_IMU);

    if (is_use_visual_) {
        // 同步图像数据, 必须有图像数据

        // 等待图像数据有效
        VisualFrame::Ptr frame;
        while (!is_finished_) {
            visual_frame_buffer_mutex_.lock();
            frame = visual_frame_buffer_.back();
            visual_frame_buffer_mutex_.unlock();
            // 确保当前节点与最新视觉节点大于10ms, 需考虑时延
            double image_stamp = frame->stamp() + td_b_c_;
            if ((image_stamp - current_time) > 0.01) {
                break;
            }

            usleep(WAITING_DELAY_IN_US);
        }

        // 同步到当前时刻
        while (!is_finished_) {
            visual_frame_buffer_mutex_.lock();
            frame = visual_frame_buffer_.front();
            visual_frame_buffer_mutex_.unlock();
            double image_stamp = frame->stamp() + td_b_c_;
            if ((image_stamp - current_time) > 0.01) {
                break;
            }
            visual_frame_buffer_.pop();
        }
    }

    // 初始点云帧
    if (is_use_lidar_) {
        // 全部投影到当前时刻
        accumulateDynamicPointcloud(current_time);

        // 可视化地图点云
        if (is_use_visualization_) {
            lidar_viewer_->updateMapPointCloud(lidar_map_->keyframes().back()->pointCloudWorld());
        }
    }

    return true;
}

void LVIO::accumulateStationaryPointcloud(const IntegrationState &state, double end_time) {
    vector<LidarFrame::Ptr> lidar_frames;
    {
        Lock lock(lidar_frame_buffer_mutex_);
        while (!lidar_frame_buffer_.empty()) {
            auto frame = lidar_frame_buffer_.front();

            if ((frame->endTime() + td_b_l_) < end_time) {
                // 取出雷达帧
                lidar_frame_buffer_.pop();

                lidar_frames.push_back(frame);
            } else {
                break;
            }
        }
    }

    if (lidar_frames.empty()) {
        return;
    }

    for (auto &frame : lidar_frames) {
        // 更新时标
        frame->setTimeDelay(td_b_l_);
        frame->setStamp(frame->stamp() + td_b_l_);

        // 降采样投影
        lidar_map_->lidarFrameProcessing(state, pose_b_l_, frame);
    }

    // 可视化最新一帧的点云
    if (is_use_visualization_) {
        lidar_viewer_->updateCurrentPointCloud(lidar_frames.back()->pointCloudWorld());
    }
}

bool LVIO::accumulateDynamicPointcloud(double end_time) {
    // 确保雷达数据有效
    LidarFrame::Ptr latest_frame;
    int wait_cnts = 0;
    while (!is_finished_) {
        lidar_frame_buffer_mutex_.lock();
        latest_frame = lidar_frame_buffer_.back();
        lidar_frame_buffer_mutex_.unlock();
        if (latest_frame->stamp() > end_time) {
            break;
        }

        usleep(WAITING_DELAY_IN_US);

        // 处理可能导致的程序不自动退出, 超时时间为0.2s
        wait_cnts++;
        if (wait_cnts > 2000) {
            break;
        }
    }

    vector<LidarFrame::Ptr> lidar_frames;
    if (second_new_lidar_frame_) {
        // 已被移除的点云帧
        lidar_frames.push_back(second_new_lidar_frame_);
        second_new_lidar_frame_ = nullptr;
    }
    {
        Lock lock(lidar_frame_buffer_mutex_);
        while (!lidar_frame_buffer_.empty()) {
            auto frame = lidar_frame_buffer_.front();

            // 所有时间小于当前时刻的帧
            if ((frame->endTime() + td_b_l_) < end_time) {
                // 取出雷达帧
                lidar_frame_buffer_.pop();

                lidar_frames.push_back(frame);
            } else {
                break;
            }
        }
    }

    // 无有效雷达帧
    if (lidar_frames.empty()) {
        return false;
    }

    // 取最新的雷达帧投影作为关键帧, 其他帧合并
    auto keyframe = lidar_frames.back();
    // 关键帧预处理, 投影到当前时刻
    keyframe->setTimeDelay(td_b_l_);
    keyframe->setStamp(end_time);
    lidar_map_->lidarFrameProcessing(ins_window_, pose_b_l_, keyframe);

    if (!lidar_map_->keyframes().empty()) {
        // 初始处理无需累积点云
        auto last_keyframe = lidar_map_->keyframes().back();

        vector<LidarFrame::Ptr> non_keyframes;
        if ((last_keyframe->numMergedFrames() < 5) && (lidar_frames.size() > 1)) {
            non_keyframes.resize(lidar_frames.size() - 1);
            tbb::parallel_for(size_t(0), lidar_frames.size() - 1, [&, lidar_frames](const size_t &k) {
                auto frame = lidar_frames[k];

                // 更新时标
                frame->setTimeDelay(td_b_l_);
                frame->setStamp(frame->stamp() + td_b_l_);
                lidar_map_->lidarFrameProcessing(ins_window_, pose_b_l_, frame);

                non_keyframes[k] = frame;
            });

            // 合并非关键帧
            lidar_map_->mergeNonKeyframes(non_keyframes, last_keyframe);
        }

        LOGI << "Accumulate " << non_keyframes.size() << " frames to " << Logging::doubleData(last_keyframe->stamp())
             << " with " << last_keyframe->pointCloudMap()->size() << " points";
    }

    // 加入地图
    addNewLidarFrameTimeNode(keyframe, false);

    // 更新点云特征
    if (lidar_map_->keyframes().size() > 1) {
        lidar_map_->updateLidarFeaturesInMap();
    }

    return true;
}

bool LVIO::interpolatePixelColor(const cv::Mat &image, const cv::Point2f &pixel, vector<float> &rgb) {
    float u_ref = pixel.x;
    float v_ref = pixel.y;
    int u_ref_i = floorf(u_ref);
    int v_ref_i = floorf(v_ref);

    float subpix_u_ref = (u_ref - u_ref_i);
    float subpix_v_ref = (v_ref - v_ref_i);

    float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
    float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
    float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
    float w_ref_br = subpix_u_ref * subpix_v_ref;

    int width        = image.cols;
    uint8_t *img_ptr = (uint8_t *) image.data + (v_ref_i * width + u_ref_i) * 3;

    rgb.resize(3);
    rgb[2] = w_ref_tl * img_ptr[0] + w_ref_tr * img_ptr[0 + 3] + w_ref_bl * img_ptr[width * 3] +
             w_ref_br * img_ptr[width * 3 + 0 + 3];
    rgb[1] = w_ref_tl * img_ptr[1] + w_ref_tr * img_ptr[1 + 3] + w_ref_bl * img_ptr[1 + width * 3] +
             w_ref_br * img_ptr[width * 3 + 1 + 3];
    rgb[0] = w_ref_tl * img_ptr[2] + w_ref_tr * img_ptr[2 + 3] + w_ref_bl * img_ptr[2 + width * 3] +
             w_ref_br * img_ptr[width * 3 + 2 + 3];

    return true;
}

void LVIO::writeStateResult(const IntegrationState &state) {

    // 保存结果
    vector<double> result;

    double time = state.time;
    Vector3d bg = state.bg * R2D * 3600;
    Vector3d ba = state.ba * 1e5;

    {
        // imu error file
        result.clear();

        result.push_back(time);
        result.push_back(bg[0]);
        result.push_back(bg[1]);
        result.push_back(bg[2]);
        result.push_back(ba[0]);
        result.push_back(ba[1]);
        result.push_back(ba[2]);

        imu_err_filesaver_->dump(result);
        imu_err_filesaver_->flush();
    }

    {
        // trajectory
        result.clear();

        result.push_back(time);
        result.push_back(state.p[0]);
        result.push_back(state.p[1]);
        result.push_back(state.p[2]);
        result.push_back(state.q.x());
        result.push_back(state.q.y());
        result.push_back(state.q.z());
        result.push_back(state.q.w());
        traj_filesaver_->dump(result);
        traj_filesaver_->flush();
    }
}