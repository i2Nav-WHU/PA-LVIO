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

#include "fusion.h"
#include "lidar/lidar_viewer_rviz.h"
#include "visual/visual_drawer_rviz.h"

#include "common/logging.h"
#include "common/misc.h"
#include "lidar/lidar_frame.h"
#include "visual/visual_frame.h"

#include <absl/strings/str_format.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>

#include <filesystem>
#include <sensor_msgs/image_encodings.h>
#include <yaml-cpp/yaml.h>

#include <set>

#include <rosbag/bag.h>
#include <rosbag/view.h>

std::atomic<bool> global_finished = false;

void Fusion::setFinished() {
    if (lvio_) {
        lvio_->setFinished();
    }
}

void Fusion::run(const string &config_file, bool logtostderr) {
    // ROS Handle
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // 加载配置
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_file);
    } catch (YAML::Exception &exception) {
        std::cout << "Failed to open configuration file " << config_file << std::endl;
        return;
    }
    auto outputpath        = config["outputpath"].as<string>();
    auto is_make_outputdir = config["is_make_outputdir"].as<bool>();

    // 如果文件夹不存在, 尝试创建
    if (!std::filesystem::is_directory(outputpath)) {
        std::filesystem::create_directory(outputpath);
    }
    if (!std::filesystem::is_directory(outputpath)) {
        std::cout << "Failed to open outputpath" << std::endl;
        return;
    }

    if (is_make_outputdir) {
        absl::CivilSecond cs = absl::ToCivilSecond(absl::Now(), absl::LocalTimeZone());
        absl::StrAppendFormat(&outputpath, "/T%04d%02d%02d%02d%02d%02d", cs.year(), cs.month(), cs.day(), cs.hour(),
                              cs.minute(), cs.second());
        std::filesystem::create_directory(outputpath);
    }
    string log_file = outputpath + "/lvio.log";
    Logging::initialization(logtostderr, true, log_file);

    // message topic
    string imu_topic, image_topic, lidar_topic;
    imu_topic   = config["ros"]["imu_topic"].as<string>();
    image_topic = config["ros"]["image_topic"].as<string>();
    lidar_topic = config["ros"]["lidar_topic"].as<string>();
    if (config["ros"]["use_compressed_image"]) {
        use_compressed_image_ = config["ros"]["use_compressed_image"].as<bool>();
    }
    if (config["ros"]["resize_image_to_half"]) {
        resize_image_to_half_ = config["ros"]["resize_image_to_half"].as<bool>();
    }

    // 读取ROS包
    bool is_read_bag = config["ros"]["is_read_bag"].as<bool>();
    string bag_file  = config["ros"]["bag_file"].as<string>();

    // ROS具有高优先级, 重置配置文件的ROS读取配置
    bool is_read_bag_ros;
    if (pnh.param<bool>("is_read_bag", is_read_bag_ros, false)) {
        is_read_bag = is_read_bag_ros;
        string bag_file_ros;
        pnh.param<string>("bagfile", bag_file_ros, "");
        if (bag_file_ros.compare("") != 0) {
            bag_file = bag_file_ros;
        }
    }

    double imu_data_rate = config["imu"]["imudatarate"].as<double>();
    imu_data_dt_         = 1.0 / imu_data_rate;

    is_use_visual_      = config["visual"]["is_use_visual"].as<bool>();
    is_use_lidar_depth_ = config["visual"]["is_use_lidar_depth"].as<bool>();

    // LiDAR参数
    is_use_lidar_            = config["lidar"]["is_use_lidar"].as<bool>();
    lidar_type_              = config["lidar"]["lidar_type"].as<int>();
    int scan_line            = config["lidar"]["scan_line"].as<int>();
    double nearest_distance  = config["lidar"]["nearest_distance"].as<double>();
    double farthest_distance = config["lidar"]["farthest_distance"].as<double>();
    double frame_rate        = config["lidar"]["frame_rate"].as<double>();

    if (!is_use_visual_ && !is_use_lidar_) {
        LOGE << "Unsupported configuration without visual and lidar";
        return;
    }

    if (is_use_lidar_ || (is_use_visual_ && is_use_lidar_depth_)) {
        // lidar数据转换对象
        lidar_converter_ =
            std::make_shared<LidarConverter>(frame_rate, scan_line, 1, nearest_distance, farthest_distance);
    }

    // ROS传感器数据
    LOGI << "Process ROS IMU messages";
    if (is_use_visual_) {
        LOGI << "Process camera messages";
    }
    if (is_use_lidar_ || (is_use_visual_ && is_use_lidar_depth_)) {
        if (lidar_type_ == Livox) {
            LOGI << "Process livox lidar messages";
        } else if (lidar_type_ == Velodyne) {
            LOGI << "Process velodyne lidar messages";
        } else if (lidar_type_ == Ouster) {
            LOGI << "Process ouster lidar messages";
        } else if (lidar_type_ == Hesai) {
            LOGI << "Process hesai lidar messages";
        } else {
            LOGE << "Unsupported lidar type";
            return;
        }
    }

    // 创建LVIO
    VisualDrawerRviz::Ptr visual_drawer = nullptr;
    LidarViewerRviz::Ptr lidar_viewer   = nullptr;
    bool is_use_visualization_          = config["is_use_visualization"].as<bool>();
    if (is_use_visualization_) {
        if (is_use_visual_) {
            visual_drawer = std::make_shared<VisualDrawerRviz>(nh);
        }
        if (is_use_lidar_) {
            lidar_viewer = std::make_shared<LidarViewerRviz>(nh);
        }
    }
    lvio_ = std::make_shared<LVIO>(config_file, outputpath, visual_drawer, lidar_viewer);

    // check is initialized
    if (!lvio_->isRunning()) {
        LOGE << "Fusion ROS terminate";
        return;
    }

    // 处理ROS数据
    if (is_read_bag) {
        LOGI << "Start to read ROS bag file";
        processRead(imu_topic, image_topic, lidar_topic, bag_file);
        LOGI << "Finish to read ROS bag file";

        // 结束处理
        global_finished = true;
    } else {
        processSubscribe(imu_topic, image_topic, lidar_topic, nh);
    }
}

void Fusion::processSubscribe(const string &imu_topic, const string &image_topic, const string &lidar_topic,
                              ros::NodeHandle &nh) {
    // IMU
    ros::Subscriber imu_sub;
    imu_sub = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200, &Fusion::imuCallback, this);

    // Visual image
    ros::Subscriber image_sub;
    if (is_use_visual_) {
        if (use_compressed_image_) {
            image_sub = nh.subscribe<sensor_msgs::CompressedImage>(image_topic, 20, &Fusion::imageCallback, this);
        } else {
            image_sub = nh.subscribe<sensor_msgs::Image>(image_topic, 20, &Fusion::imageCallback, this);
        }
    }

    // Livox lidar
    ros::Subscriber lidar_sub;
    if (is_use_lidar_ || (is_use_visual_ && is_use_lidar_depth_)) {
        if (lidar_type_ == Livox) {
            lidar_sub = nh.subscribe<livox_ros_driver::CustomMsg>(lidar_topic, 10, &Fusion::livoxCallback, this);
        } else if ((lidar_type_ == Velodyne) || (lidar_type_ == Ouster) || (lidar_type_ == Hesai)) {
            lidar_sub = nh.subscribe<sensor_msgs::PointCloud2>(lidar_topic, 10, &Fusion::pointCloudCallback, this);
        }
    }

    LOGI << "Waiting ROS message...";

    // Enter message loopback
    ros::spin();
}

void Fusion::processRead(const string &imu_topic, const string &image_topic, const string &lidar_topic,
                         const string &bagfile) {
    // 消息列表
    vector<string> topics;
    topics.emplace_back(imu_topic);
    topics.emplace_back(image_topic);
    topics.emplace_back(lidar_topic);

    // 遍历ROS包
    rosbag::Bag bag(bagfile);
    rosbag::View view(bag, rosbag::TopicQuery(topics));
    bool is_read_lidar = is_use_lidar_ || (is_use_visual_ && is_use_lidar_depth_);

    LOGI << "Read bag with topics: " << imu_topic << ", " << image_topic << ", " << lidar_topic;
    for (rosbag::MessageInstance const &msg : view) {
        // 等待数据处理完毕
        while (!global_finished && !lvio_->canAddData()) {
            usleep(100);
        }

        // 强制退出信号
        if (global_finished) {
            return;
        }

        // 按数据包内消息的实际类型分发, 避免配置与数据包类型不一致时实例化为空
        if (msg.getTopic() == imu_topic) {
            auto imumsg = msg.instantiate<sensor_msgs::Imu>();
            imuCallback(imumsg);
        } else if (is_use_visual_ && (msg.getTopic() == image_topic)) {
            if (use_compressed_image_) {
                sensor_msgs::CompressedImagePtr imagemsg = msg.instantiate<sensor_msgs::CompressedImage>();
                imageCallback(imagemsg);
            } else {
                sensor_msgs::ImageConstPtr imagemsg = msg.instantiate<sensor_msgs::Image>();
                imageCallback(imagemsg);
            }
        } else if (is_read_lidar && (msg.getTopic() == lidar_topic)) {
            if (lidar_type_ == Livox) {
                livox_ros_driver::CustomMsgConstPtr livox_ptr = msg.instantiate<livox_ros_driver::CustomMsg>();
                livoxCallback(livox_ptr);
            } else if ((lidar_type_ == Velodyne) || (lidar_type_ == Ouster) || (lidar_type_ == Hesai)) {
                sensor_msgs::PointCloud2ConstPtr points_ptr = msg.instantiate<sensor_msgs::PointCloud2>();
                pointCloudCallback(points_ptr);
            }
        }
    }

    // 等待数据处理结束
    int sec_cnts = 0;
    while (!lvio_->isBufferEmpty()) {
        sleep(1);
        if (sec_cnts++ > 20) {
            LOGW << "Waiting lvio processing timeout";
            break;
        }
    }
}

void Fusion::imuCallback(const sensor_msgs::ImuConstPtr &imumsg) {
    imu_pre_ = imu_;

    // Time
    imu_.time = imumsg->header.stamp.toSec();
    // delta time
    imu_.dt = imu_.time - imu_pre_.time;

    // IMU measurements, Front-Right-Down
    imu_.dtheta[0] = imumsg->angular_velocity.x * imu_.dt;
    imu_.dtheta[1] = imumsg->angular_velocity.y * imu_.dt;
    imu_.dtheta[2] = imumsg->angular_velocity.z * imu_.dt;
    imu_.dvel[0]   = imumsg->linear_acceleration.x * imu_.dt;
    imu_.dvel[1]   = imumsg->linear_acceleration.y * imu_.dt;
    imu_.dvel[2]   = imumsg->linear_acceleration.z * imu_.dt;

    // 数据未准备好
    if (imu_pre_.time == 0) {
        return;
    }

    addImuData(imu_);
}

void Fusion::addImuData(const IMU &imu) {
    imu_buffer_.push(imu);
    while (!imu_buffer_.empty()) {
        auto temp = imu_buffer_.front();

        // add new IMU
        if (lvio_->addNewImu(temp)) {
            imu_buffer_.pop();
        } else {
            // thread lock failed, try next time
            break;
        }
    }
}

void Fusion::imageCallback(const sensor_msgs::CompressedImageConstPtr &imagemsg) {
    cv::Mat image;

    vector<uint8_t> buffer(imagemsg->data);
    image = cv::imdecode(buffer, cv::IMREAD_COLOR);

    if (resize_image_to_half_) {
        cv::resize(image, image, image.size() / 2);
    }

    // Add new Image to LVIO
    auto latest_frame = visual::VisualFrame::createFrame(imagemsg->header.stamp.toSec(), image);

    visual_frame_buffer_.push(latest_frame);
    while (!visual_frame_buffer_.empty()) {
        auto frame = visual_frame_buffer_.front();
        if (lvio_->addNewVisualFrame(frame)) {
            visual_frame_buffer_.pop();
        } else {
            break;
        }
    }
}

void Fusion::imageCallback(const sensor_msgs::ImageConstPtr &imagemsg) {
    cv::Mat image;

    // 构造图像
    if (imagemsg->encoding == sensor_msgs::image_encodings::MONO8) {
        // Gray
        image = cv::Mat(static_cast<int>(imagemsg->height), static_cast<int>(imagemsg->width), CV_8UC1);
        memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width);
    } else if (imagemsg->encoding == sensor_msgs::image_encodings::BGR8) {
        // BGR8

        // OpenCV color format is RGB
        image = cv::Mat(static_cast<int>(imagemsg->height), static_cast<int>(imagemsg->width), CV_8UC3);
        memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width * 3);
    } else if (imagemsg->encoding == sensor_msgs::image_encodings::BAYER_RGGB8) {
        // BayerBG8
        image = cv::Mat(static_cast<int>(imagemsg->height), static_cast<int>(imagemsg->width), CV_8UC1);
        memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width);

        // OpenCV color format is RGB
        cv::cvtColor(image, image, cv::COLOR_BayerBG2RGB);
    } else if (imagemsg->encoding == sensor_msgs::image_encodings::BAYER_BGGR8) {
        // BayerRG8
        image = cv::Mat(static_cast<int>(imagemsg->height), static_cast<int>(imagemsg->width), CV_8UC1);
        memcpy(image.data, imagemsg->data.data(), imagemsg->height * imagemsg->width);

        // OpenCV color format is RGB
        cv::cvtColor(image, image, cv::COLOR_BayerRG2RGB);
    }

    if (resize_image_to_half_) {
        cv::resize(image, image, image.size() / 2);
    }

    // Add new Image to LVIO
    auto latest_frame = visual::VisualFrame::createFrame(imagemsg->header.stamp.toSec(), image);

    visual_frame_buffer_.push(latest_frame);
    while (!visual_frame_buffer_.empty()) {
        auto frame = visual_frame_buffer_.front();
        if (lvio_->addNewVisualFrame(frame)) {
            visual_frame_buffer_.pop();
        } else {
            break;
        }
    }

    LOG_EVERY_N(INFO, 20) << "Raw data time " << Logging::doubleData(imu_.time) << ", "
                          << Logging::doubleData(latest_frame->stamp());
}

void Fusion::livoxCallback(const livox_ros_driver::CustomMsgConstPtr &lidarmsg) {
    PointCloudCustomPtr pointcloud_ds  = PointCloudCustomPtr(new PointCloudCustom);
    PointCloudCustomPtr pointcloud_raw = PointCloudCustomPtr(new PointCloudCustom);
    double start, end;

    lidar_converter_->livoxPointCloudConvertion(lidarmsg, pointcloud_raw, pointcloud_ds, start, end);

    // 无有效点云
    if (pointcloud_ds->points.empty()) {
        return;
    }

    if (is_use_lidar_) {
        // Create lidar frame
        auto frame = LidarFrame::createFrame(pointcloud_ds->points.front().time, pointcloud_ds->points.back().time,
                                             pointcloud_ds);
        lvio_->addNewLidarFrame(frame);
    }
    if (is_use_visual_ && is_use_lidar_depth_) {
        // 激光深度增强点云
        lvio_->addNewPointCloud(pointcloud_raw);
    }
}

void Fusion::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr &lidarmsg) {
    PointCloudCustomPtr pointcloud = PointCloudCustomPtr(new PointCloudCustom);
    double start = 0, end = 0;

    // Velodyne自定义格式
    if (lidar_type_ == Velodyne) {
        lidar_converter_->velodynePointCloudConvertion(lidarmsg, pointcloud, start, end);
    } else if (lidar_type_ == Ouster) {
        lidar_converter_->ousterPointCloudConvertion(lidarmsg, pointcloud, start, end);
    } else if (lidar_type_ == Hesai) {
        lidar_converter_->hesaiPointCloudConvertion(lidarmsg, pointcloud, start, end);
    }

    if (is_use_lidar_) {
        // Create lidar frame
        auto frame = LidarFrame::createFrame(start, end, pointcloud);
        lvio_->addNewLidarFrame(frame);
    }
    if (is_use_visual_ && is_use_lidar_depth_) {
        // 激光深度增强点云
        lvio_->addNewPointCloud(pointcloud);
    }
}
