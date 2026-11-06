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

#ifndef INTEGRATION_DEFINE_H
#define INTEGRATION_DEFINE_H

#include <Eigen/Geometry>
#include <vector>

using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::vector;

typedef struct IntegrationState {
    double time;

    Vector3d p{0, 0, 0};
    Quaterniond q{0, 0, 0, 0};
    Vector3d v{0, 0, 0};

    Vector3d bg{0, 0, 0};
    Vector3d ba{0, 0, 0};
} IntegrationState;

typedef struct IntegrationStateData {
    double time;

    double pose[7]; // pose : 3 + 4 = 7

    // mix parameters
    // vel + bias : 3 + 6 = 9
    double mix[9];
} IntegrationStateData;

typedef struct IntegrationParameters {
    // IMU噪声为白噪声, 积分为随机游走, 即VRW和ARW
    // 零偏建模为一阶高斯马尔卡夫过程, 参数为标准差及相关时间

    double acc_vrw;      // 速度随机游走, m / s^1.5
    double gyr_arw;      // 角度随机游走, rad / s^0.5
    double gyr_bias_std; // 陀螺零偏标准差, rad / s
    double acc_bias_std; // 加表零偏标准差, m / s^2
    double corr_time;    // 相关时间, s

    Vector3d gravity; // 当地重力矢量, m / s^2
} IntegrationParameters;

#endif // INTEGRATION_DEFINE_H
