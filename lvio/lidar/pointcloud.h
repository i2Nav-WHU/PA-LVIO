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

#ifndef LIDAR_POINTCLOUD_H
#define LIDAR_POINTCLOUD_H

#include "common/types.h"
#include "lidar/lidar.h"

namespace lidar {

using std::vector;

class PointCloudCommon {

public:
    // 点云相对投影, 原始点云在l1系
    static PointType relativeProjection(const Pose &pose0, const Pose &pose1, const PointType &point_l1);
    static PointType relativeProjectionToCamera(const Pose &pose_c_l, const Pose &pose_cam0, const Pose &pose_cam1,
                                                const PointType &point_lidar);

    static void pointCloudProjection(const Pose &pose, PointCloudPtr src, PointCloudPtr dst);

    static bool planeFittingWithDistance(const vector<Vector3d> &points, double threshold, Vector3d &normal_vector,
                                         Vector3d &center, double &variance, vector<double> &distance);
    static void planeFitting(const vector<Vector3d> &points, Vector3d &normal_vector, Vector3d &center,
                             double &variance);
    static void planeFittingWithJacobian(const vector<Vector3d> &points, Vector3d &normal_vector, Vector3d &center,
                                         double &variance, vector<Matrix3d> &jaco_n_p);

    // 点云投影
    static PointType pointProjection(const Pose &pose, const PointType &point);

    static double pointSquareDistance(const PointType &point0, const PointType &point1);

    static void pointVectorConversion(const PointVector &point_vector, vector<Vector3d> &points);

    static double pointToPlaneStd(const PointVector &nearest, const PointType &target, double point_std);

private:
    static void planeFitting(const vector<Vector3d> &points, Vector3d &normal_vector, Vector3d &center,
                             double &variance, Matrix3d &evecs_real, Vector3d &evals_real, int &evals_min);
};

} // namespace lidar

#endif // LIDAR_POINTCLOUD_H
