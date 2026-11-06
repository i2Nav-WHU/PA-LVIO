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

#ifndef LIDAR_FEATURE_H
#define LIDAR_FEATURE_H

#include "lidar/pointcloud.h"

#include <Eigen/Geometry>
#include <memory>

namespace lidar {

using Eigen::Vector3d;

enum FeatureType {
    FEATURE_NONE     = -1,
    FEATURE_PLANE    = 0,
    FEATURE_PLANE_BA = 1,
};

class LidarFeature {

public:
    typedef std::shared_ptr<LidarFeature> Ptr;

    LidarFeature() = default;
    LidarFeature(const vector<Vector3d> &nearest_points, const vector<double> &point_to_plane_dis,
                 const Vector3d &normal_vector, const Vector3d &center, double variance, FeatureType type) {
        updateFeature(nearest_points, point_to_plane_dis, normal_vector, center, variance, type, false);
    }

    static Ptr createFeature(const vector<Vector3d> &nearest_points, const vector<double> &point_to_plane_dis,
                             const Vector3d &normal_vector, const Vector3d &center, double variance, FeatureType type) {
        return std::make_shared<LidarFeature>(nearest_points, point_to_plane_dis, normal_vector, center, variance,
                                              type);
    }

    FeatureType featureType() {
        return type_;
    }

    void setFeatureType(FeatureType type) {
        type_ = type;
    }

    void updateFeature(vector<Vector3d> nearest_points, vector<double> point_to_plane_dis, Vector3d normal_vector,
                       Vector3d center, double variance, FeatureType type, bool is_outlier) {
        nearest_points_     = std::move(nearest_points);
        point_to_plane_dis_ = std::move(point_to_plane_dis);
        normal_vector_      = std::move(normal_vector);
        center_             = std::move(center);
        variance_           = variance;
        type_               = type;
        is_outlier_         = is_outlier;
    }

    const vector<Vector3d> &nearestPoints() {
        return nearest_points_;
    }

    const Vector3d &normalVector() {
        return normal_vector_;
    }

    double variance() const {
        return variance_;
    }

    void setOutlier() {
        is_outlier_ = true;
    }

    bool isOutlier() const {
        return is_outlier_;
    }

    const Vector3d &center() {
        return center_;
    }

    void setPointStd(double std) {
        point_std_ = std;
    }

    double pointStd() {
        return point_std_;
    }

private:
    vector<Vector3d> nearest_points_;
    vector<double> point_to_plane_dis_;
    Vector3d center_;
    Vector3d normal_vector_;
    double point_std_;
    double variance_;

    bool is_outlier_{false};
    FeatureType type_{FEATURE_NONE};
};

} // namespace lidar

#endif // LIDAR_FEATURE_H