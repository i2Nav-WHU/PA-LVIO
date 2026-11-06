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

#ifndef LIDAR_FACTOR_H
#define LIDAR_FACTOR_H

#include "common/types.h"
#include "lidar/lidar.h"

#include <ceres/ceres.h>

using std::vector;

class LidarBAFactor : public ceres::CostFunction {

public:
    LidarBAFactor(vector<Vector3d> points_lidar, vector<Vector3d> vels, vector<Vector3d> omegas, vector<double> tds,
                  double std);
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override;

private:
    vector<Vector3d> points_lidar_;
    size_t points_size_;
    double num_n_, num_n_inverse_;

    vector<Vector3d> vels_;
    vector<Vector3d> omegas_;
    vector<double> tds_;

    double sqrt_info_;
};

class LidarFactor : public ceres::SizedCostFunction<1, 7, 7, 7, 1> {

public:
    LidarFactor(const PointType &point_lidar, Vector3d normal_vector, Vector3d center, double std, Vector3d vel0,
                Vector3d omega0, double td0, Vector3d vel1, Vector3d omega1, double td1);

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override;

private:
    Vector3d normal_vector_;
    Vector3d center_;

    Vector3d point_lidar_;
    double sqrt_info_;

    Vector3d vel0_, vel1_;
    Vector3d omega0_, omega1_;
    double td0_, td1_;
};

class LidarFrameToMapFactor : public ceres::SizedCostFunction<1, 7> {

public:
    LidarFrameToMapFactor(const PointType &point_lidar, Vector3d normal_vector, Vector3d center, double std);
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override;

private:
    Vector3d normal_vector_;
    Vector3d center_;

    Vector3d point_lidar_;
    double sqrt_info_;
};

#endif // LIDAR_FACTOR_H
