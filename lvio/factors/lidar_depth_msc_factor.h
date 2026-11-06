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

#ifndef LIDAR_DEPTH_MSC_FACTOR_H
#define LIDAR_DEPTH_MSC_FACTOR_H

#include "common/rotation.h"

#include <ceres/ceres.h>

using std::vector;

class LiDARDepthMscFactor : public ceres::CostFunction {

public:
    LiDARDepthMscFactor(const vector<std::shared_ptr<ceres::CostFunction>> costs, int num_frames,
                        double mappoint_depth);
    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override;

private:
    vector<std::shared_ptr<ceres::CostFunction>> costs_;
    int num_frames_;
    double mappoint_depth_;
};

#endif // LIDAR_DEPTH_MSC_FACTOR_H