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

#ifndef DEPTH_FACTOR_H
#define DEPTH_FACTOR_H

#include "common/rotation.h"

#include <ceres/ceres.h>

class DepthFactor {

public:
    DepthFactor(double depth, double std)
        : depth_(depth)
        , std_(std) {
    }

    static ceres::CostFunction *create(double depth, double std) {
        return (new ceres::AutoDiffCostFunction<DepthFactor, 1, 1>(new DepthFactor(depth, std)));
    }

    template <class T> bool operator()(const T *const invdepth, T *residuals) const {
        residuals[0] = (T(depth_) - T(1.0) / invdepth[0]) / T(std_);

        return true;
    }

private:
    double depth_, std_;
};

class PoseOnlyDepthFactor : public ceres::SizedCostFunction<1, 7, 7, 7, 1, 1> {

public:
    PoseOnlyDepthFactor() = delete;
    PoseOnlyDepthFactor(Vector3d pts0, Vector3d pts2, Vector3d vel0, Vector3d vel2, double td0, double td2,
                        double depth, double depth_std);

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override;

private:
    Vector3d pts0_, pts2_;
    Vector3d vel0_, vel2_;
    double td0_, td2_;

    double depth_;
    double sqrt_info_depth_;
};

#endif // DEPTH_FACTOR_H
