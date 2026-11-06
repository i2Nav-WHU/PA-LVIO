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

#include "preintegration/preintegration_normal.h"

#include <ceres/ceres.h>

class ImuMixPriorFactor : public ceres::CostFunction {

public:
    ImuMixPriorFactor(const double *mix, const double *mix_std) {

        memcpy(mix_, mix, sizeof(double) * 9);
        memcpy(mix_std_, mix_std, sizeof(double) * 9);

        *mutable_parameter_block_sizes() = vector<int>{PreintegrationNormal::NUM_MIX};
        set_num_residuals(PreintegrationNormal::NUM_MIX);
    }

    bool Evaluate(const double *const *parameters, double *residuals, double **jacobians) const override {

        // parameters: vel[3], bg[3], ba[3]

        for (size_t k = 0; k < 9; k++) {
            residuals[k] = (parameters[0][k] - mix_[k]) / mix_std_[k];
        }

        if (jacobians && jacobians[0]) {
            Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> jaco(jacobians[0]);
            jaco.setZero();
            for (size_t k = 0; k < 9; k++) {
                jaco(k, k) = 1.0 / mix_std_[k];
            }
        }

        return true;
    }

private:
    double mix_[9], mix_std_[9];
};
