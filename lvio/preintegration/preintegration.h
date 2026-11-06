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

#ifndef PREINTEGRATION_H
#define PREINTEGRATION_H

#include "preintegration/preintegration_base.h"
#include "preintegration/preintegration_normal.h"

class Preintegration {

public:
    Preintegration() = default;

    static std::shared_ptr<PreintegrationBase>
        createPreintegration(const std::shared_ptr<const IntegrationParameters> &parameters, const IMU &imu0,
                             const IntegrationState &state) {
        return std::make_shared<PreintegrationNormal>(parameters, imu0, state);
    }

    static int numPoseParameter() {
        return PreintegrationBase::NUM_POSE;
    }

    static int numMixParameter() {
        return PreintegrationNormal::NUM_MIX;
    }

    static IntegrationStateData stateToData(const IntegrationState &state) {
        return PreintegrationNormal::stateToData(state);
    }

    static IntegrationState stateFromData(const IntegrationStateData &data) {
        return PreintegrationNormal::stateFromData(data);
    }
};

#endif // PREINTEGRATION_H
