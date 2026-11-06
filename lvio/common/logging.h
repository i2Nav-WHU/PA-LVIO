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

#ifndef LOGGING_H
#define LOGGING_H

#include <Eigen/Geometry>
#include <absl/log/log.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

using std::string;

#define LOGI LOG(INFO)
#define LOGW LOG(WARNING)
#define LOGE LOG(ERROR)
#define LOGF LOG(FATAL)

class Logging {

public:
    static void initialization(bool logtostderr = true, bool logtofile = false, const string &log_file = "log.log");
    static void shutdownLogging();

    template <typename T, int Rows, int Cols>
    static void printMatrix(const Eigen::Matrix<T, Rows, Cols> &matrix, const string &prefix = "Matrix: ") {
        std::cout << prefix << matrix.rows() << "x" << matrix.cols() << std::endl;
        if (matrix.cols() == 1) {
            std::cout << matrix.transpose() << std::endl;
        } else {
            std::cout << matrix << std::endl;
        }
    }

    static string doubleData(double data);

    template <typename... Args> static string sequenceData(const Args &...args) {
        std::ostringstream ss;

        printSequence(ss, args...);

        return ss.str();
    }

private:
    template <typename T> static void printSequence(std::ostringstream &ss, const T &t) {
        ss << t;
    }

    template <typename T, typename... Args>
    static void printSequence(std::ostringstream &ss, const T &t, const Args &...args) {
        ss << t << ", ";
        printSequence(ss, args...);
    }
};

#endif // LOGGING_H
