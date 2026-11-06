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

#include "common/logging.h"

#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log_entry.h>
#include <absl/log/log_sink.h>
#include <absl/log/log_sink_registry.h>
#include <absl/strings/str_format.h>
#include <absl/time/clock.h>

class DualLogSink : public absl::LogSink {
public:
    explicit DualLogSink(bool logtostderr = true, bool logtofile = false, const string &log_file = "log.log")
        : logtostderr_(logtostderr)
        , logtofile_(logtofile) {
        // 创建日志文件
        if (logtofile_) {
            log_file_.open(log_file);
            if (!log_file_.is_open()) {
                std::cerr << "Failed to open log file: " << log_file << std::endl;
            }
            auto cs      = absl::ToCivilSecond(absl::Now(), absl::LocalTimeZone());
            auto message = absl::StrFormat("Log file created at: %04d/%02d/%02d %02d:%02d:%02d", cs.year(), cs.month(),
                                           cs.day(), cs.hour(), cs.minute(), cs.second());
            log_file_ << message << std::endl;
        }
    }

    ~DualLogSink() override {
        if (logtofile_ && log_file_.is_open()) {
            log_file_.close();
        }
    }

    void Send(const absl::LogEntry &entry) override {
        // 将日志条目格式化为带前缀的完整字符串
        std::string message =
            FormatPrefix(entry.log_severity(), entry.source_basename(), entry.source_line(), entry.timestamp());
        absl::StrAppendFormat(&message, " %s", entry.text_message_with_newline());

        // 输出到控制台
        if (logtostderr_) {
            std::string color_code = GetColorCode(entry.log_severity());
            std::string reset_code = "\033[0m"; // 重置颜色
            std::cout << color_code << message << reset_code;
        }

        // 输出到文件
        if (logtofile_ && log_file_.is_open()) {
            log_file_ << message;
            log_file_.flush();
        }
    }

private:
    // 根据日志级别返回ANSI颜色代码
    static std::string GetColorCode(absl::LogSeverity severity) {
        switch (severity) {
            case absl::LogSeverity::kError:
                return "\033[31m"; // 红色加粗
            case absl::LogSeverity::kFatal:
                return "\033[31m"; // 红色加粗
            case absl::LogSeverity::kWarning:
                return "\033[33m"; // 黄色加粗
            case absl::LogSeverity::kInfo:
                // return "\033[1;39m"; // 绿色加粗
            default:
                return "\033[0m"; // 默认无颜色
        }
    }

    // 格式化日志前缀
    static std::string FormatPrefix(absl::LogSeverity severity, absl::string_view file, int line,
                                    absl::Time timestamp) {
        // 日志等级
        string severity_str = "?";
        switch (severity) {
            case absl::LogSeverity::kInfo:
                severity_str = "INFO";
                break;
            case absl::LogSeverity::kWarning:
                severity_str = "WARN";
                break;
            case absl::LogSeverity::kError:
                severity_str = "ERROR";
                break;
            case absl::LogSeverity::kFatal:
                severity_str = "FATAL";
                break;
            default:
                break;
        }

        // 时间信息
        auto tz       = absl::LocalTimeZone();
        auto microsec = absl::ToInt64Microseconds(timestamp - absl::UnixEpoch()) % 1000000;
        auto time_str = absl::StrFormat("%s.%06d", absl::FormatTime("%H:%M:%S", timestamp, tz), microsec);

        return absl::StrFormat("%s %s %s:%d]", severity_str, time_str, file, line);
    }

private:
    bool logtostderr_;
    bool logtofile_;
    std::ofstream log_file_;
};

static DualLogSink *dual_sink = nullptr;
void Logging::initialization(bool logtostderr, bool logtofile, const string &log_file) {
    // 初始化 Abseil 日志系统
    absl::InitializeLog();

    // 禁用默认的stderr输出（将所有日志的最低级别设为 "无限高"，即不输出）
    absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfinity);

    // 创建自定义Sink（同时输出到控制台和文件）
    dual_sink = new DualLogSink(logtostderr, logtofile, log_file);
    absl::AddLogSink(dual_sink);
}

string Logging::doubleData(double data) {
    return absl::StrFormat("%0.6lf", data);
}

void Logging::shutdownLogging() {
    // 程序结束时清理
    absl::RemoveLogSink(dual_sink);
    delete dual_sink;
}