// src/detect/parking_detector.cpp (중복 ID 제거 반영)

#include "detector.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;
using json = nlohmann::json;

// 파라미터
static constexpr double STOP_SECONDS_THRESHOLD     = 10.0;  // 정지 시간 기준 (초)
static constexpr double PER_FRAME_MOVE_THRESHOLD   = 50.0;  // 프레임당 이동량 기준
static constexpr int    MIN_HISTORY_LENGTH         = 1;     // 최소 history 길이

// 차량별 상태 저장
struct ParkingInfo {
    double stopped_time = 0.0;
    std::vector<double> prev_box;
    double prev_time = 0.0;
};
static std::unordered_map<int, ParkingInfo> parking_map;

// 최신 shm_meta 파일 경로 반환
static std::string find_latest_meta() {
    std::string latest_path;
    fs::file_time_type latest_time;
    for (const auto& entry : fs::directory_iterator("/dev/shm")) {
        const auto& name = entry.path().filename().string();
        if (name.rfind("shm_meta_", 0) != 0) continue;

        auto t = fs::last_write_time(entry);
        if (latest_path.empty() || t > latest_time) {
            latest_path = entry.path().string();
            latest_time = t;
        }
    }
    return latest_path;
}

// 중심좌표 거리 계산
static double center_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double cx1 = a[0] + a[2] / 2.0, cy1 = a[1] + a[3] / 2.0;
    double cx2 = b[0] + b[2] / 2.0, cy2 = b[1] + b[3] / 2.0;
    return std::hypot(cx2 - cx1, cy2 - cy1);
}

/// @brief  불법 주정차로 판단된 객체의 ID 목록을 반환
/// @return 일정 시간 이상 정지한 car/truck 객체의 id 리스트 (중복 제거됨)
std::vector<int> detect_illegal_parking_ids() {
    std::unordered_set<int> unique_ids;

    std::string path = find_latest_meta();
    if (path.empty()) return {};

    std::ifstream fin(path);
    if (!fin.is_open()) return {};

    json meta;
    try {
        fin >> meta;
    } catch (...) {
        return {};
    }

    for (auto& [label, arr] : meta.items()) {
        if (label != "car" && label != "truck") continue;

        for (auto& obj : arr) {
            int id = obj["id"];
            const auto& hist = obj["history"];
            if (!hist.is_array() || hist.size() < MIN_HISTORY_LENGTH) continue;

            const auto& last = hist.back();
            if (last.size() < 5) continue;

            std::vector<double> cur_box = {last[0], last[1], last[2], last[3]};
            double cur_time = last[4];

            auto& info = parking_map[id];
            if (!info.prev_box.empty()) {
                double dist = center_distance(info.prev_box, cur_box);
                double dt = cur_time - info.prev_time;

                if (dist < PER_FRAME_MOVE_THRESHOLD) {
                    info.stopped_time += std::max(0.0, dt);
                } else {
                    info.stopped_time = 0.0;  // 움직임 있으면 초기화
                }
            }

            info.prev_box = cur_box;
            info.prev_time = cur_time;

            std::cerr << "[DEBUG] ID=" << id
                      << " label=" << label
                      << " stopped=" << info.stopped_time << "s\n";

            if (info.stopped_time >= STOP_SECONDS_THRESHOLD) {
                unique_ids.insert(id);
            }
        }
    }

    return std::vector<int>(unique_ids.begin(), unique_ids.end());
}
