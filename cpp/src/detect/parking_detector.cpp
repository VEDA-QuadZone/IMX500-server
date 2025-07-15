// src/detect/parking_detector.cpp

#include "detector.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <set>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

// 최소 히스토리 길이와 움직임 임계값
static constexpr int    HISTORY_MIN_LENGTH = 20;
static constexpr double MOVEMENT_THRESHOLD = 5.0;

// /dev/shm 에서 가장 최근에 생성된 shm_meta_ 파일 경로를 찾음
static std::string find_latest_meta() {
    std::string latest_path;
    fs::file_time_type latest_time;
    for (auto& entry : fs::directory_iterator("/dev/shm")) {
        auto name = entry.path().filename().string();
        if (name.rfind("shm_meta_", 0) != 0) continue;
        auto t = fs::last_write_time(entry);
        if (latest_path.empty() || t > latest_time) {
            latest_path = entry.path().string();
            latest_time = t;
        }
    }
    return latest_path;
}

// history 배열을 따라 중심 좌표 변화량의 평균을 계산
static double avg_center_movement(const json& history) {
    double total = 0.0;
    for (size_t i = 1; i < history.size(); ++i) {
        auto& p = history[i-1];
        auto& c = history[i];
        double cx1 = p[0].get<double>() + p[2].get<double>()/2;
        double cy1 = p[1].get<double>() + p[3].get<double>()/2;
        double cx2 = c[0].get<double>() + c[2].get<double>()/2;
        double cy2 = c[1].get<double>() + c[3].get<double>()/2;
        total += std::hypot(cx2 - cx1, cy2 - cy1);
    }
    return total / (history.size() - 1);
}

/// @brief  불법 주정차로 판단된 객체의 ID 목록을 반환
/// @return 움직임이 임계값 이하인 car/truck 객체의 id 리스트 (없으면 빈 벡터)
std::vector<int> detect_illegal_parking_ids() {
    auto path = find_latest_meta();
    if (path.empty()) {
        // 메타 파일이 없으면 빈 리스트 반환
        return {};
    }

    std::ifstream fin(path);
    if (!fin.is_open()) {
        return {};
    }

    json meta;
    try {
        fin >> meta;
    } catch (...) {
        return {};
    }

    std::vector<int> ids;
    // "car" 또는 "truck" 만 검사
    for (auto& [label, arr] : meta.items()) {
        if (label != "car" && label != "truck") 
            continue;

        for (auto& obj : arr) {
            const auto& hist = obj["history"];
            if (!hist.is_array() || hist.size() < HISTORY_MIN_LENGTH)
                continue;

            double move = avg_center_movement(hist);
            if (move < MOVEMENT_THRESHOLD) {
                // json 안의 "id" 필드를 정수로 가져와 저장
                ids.push_back(obj["id"].get<int>());
            }
        }
    }

    return ids;
}
