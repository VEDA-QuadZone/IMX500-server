// src/detect/parking_detector.cpp

#include "detector.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <set>

namespace fs = std::filesystem;
using json = nlohmann::json;

static constexpr int    HISTORY_MIN_LENGTH = 20;
static constexpr double MOVEMENT_THRESHOLD = 5.0;

static int last_frame_id = -1;
static std::set<int> reported_ids;

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

static double avg_center_movement(const json& history) {
    double total = 0.0;
    for (size_t i = 1; i < history.size(); ++i) {
        auto& p = history[i-1], &c = history[i];
        double cx1 = p[0].get<double>() + p[2].get<double>()/2;
        double cy1 = p[1].get<double>() + p[3].get<double>()/2;
        double cx2 = c[0].get<double>() + c[2].get<double>()/2;
        double cy2 = c[1].get<double>() + c[3].get<double>()/2;
        total += std::hypot(cx2-cx1, cy2-cy1);
    }
    return total / (history.size()-1);
}

bool detect_illegal_parking(const json&) {
    auto path = find_latest_meta();
    if (path.empty()) return false;
    std::ifstream f(path);
    if (!f.is_open()) return false;

    json meta; 
    try { f >> meta; } catch (...) { return false; }

    for (auto& [key, arr] : meta.items()) {
        // **car/truck** 만 검사
        if (key != "car" && key != "truck") continue;

        for (auto& obj : arr) {
            const auto& hist = obj["history"];
            if (!hist.is_array() || hist.size() < HISTORY_MIN_LENGTH) continue;
            double move = avg_center_movement(hist);
            if (move < MOVEMENT_THRESHOLD) {
                return true;
            }
        }
    }
    return false;
}