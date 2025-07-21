// src/detect/parking_detector.cpp

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
#include <opencv2/opencv.hpp>
#include <ctime>

namespace fs = std::filesystem;
using json = nlohmann::json;

// 파라미터
static constexpr double STOP_SECONDS_THRESHOLD     = 10.0;  // 정지 시간 기준 (초)
static constexpr double PER_FRAME_MOVE_THRESHOLD   = 50.0;  // 프레임당 이동량 기준
static constexpr int    MIN_HISTORY_LENGTH         = 1;     // 최소 history 길이
static constexpr int    SHM_FRAME_COUNT            = 8;     // shm_frame_0~7
static constexpr int    FRAME_WIDTH                = 1280;  // 실제 해상도에 맞게 수정
static constexpr int    FRAME_HEIGHT               = 720;   // 실제 해상도에 맞게 수정

// 차량별 상태 저장
struct ParkingInfo {
    double stopped_time = 0.0;
    std::vector<double> prev_box;
    double prev_time = 0.0;
    bool is_stopped = false;
    bool start_shot_saved = false;
    bool end_shot_saved = false;
    std::string stop_start_time_str;
    int stop_start_frame = -1;
};

// (id별) 감시 및 "이미 처리된 id" 세트
static std::unordered_map<int, ParkingInfo> parking_map;
static std::unordered_set<int> finished_ids; // 이미 처리 끝난 id

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

// timestamp(double, sec) → "YYYYMMDD_HHMMSS" 문자열
static std::string format_time(double timestamp) {
    time_t t = static_cast<time_t>(timestamp);
    struct tm tm;
    localtime_r(&t, &tm);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);
    return std::string(buf);
}

// shm_frame_N(BGRA) → jpeg 저장
static bool save_jpeg_shm_frame(int frame_id, const std::string& filename) {
    if (frame_id < 0) return false;
    std::string shm_path = "/dev/shm/shm_frame_" + std::to_string(frame_id % SHM_FRAME_COUNT);
    FILE* fp = fopen(shm_path.c_str(), "rb");
    if (!fp) return false;

    std::vector<unsigned char> buf(FRAME_WIDTH * FRAME_HEIGHT * 4); // BGRA
    size_t nread = fread(buf.data(), 1, buf.size(), fp);
    fclose(fp);

    if (nread != buf.size()) return false;

    // BGRA → BGR (OpenCV)
    cv::Mat img(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC4, buf.data());
    cv::Mat img_bgr;
    cv::cvtColor(img, img_bgr, cv::COLOR_BGRA2BGR);
    return cv::imwrite(filename, img_bgr);
}

std::vector<int> detect_illegal_parking_ids() {
    std::unordered_set<int> unique_ids;

    std::string path = find_latest_meta();
    if (path.empty()) return {};

    std::ifstream fin(path);
    if (!fin.is_open()) return {};

    json meta;
    try { fin >> meta; } catch (...) { return {}; }

    int meta_frame_id = -1;
    double meta_frame_time = 0;
    if (meta.contains("frame_id")) meta_frame_id = int(meta["frame_id"]);
    if (meta.contains("timestamp")) {
        try {
            std::string ts = meta["timestamp"];
            struct tm tm = {};
            strptime(ts.c_str(), "%Y-%m-%dT%H:%M:%S", &tm);
            meta_frame_time = mktime(&tm);
        } catch (...) {}
    }

    for (auto& [label, arr] : meta.items()) {
        if (label != "car" && label != "truck") continue;

        for (auto& obj : arr) {
            int id = obj["id"];
            if (finished_ids.count(id)) continue; // 이미 끝난 id는 무조건 건너뜀

            int active = obj.value("active", 0);
            if (active != 1) continue; // 현재 프레임에 있는 객체만

            const auto& hist = obj["history"];
            if (!hist.is_array() || hist.size() < MIN_HISTORY_LENGTH) continue;

            const auto& last = hist.back();
            if (last.size() < 5) continue;

            std::vector<double> cur_box = {last[0], last[1], last[2], last[3]};
            double cur_time = last[4];
            int cur_frame = meta_frame_id;

            auto& info = parking_map[id];
            bool was_stopped = info.is_stopped;

            // 이동량 계산
            if (!info.prev_box.empty()) {
                double dist = center_distance(info.prev_box, cur_box);
                double dt = cur_time - info.prev_time;

                if (dist < PER_FRAME_MOVE_THRESHOLD) {
                    info.stopped_time += std::max(0.0, dt);
                    info.is_stopped = true;
                } else {
                    info.stopped_time = 0.0;
                    info.is_stopped = false;
                    info.start_shot_saved = false;
                    info.end_shot_saved = false;
                }
            }
            info.prev_box = cur_box;
            info.prev_time = cur_time;

            // 정지 시작 샷
            if (info.is_stopped && !was_stopped && !info.start_shot_saved) {
                info.start_shot_saved = true;
                info.stop_start_time_str = format_time(cur_time);
                info.stop_start_frame = cur_frame;
                std::string fname = "/dev/shm/shm_startshot_" + std::to_string(id) + "_" + info.stop_start_time_str + ".jpg";
                if (save_jpeg_shm_frame(cur_frame, fname))
                    std::cerr << "[INFO] Startshot saved: " << fname << std::endl;
                else
                    std::cerr << "[WARN] Startshot failed: " << fname << std::endl;
            }
            // 10초 경과 엔드샷 및 "완료 id" 등록
            if (info.is_stopped && info.stopped_time >= STOP_SECONDS_THRESHOLD && !info.end_shot_saved) {
                info.end_shot_saved = true;
                std::string time_str = format_time(cur_time);
                std::string fname = "/dev/shm/shm_endshot_" + std::to_string(id) + "_" + time_str + ".jpg";
                if (save_jpeg_shm_frame(cur_frame, fname))
                    std::cerr << "[INFO] Endshot saved: " << fname << std::endl;
                else
                    std::cerr << "[WARN] Endshot failed: " << fname << std::endl;

                finished_ids.insert(id); // 앞으로 이 id는 영구 제외!
            }

            if (info.stopped_time >= STOP_SECONDS_THRESHOLD)
                unique_ids.insert(id);
        }
    }
    return std::vector<int>(unique_ids.begin(), unique_ids.end());
}
