// src/detect/person_detector.cpp
#include "detector.hpp"            // 선언: std::vector<int> detect_persons(const json&);
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// 같은 이름패턴으로 최신 메타파일 찾기
static std::string find_latest_meta() {
    std::string latest_path;
    fs::file_time_type latest_time;
    for (auto& e : fs::directory_iterator("/dev/shm")) {
        auto n = e.path().filename().string();
        if (n.rfind("shm_meta_", 0) != 0) continue;
        auto t = fs::last_write_time(e);
        if (latest_path.empty() || t > latest_time) {
            latest_path = e.path().string();
            latest_time = t;
        }
    }
    return latest_path;
}

std::vector<int> detect_persons(const json&) {
    auto path = find_latest_meta();
    if (path.empty()) return {};

    std::ifstream file(path);
    if (!file.is_open()) return {};

    json meta;
    try { file >> meta; }
    catch(...) { return {}; }

    std::vector<int> ids;
    // “person” 키만 살펴봅니다
    if (meta.contains("person") && meta["person"].is_array()) {
        for (auto& obj : meta["person"]) {
            int id = obj.value("id", -1);
            if (id >= 0) ids.push_back(id);
        }
    }
    return ids;
}
