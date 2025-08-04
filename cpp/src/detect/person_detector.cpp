#include "../../include/detector.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <set>

static constexpr const char* META_SHM_BASE = "shm_meta";

// /dev/shm/shm_index를 4바이트 리틀엔디언 정수로 읽음
static int read_shm_index() {
    std::ifstream in("/dev/shm/shm_index", std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[!] Cannot open /dev/shm/shm_index\n";
        return -1;
    }
    int slot;
    in.read(reinterpret_cast<char*>(&slot), sizeof(slot));
    return slot;
}

// shm_index 기반으로 메타 파일 경로 조합
static std::string find_latest_meta() {
    int slot = read_shm_index();
    if (slot < 0) return "";

    // 이제 META_SHM_BASE 에 언더바가 포함되어 있으니
    // 그냥 바로 slot 번호만 붙이면 됩니다.
    return "/dev/shm/" + std::string(META_SHM_BASE) 
                      + "_" + std::to_string(slot);
}

std::vector<int> detect_persons(const nlohmann::json& /*meta*/) {
    auto path = find_latest_meta();
    if (path.empty()) {
        std::cerr << "[!] Empty meta path\n";
        return {};
    }

    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[!] Unable to open meta file: " << path << "\n";
        return {};
    }

    nlohmann::json meta;
    try {
        file >> meta;
    } catch (const nlohmann::json::parse_error& e) {
        std::cerr << "[!] JSON parse error in " << path << ": " << e.what() << "\n";
        return {};
    }

    std::vector<int> ids;
    std::set<int> seen_ids;
    if (meta.contains("person") && meta["person"].is_array()) {
        for (auto& obj : meta["person"]) {
            int id = obj.value("id", -1);
            if (id >= 0 && seen_ids.find(id) == seen_ids.end()) {
                ids.push_back(id);
                seen_ids.insert(id);
            }
        }
    }
    return ids;
}
