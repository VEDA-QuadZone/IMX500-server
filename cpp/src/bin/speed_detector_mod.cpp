// src/detect/speed_detector.cpp

#include "detector.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <limits>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>

// ─── 매크로 및 alias ───────────────────────────────────
#define TEST_MODE       1
#define MAX_TIME_DELTA 300LL
#define IN_LANE_X  250.0f
#define IN_LANE_Y  350.0f
#define OUT_LANE_X 550.0f
#define OUT_LANE_Y 350.0f

using json = nlohmann::json;
// ────────────────────────────────────────────────────────

//
// CarInfo 구조체 및 메타데이터 로드 함수
//
struct CarInfo {
    int         id;
    long long   latest_ts_ms;
    float       x, y, w, h;
};

static std::vector<CarInfo> loadMetadata(const std::string& /*metaPath*/) {
#if TEST_MODE
    static const std::string dummy = R"({
      "frame_id": 1,
      "timestamp": "2025-07-16T00:00:00",
      "car": [
        { "id": 0,  "conf": 1.0, "box": [200, 300, 100, 100], "history":[[200,300,100,100,0.0]] },
        { "id": 11, "conf": 1.0, "box": [500, 300, 100, 100], "history":[[500,300,100,100,0.0]] }
      ]
    })";
    json j = json::parse(dummy);

    long long now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(
                           std::chrono::system_clock::now()
                       ).time_since_epoch().count();

    std::vector<CarInfo> cars;
    for (auto& item : j["car"]) {
        auto& b = item["box"];
        CarInfo c;
        c.id           = item["id"].get<int>();
        c.latest_ts_ms = now_ms;
        c.x = b[0].get<float>();
        c.y = b[1].get<float>();
        c.w = b[2].get<float>();
        c.h = b[3].get<float>();
        cars.push_back(c);
    }
    return cars;
#else
    static constexpr int BUFFER_SLOTS = 8;
    static const std::string META_SHM_BASE = "/dev/shm/shm_meta_";

    std::vector<CarInfo> cars;
    for (int slot = 0; slot < BUFFER_SLOTS; ++slot) {
        std::string path = META_SHM_BASE + std::to_string(slot);
        std::ifstream ifs(path);
        if (!ifs.is_open()) continue;

        json j;
        try { ifs >> j; }
        catch (...) { continue; }

        if (!j.contains("car") || !j["car"].is_array()) continue;
        for (auto& item : j["car"]) {
            auto& b = item["box"];
            auto& h = item["history"];
            if (!b.is_array() || b.size() < 4 || !h.is_array() || h.empty())
                continue;
            CarInfo c;
            c.id = item["id"].get<int>();
            double ts = h.back()[4].get<double>();
            c.latest_ts_ms = static_cast<long long>(ts * 1000.0);
            c.x = b[0].get<float>();
            c.y = b[1].get<float>();
            c.w = b[2].get<float>();
            c.h = b[3].get<float>();
            cars.push_back(c);
        }
    }
    return cars;
#endif
}

//
// BT 메시지 파싱 및 차량 매칭
//
static bool parseBTMessage(const std::string& msg, float& speed, bool& inLane) {
    std::istringstream iss(msg);
    std::string lane;
    if (!(iss >> speed >> lane)) return false;
    inLane = (lane == "IN");
    return true;
}

static int findMatchingCarId(long long eventTs, float sx, float sy, const std::vector<CarInfo>& cars) {
    long long bestDelta = std::numeric_limits<long long>::max();
    int bestId = -1;
    for (auto& c : cars) {
        if (sx < c.x || sx > c.x + c.w || sy < c.y || sy > c.y + c.h)
            continue;
        long long delta = std::llabs(eventTs - c.latest_ts_ms);
        if (delta < bestDelta) {
            bestDelta = delta;
            bestId    = c.id;
        }
    }
    return (bestDelta <= MAX_TIME_DELTA) ? bestId : -1;
}

//
// 모듈 전역 변수
//
static int serial_fd = -1;
static std::string linebuf;

//
// 초기화 함수
//
bool init_speed_detector(const std::string& dev) {
    serial_fd = open(dev.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (serial_fd < 0) {
        std::cerr << "[ERROR] open " << dev << ": " << strerror(errno) << "\n";
        return false;
    }
    fcntl(serial_fd, F_SETFL, 0);

    termios opts;
    if (tcgetattr(serial_fd, &opts) != 0) {
        std::cerr << "[ERROR] tcgetattr: " << strerror(errno) << "\n";
        close(serial_fd);
        serial_fd = -1;
        return false;
    }
    cfsetispeed(&opts, B9600);
    cfsetospeed(&opts, B9600);
    opts.c_cflag |= (CLOCAL | CREAD);
    opts.c_cflag &= ~PARENB;
    opts.c_cflag &= ~CSTOPB;
    opts.c_cflag &= ~CSIZE;
    opts.c_cflag |= CS8;
    opts.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    opts.c_iflag &= ~(IXON | IXOFF | IXANY);
    opts.c_oflag &= ~OPOST;
    if (tcsetattr(serial_fd, TCSANOW, &opts) != 0) {
        std::cerr << "[ERROR] tcsetattr: " << strerror(errno) << "\n";
        close(serial_fd);
        serial_fd = -1;
        return false;
    }
    return true;
}

//
// 과속 차량 ID 반환 함수 (lazy init 포함)
//
std::vector<int> detect_speeding_cars() {
    std::vector<int> speeding_ids;

    // ─── lazy init: 첫 호출 시 블루투스 포트 열고 설정 ────────────────────
    if (serial_fd < 0) {
        serial_fd = open("/dev/rfcomm0", O_RDWR | O_NOCTTY | O_NDELAY);
        if (serial_fd < 0) {
            std::cerr << "[ERROR] open /dev/rfcomm0: " << strerror(errno) << "\n";
            return {};
        }
        fcntl(serial_fd, F_SETFL, 0);

        termios opts;
        if (tcgetattr(serial_fd, &opts) != 0) {
            std::cerr << "[ERROR] tcgetattr: " << strerror(errno) << "\n";
            close(serial_fd);
            serial_fd = -1;
            return {};
        }
        cfsetispeed(&opts, B9600);
        cfsetospeed(&opts, B9600);
        opts.c_cflag |= (CLOCAL | CREAD);
        opts.c_cflag &= ~PARENB;
        opts.c_cflag &= ~CSTOPB;
        opts.c_cflag &= ~CSIZE;
        opts.c_cflag |= CS8;
        opts.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
        opts.c_iflag &= ~(IXON | IXOFF | IXANY);
        opts.c_oflag &= ~OPOST;
        if (tcsetattr(serial_fd, TCSANOW, &opts) != 0) {
            std::cerr << "[ERROR] tcsetattr: " << strerror(errno) << "\n";
            close(serial_fd);
            serial_fd = -1;
            return {};
        }
    }
    // ────────────────────────────────────────────────────────────────

    char buf[256];
    int n = read(serial_fd, buf, sizeof(buf));
    if (n > 0) {
        linebuf.append(buf, n);
        size_t pos;
        while ((pos = linebuf.find('\n')) != std::string::npos) {
            std::string msg = linebuf.substr(0, pos);
            linebuf.erase(0, pos + 1);
            msg.erase(std::remove_if(msg.begin(), msg.end(),
                                     [](unsigned char c){
                                         return std::isspace(c) || c == '\r';
                                     }),
                      msg.end());
            if (msg.empty()) continue;

            float speed; bool inLane;
            if (!parseBTMessage(msg, speed, inLane)) continue;
            long long now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now()
                               ).time_since_epoch().count();

            auto cars = loadMetadata("/dev/shm/meta.json");
            float sx = inLane ? IN_LANE_X : OUT_LANE_X;
            float sy = inLane ? IN_LANE_Y : OUT_LANE_Y;
            int id = findMatchingCarId(now_ms, sx, sy, cars);
            if (id >= 0) {
                speeding_ids.push_back(id);
            }
        }
    }
    return speeding_ids;
}

//
// 정리 함수
//
void cleanup_speed_detector() {
    if (serial_fd >= 0) {
        close(serial_fd);
        serial_fd = -1;
    }
}
