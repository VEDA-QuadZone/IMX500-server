// src/detect/speed_detector_daemon.cpp

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <limits>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <errno.h>
#include <algorithm>
#include <cctype>
#include <nlohmann/json.hpp>
#include <zmq.hpp>

#define TEST_MODE       1
#define MAX_TIME_DELTA 300LL
#define IN_LANE_X  250.0f
#define IN_LANE_Y  350.0f
#define OUT_LANE_X 550.0f
#define OUT_LANE_Y 350.0f

using json = nlohmann::json;

struct CarInfo {
    int         id;
    long long   latest_ts_ms;
    float       x, y, w, h;
};

std::vector<CarInfo> loadMetadata(const std::string& /*metaPath*/) {
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

bool parseBTMessage(const std::string& msg, float& speed, bool& inLane) {
    std::istringstream iss(msg);
    std::string lane;
    if (!(iss >> speed >> lane)) return false;
    inLane = (lane == "IN");
    return true;
}

int findMatchingCarId(long long eventTs, float sx, float sy, const std::vector<CarInfo>& cars) {
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

int main() {
    // ZeroMQ PUB 소켓 초기화
    zmq::context_t ctx(1);
    zmq::socket_t pub(ctx, zmq::socket_type::pub);
    pub.bind("ipc:///tmp/speed_detector.ipc");
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // subscriber 대기

    const char* dev = "/dev/rfcomm0";
    int fd = open(dev, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) {
        std::cerr << "[ERROR] Failed to open " << dev << ": "
                  << strerror(errno) << std::endl;
        return 1;
    }
    fcntl(fd, F_SETFL, 0);

    struct termios opts;
    if (tcgetattr(fd, &opts) != 0) {
        std::cerr << "[ERROR] tcgetattr: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
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
    if (tcsetattr(fd, TCSANOW, &opts) != 0) {
        std::cerr << "[ERROR] tcsetattr: " << strerror(errno) << std::endl;
        close(fd);
        return 1;
    }

    std::cout << "[INFO] Waiting for Bluetooth data on " << dev << "...\n";
    char buf[256];
    std::string linebuf;
    while (true) {
        int n = read(fd, buf, sizeof(buf));
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
                if (cars.empty()) continue;

                float sx = inLane ? IN_LANE_X : OUT_LANE_X;
                float sy = inLane ? IN_LANE_Y : OUT_LANE_Y;
                int id = findMatchingCarId(now_ms, sx, sy, cars);

                if (id >= 0) {
                    // ZeroMQ로 과속 감지 이벤트 JSON 전송
                    json out = {
                        {"event",     "ILLEGAL_SPEEDING"},
                        {"id",        id},
                        {"speed",     speed},
                        {"timestamp", now_ms},
                        {"sensor",    inLane ? "IN" : "OUT"}
                    };
                    pub.send(zmq::buffer(out.dump()), zmq::send_flags::none);
                    std::cout << "[PUB][SPEEDING] Car " << id
                              << " speed=" << speed
                              << " sensor=" << (inLane ? "IN" : "OUT") << "\n";
                }
            }
        } else if (n < 0 && errno != EINTR) {
            std::cerr << "[ERROR] Read failed: " << strerror(errno) << std::endl;
            break;
        }
        // 너무 빠른 loop 방지 (필요시)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    close(fd);
    return 0;
}
