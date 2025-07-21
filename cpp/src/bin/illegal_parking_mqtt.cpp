#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <set>
#include <cmath>
#include <ctime>
#include <thread>
#include <mosquitto.h>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

const std::string BROKER_ADDRESS = "localhost";
const int BROKER_PORT = 1883;
const std::string TOPIC = "alert/parking";
const int HISTORY_MIN_LENGTH = 20;
const double MOVEMENT_THRESHOLD = 5.0;

std::set<int> reported_ids;

std::string get_current_timestamp() {
    std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%FT%T", std::localtime(&now));
    return std::string(buf);
}

std::string get_latest_meta_file() {
    std::string latest_file;
    std::filesystem::file_time_type latest_time;

    for (const auto& entry : fs::directory_iterator("/dev/shm")) {
        const auto& path = entry.path();
        if (path.filename().string().find("shm_meta_") == 0) {
            auto t = fs::last_write_time(path);
            if (latest_file.empty() || t > latest_time) {
                latest_file = path;
                latest_time = t;
            }
        }
    }
    return latest_file;
}

double avg_center_movement(const json& history) {
    if (!history.is_array() || history.size() < 2) return 0.0;
    double total = 0.0;
    for (size_t i = 1; i < history.size(); ++i) {
        auto prev = history[i - 1];
        auto curr = history[i];
        double cx1 = prev[0].get<double>() + prev[2].get<double>() / 2;
        double cy1 = prev[1].get<double>() + prev[3].get<double>() / 2;
        double cx2 = curr[0].get<double>() + curr[2].get<double>() / 2;
        double cy2 = curr[1].get<double>() + curr[3].get<double>() / 2;
        total += std::hypot(cx2 - cx1, cy2 - cy1);
    }
    return total / (history.size() - 1);
}

void publish_event(struct mosquitto* mosq, int event_code) {
    json payload = {
        {"timestamp", get_current_timestamp()},
        {"event", event_code}
    };
    std::string msg = payload.dump();
    mosquitto_publish(mosq, nullptr, TOPIC.c_str(), msg.size(), msg.c_str(), 1, false);
    std::cout << "📡 MQTT published:\n" << msg << std::endl;
}

int main() {
    mosquitto_lib_init();
    struct mosquitto* mosq = mosquitto_new("detector-client", true, nullptr);
    if (!mosq) {
        std::cerr << "Failed to create Mosquitto client\n";
        return 1;
    }

    if (mosquitto_connect(mosq, BROKER_ADDRESS.c_str(), BROKER_PORT, 60) != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to connect to MQTT broker\n";
        return 1;
    }

    // ✅ 연결되면 최초에 한 번 event:-1 알림
    publish_event(mosq, -1);

    while (true) {
        std::string meta_path = get_latest_meta_file();
        if (meta_path.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            continue;
        }

        std::ifstream file(meta_path);
        if (!file.is_open()) continue;

        json meta;
        try {
            file >> meta;
        } catch (...) {
            continue;
        }

        for (const auto& [key, val] : meta.items()) {
            if (key == "frame_id" || key == "timestamp") continue;
            for (const auto& obj : val) {
                int id = obj["id"];
                const auto& history = obj["history"];
                if (history.size() < HISTORY_MIN_LENGTH || reported_ids.count(id)) continue;

                double move = avg_center_movement(history);
                if (move < MOVEMENT_THRESHOLD) {
                    publish_event(mosq, 0); // 0: 불법주정차
                    reported_ids.insert(id);
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return 0;
}
//g++ illegal_parking_mqtt.cpp -o mqtt_alert -lmosquitto