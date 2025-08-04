// include/mqtt_utils.hpp
#pragma once
#include <mosquitto.h>
#include <nlohmann/json.hpp>
#include "config.hpp"

using json = nlohmann::json;

inline void publish_event(struct mosquitto* mosq, int event_code) {
    json payload = {
        {"timestamp", [](){
            std::time_t now = std::time(nullptr);
            char buf[32];
            std::strftime(buf, sizeof(buf), "%FT%T", std::localtime(&now));
            return std::string(buf);
        }()},
        {"event", event_code}
    };
    auto msg = payload.dump();
    mosquitto_publish(mosq, nullptr,
                      TOPIC_NAME.c_str(),
                      msg.size(), msg.c_str(),
                      1, false);
    std::cout << "📡 MQTT published:\n" << msg << std::endl;
}
