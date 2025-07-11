// src/mqtt/mqtt_alert.cpp
#include <mosquitto.h>
#include <iostream>
#include <thread>
#include "config.hpp"
#include "mqtt_utils.hpp"
#include "detector.hpp"

int main(){
    mosquitto_lib_init();
    auto mosq = mosquitto_new("detector-client", true, nullptr);
    if (!mosq) {
        std::cerr<<"Failed to create client\n"; return 1;
    }
    if (mosquitto_connect(mosq,
          BROKER_ADDRESS.c_str(),
          BROKER_PORT, 60) != MOSQ_ERR_SUCCESS) {
        std::cerr<<"Failed to connect broker\n"; return 1;
    }

    // **네트워크 루프 백그라운드 스레드로 시작**
    mosquitto_loop_start(mosq);

    publish_event(mosq, CONNECTION_SUCCESS);

    while(true){
        if (detect_illegal_parking({})){
            publish_event(mosq, ILLEGAL_PARKING);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 종료 시 정리 (실제론 루프 벗어나지 않음)
    mosquitto_loop_stop(mosq, true);
    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return 0;
}
