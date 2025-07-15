// src/mqtt/mqtt_alert.cpp

#include <mosquitto.h>
#include <iostream>
#include <thread>
#include "config.hpp"
#include "mqtt_utils.hpp"
#include "detector.hpp"    // detect_persons()

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

    // 네트워크 루프 백그라운드 스레드로 시작
    mosquitto_loop_start(mosq);

    // 연결 성공 이벤트 발행
    publish_event(mosq, CONNECTION_SUCCESS);

    while(true){
        // 불법 주정차 검사
        {
    auto illegal_ids = detect_illegal_parking_ids();
    if (!illegal_ids.empty()) {
        publish_event(mosq, ILLEGAL_PARKING);
    }
}

        // 사람 감지 검사
        {
            auto person_ids = detect_persons();
            if (!person_ids.empty()) {
                // 감지된 사람 ID가 하나라도 있으면 이벤트 발행
                publish_event(mosq, PERSON_DETECTED);
            }
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // (실제론 여기엔 도달하지 않음)
    mosquitto_loop_stop(mosq, true);
    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return 0;
}
