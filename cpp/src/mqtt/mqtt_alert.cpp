#include <mosquitto.h>
#include <iostream>
#include <thread>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "mqtt_utils.hpp"
#include "detector.hpp"    // detect_persons()

using json = nlohmann::json;

int main() {
    // 1. MQTT 초기화
    mosquitto_lib_init();
    auto mosq = mosquitto_new("detector-client", true, nullptr);
    if (!mosq) {
        std::cerr << "Failed to create client\n"; return 1;
    }
    if (mosquitto_connect(mosq, BROKER_ADDRESS.c_str(), BROKER_PORT, 60)
        != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to connect broker\n"; return 1;
    }
    mosquitto_loop_start(mosq);
    publish_event(mosq, CONNECTION_SUCCESS);

    // 2. ZeroMQ SUB 소켓 연결 (speed_detector_daemon에 구독)
    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.connect("ipc:///tmp/speed_detector.ipc");
    sub.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    // 3. 기존 로직(불법 주정차/보행자 감지)도 같이 사용
    std::thread meta_loop([&]() {
        while (true) {
            auto illegal_ids = detect_illegal_parking_ids();
            if (!illegal_ids.empty()) publish_event(mosq, ILLEGAL_PARKING);

            auto person_ids = detect_persons();
            if (!person_ids.empty()) publish_event(mosq, PERSON_DETECTED);

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    // 4. 과속 감지 메시지 수신 loop (ZeroMQ)
    while (true) {
        zmq::message_t frame;
        sub.recv(frame, zmq::recv_flags::none);

        auto msg = json::parse(frame.to_string());
        if (msg["event"] == "ILLEGAL_SPEEDING") {
            // 과속차 감지시 MQTT로 alert
            publish_event(mosq, ILLEGAL_SPEEDING);
        }
    }

    // 종료시 정리
    meta_loop.join();
    mosquitto_loop_stop(mosq, true);
    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return 0;
}
