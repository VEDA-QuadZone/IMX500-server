#include <mosquitto.h>
#include <iostream>
#include <thread>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include "config.hpp"
#include "mqtt_utils.hpp"
#include "detector.hpp"    // detect_persons()

using json = nlohmann::json;

// 비밀번호 콜백 함수 (개인키 비밀번호가 "1234"라고 가정)
int password_callback(char *buf, int size, int rwflag, void *userdata) {
    const std::string pass = "1234";
    if ((int)pass.size() < size) {
        strncpy(buf, pass.c_str(), size);
        return (int)pass.size();
    }
    return 0; // 실패 시 0 반환
}

// Mosquitto 로그 콜백 (디버깅용)
void log_callback(struct mosquitto *mosq, void *userdata, int level, const char *str) {
    std::cerr << "[mosquitto log] " << str << std::endl;
}

// MQTT 연결 실패 에러 출력 헬퍼
void check_mosq_connect_error(int ret) {
    if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "Mosquitto connection failed: " << mosquitto_strerror(ret) << std::endl;
        exit(1);
    }
}

int main() {
    mosquitto_lib_init();

    auto mosq = mosquitto_new("detector-client", true, nullptr);
    if (!mosq) {
        std::cerr << "Failed to create mosquitto client\n";
        return 1;
    }

    // 로그 콜백 등록 (선택)
    mosquitto_log_callback_set(mosq, log_callback);

    // TLS 설정
    int ret = mosquitto_tls_set(mosq,
        "/home/yuna/myCA/certs/ca.cert.pem",
        nullptr,
        "/home/yuna/myCA/mqtt_server/certs/mqtt_server.cert.pem",
        "/home/yuna/myCA/mqtt_server/private/mqtt_server.key.pem",
        password_callback);
    if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to set TLS options: " << mosquitto_strerror(ret) << std::endl;
        mosquitto_destroy(mosq);
        mosquitto_lib_cleanup();
        return 1;
    }

    // TLS 옵션 - 서버 인증서 검증 활성화
    ret = mosquitto_tls_opts_set(mosq, 1, nullptr, nullptr);
    if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to set TLS options 2: " << mosquitto_strerror(ret) << std::endl;
        mosquitto_destroy(mosq);
        mosquitto_lib_cleanup();
        return 1;
    }

    // TLS 포트로 연결 (예: 8883)
    ret = mosquitto_connect(mosq, BROKER_ADDRESS.c_str(), 8883, 60);
    if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to connect broker with TLS: " << mosquitto_strerror(ret) << std::endl;
        mosquitto_destroy(mosq);
        mosquitto_lib_cleanup();
        return 1;
    }

    ret = mosquitto_loop_start(mosq);
    if (ret != MOSQ_ERR_SUCCESS) {
        std::cerr << "Failed to start mosquitto loop: " << mosquitto_strerror(ret) << std::endl;
        mosquitto_destroy(mosq);
        mosquitto_lib_cleanup();
        return 1;
    }

    publish_event(mosq, CONNECTION_SUCCESS);

    // ZeroMQ SUB 소켓 연결 (speed_detector_daemon에 구독)
    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.connect("ipc:///tmp/speed_detector.ipc");
    sub.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    // 기존 로직(불법 주정차/보행자 감지)
    std::thread meta_loop([&]() {
        while (true) {
            auto illegal_ids = detect_illegal_parking_ids();
            if (!illegal_ids.empty()) publish_event(mosq, ILLEGAL_PARKING);

            auto person_ids = detect_persons();
            if (!person_ids.empty()) publish_event(mosq, PERSON_DETECTED);

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    });

    // 과속 감지 메시지 수신 loop (ZeroMQ)
    while (true) {
        zmq::message_t frame;
        sub.recv(frame, zmq::recv_flags::none);

        try {
            auto msg = json::parse(frame.to_string());
            if (msg["event"] == "ILLEGAL_SPEEDING") {
                publish_event(mosq, ILLEGAL_SPEEDING);
            }
        } catch (const std::exception &e) {
            std::cerr << "JSON parse error: " << e.what() << std::endl;
        }
    }

    // 종료 처리
    meta_loop.join();

    mosquitto_loop_stop(mosq, true);
    mosquitto_disconnect(mosq);
    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return 0;
}
//g++ -std=c++17 -o my_mqtt_app mqtt_alert_s.cpp -lmosquitto -lzmq -pthread
