#include <iostream>
#include <mosquitto.h>

const std::string TOPIC = "alert";
const int BROKER_PORT = 1883;
const std::string BROKER_ADDRESS = "localhost";

// 콜백: 연결되었을 때
void on_connect(struct mosquitto *mosq, void *obj, int rc) {
    if (rc == 0) {
        std::cout << "[MQTT] 연결 성공! 토픽 구독 중..." << std::endl;
        mosquitto_subscribe(mosq, nullptr, TOPIC.c_str(), 0);
    } else {
        std::cerr << "[MQTT] 연결 실패, 코드: " << rc << std::endl;
    }
}

// 콜백: 메시지 수신 시
void on_message(struct mosquitto *mosq, void *obj, const struct mosquitto_message *msg) {
    std::cout << "\n수신된 MQTT 메시지 [" << msg->topic << "]\n"
              << std::string((char*)msg->payload, msg->payloadlen) << "\n" << std::endl;
}

int main() {
    mosquitto_lib_init();

    struct mosquitto *mosq = mosquitto_new("local-subscriber", true, nullptr);
    if (!mosq) {
        std::cerr << "Mosquitto 객체 생성 실패" << std::endl;
        return 1;
    }

    mosquitto_connect_callback_set(mosq, on_connect);
    mosquitto_message_callback_set(mosq, on_message);

    if (mosquitto_connect(mosq, BROKER_ADDRESS.c_str(), BROKER_PORT, 60) != MOSQ_ERR_SUCCESS) {
        std::cerr << "MQTT 브로커 연결 실패" << std::endl;
        return 1;
    }

    std::cout << "[MQTT] 메시지 수신 대기 중... (Ctrl+C로 종료)\n" << std::endl;
    mosquitto_loop_forever(mosq, -1, 1);

    mosquitto_destroy(mosq);
    mosquitto_lib_cleanup();
    return 0;
}
