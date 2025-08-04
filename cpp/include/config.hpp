// include/config.hpp
#pragma once

static const std::string BROKER_ADDRESS = "localhost";
static const int         BROKER_PORT    = 1883;
// 하나의 토픽으로 모든 알림을 보냅니다
static const std::string TOPIC_NAME     = "alert";

// 이벤트 코드
enum EventCode {
    CONNECTION_SUCCESS   = -1,
    ILLEGAL_PARKING      = 0,
    ILLEGAL_SPEEDING     = 1,
    PERSON_DETECTED      = 2
};
static const std::string TCP_SERVER_IP = "127.0.0.1";
static const int         TCP_SERVER_PORT = 8080;
