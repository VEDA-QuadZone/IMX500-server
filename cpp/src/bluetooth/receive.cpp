#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <string>      // std::string 사용
#include <algorithm>   // std::remove_if 사용
#include <cctype>      // std::isspace 사용
#include <sstream>     // std::stringstream 사용
#include <chrono>      // std::chrono::system_clock::now() 사용
#include <iomanip>     // std::put_time, std::setprecision 사용
#include <ctime>       // std::time_t, std::localtime 사용

// Bluetooth 시리얼 포트 장치 경로 정의
const char* RFCOMM_DEV = "/dev/rfcomm0";

// 과속 이벤트 정보를 저장할 구조체
struct SensorEvent {
    std::string timestamp;  // 이벤트 발생 시점 (라즈베리파이에서 기록)
    std::string lane_direction; // "IN" (A차선) 또는 "OUT" (B차선)
    double speed;           // 측정된 속도 (km/h)
    // int event_type;      // 이 포맷에서는 모든 메시지가 '과속'이므로 굳이 필요없음
};

/**
 * @brief 시리얼 포트 설정을 초기화합니다.
 * @param fd 열린 시리얼 포트의 파일 디스크립터
 * @return 설정 성공 시 true, 실패 시 false
 */
bool setupSerialPort(int fd) {
    struct termios options;
    if (tcgetattr(fd, &options) != 0) {
        std::cerr << "[ERROR] Failed to get port attributes: " << strerror(errno) << std::endl;
        return false;
    }

    cfsetispeed(&options, B9600);
    cfsetospeed(&options, B9600);
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;

    if (tcsetattr(fd, TCSANOW, &options) != 0) {
        std::cerr << "[ERROR] Failed to set port attributes: " << strerror(errno) << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief 수신된 문자열 메시지를 파싱하여 SensorEvent 구조체로 변환합니다.
 * 메시지 형식: "속도 방향" (예: "15.5 IN")
 * @param message 파싱할 문자열
 * @return 파싱된 SensorEvent 구조체. 파싱 실패 시 lane_direction이 비어있음.
 */
SensorEvent parseSensorEvent(const std::string& message) {
    SensorEvent event;
    std::stringstream ss(message);

    // 속도와 방향 문자열 파싱
    ss >> event.speed >> event.lane_direction;

    // 파싱이 성공했고 lane_direction이 "IN" 또는 "OUT"인지 확인
    if (!ss.fail() && (event.lane_direction == "IN" || event.lane_direction == "OUT")) {
        // 현재 시스템 시간을 timestamp로 기록 (라즈베리파이의 기준 시간 사용)
        auto now_chrono = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now_chrono);
        std::stringstream ss_ts;
        ss_ts << std::put_time(std::localtime(&now_c), "%Y-%m-%dT%H:%M:%S.");

        // 밀리초 추가
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_chrono.time_since_epoch()) % 1000;
        ss_ts << std::setfill('0') << std::setw(3) << ms.count() << "Z"; // Z는 UTC 의미, 필요시 로컬 타임존 반영

        event.timestamp = ss_ts.str();
    } else {
        std::cerr << "[WARN] Failed to parse message or invalid format: " << message << std::endl;
        event.lane_direction = ""; // 파싱 실패 표시
    }
    return event;
}

int main() {
    // Bluetooth 시리얼 포트 열기
    int fd = open(RFCOMM_DEV, O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd < 0) {
        std::cerr << "[ERROR] Failed to open port '" << RFCOMM_DEV << "': " << strerror(errno) << std::endl;
        // 포트 열기 실패 시 사용자에게 유용한 힌트 제공
        std::cerr << "[HINT] Is the HC-05 module powered on and paired? "
                  << "Is it bound to " << RFCOMM_DEV << "? Try 'sudo rfcomm' to check, "
                  << "or 'sudo rfcomm bind 0 <HC-05_MAC_ADDRESS>' to bind.\n";
        return 1;
    }

    // 시리얼 포트를 블로킹 모드로 설정 (데이터 수신 시까지 대기)
    fcntl(fd, F_SETFL, 0);

    // 시리얼 포트 설정
    if (!setupSerialPort(fd)) {
        close(fd);
        return 1;
    }

    std::cout << "[INFO] Waiting for Bluetooth data on " << RFCOMM_DEV << "...\n";

    char temp[256];           // 수신 데이터를 임시로 저장할 버퍼
    std::string messageBuffer;  // 불완전한 메시지를 저장할 버퍼

    // 무한 루프: 데이터 수신 및 처리
    while (true) {
        // 시리얼 포트에서 데이터 읽기 (블로킹 모드이므로 데이터가 올 때까지 대기)
        int n = read(fd, temp, sizeof(temp));
        if (n > 0) {
            messageBuffer.append(temp, n); // 읽은 데이터를 메시지 버퍼에 추가

            // 메시지 버퍼에서 '\n' (개행 문자) 기준으로 완전한 메시지 추출
            size_t pos;
            while ((pos = messageBuffer.find('\n')) != std::string::npos) {
                std::string line = messageBuffer.substr(0, pos); // '\n' 이전까지의 문자열 추출
                messageBuffer.erase(0, pos + 1);                // 추출된 메시지와 '\n'을 버퍼에서 제거

                // 추출된 메시지에서 불필요한 공백, 캐리지 리턴(\r) 등 제거
                line.erase(std::remove_if(line.begin(), line.end(), [](unsigned char x){
                    return std::isspace(x) || x == '\r';
                }), line.end());

                // 빈 문자열이 아니면 파싱 시도
                if (!line.empty()) {
                    std::cout << "[RECV] " << line << std::endl; // 원본 메시지 출력 (디버깅용)

                    SensorEvent event = parseSensorEvent(line);
                    if (!event.lane_direction.empty()) { // 유효한 이벤트인 경우
                        std::cout << "[PARSED] Timestamp: " << event.timestamp
                                  << ", Direction: " << event.lane_direction
                                  << ", Speed: " << std::fixed << std::setprecision(2) << event.speed << " km/h\n";

                        // TODO: 여기서 파싱된 SensorEvent (event)를 기반으로
                        // 1. 카메라 스냅샷 매칭 로직 호출
                        // 2. OCR 모듈 연동
                        // 3. 최종 데이터 저장 (ADD_HISTORY)
                        // 이 부분에 여러분의 시스템 통합 로직을 추가하시면 됩니다.
                    }
                }
            }
        } else if (n < 0) {
            // 읽기 실패 시 에러 처리
            if (errno == EINTR) {
                std::cout << "[INFO] Read interrupted, continuing...\n";
                continue;
            }
            std::cerr << "[ERROR] Read failed: " << strerror(errno) << std::endl;
            break;
        }
    }

    // 시리얼 포트 닫기
    close(fd);
    return 0;
}