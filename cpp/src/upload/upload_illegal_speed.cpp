#include <iostream>
#include <fstream>
#include <string>
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <filesystem>
#include "detector.hpp" // detect_license_by_id()
#include "config.hpp" // TCP_SERVER_IP, TCP_SERVER_PORT
#include <sstream>
#include <iomanip>
using json = nlohmann::json;

// TCP 전송 함수
bool send_file(int sock, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) return false;
    size_t filesize = file.tellg();
    file.seekg(0);

    std::vector<char> buf(4096);
    while (filesize > 0) {
        size_t chunk = std::min(buf.size(), filesize);
        file.read(buf.data(), chunk);
        ssize_t sent = send(sock, buf.data(), chunk, 0);
        if (sent <= 0) return false;
        filesize -= sent;
    }
    return true;
}

std::string get_now_date() {
    // 2025-07-07_15:30:00 형식 반환
    time_t t = time(nullptr);
    struct tm tm;
    localtime_r(&t, &tm);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d_%H:%M:%S", &tm);
    return buf;
}

int main() {
    // ZeroMQ SUB 초기화
    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.connect("ipc:///tmp/speed_detector.ipc");
    sub.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    // TCP 서버 정보
    const char* SERVER_IP = TCP_SERVER_IP.c_str();
    const int   SERVER_PORT = TCP_SERVER_PORT;

    while (true) {
        zmq::message_t frame;
        sub.recv(frame, zmq::recv_flags::none);

        auto msg = json::parse(frame.to_string());
        if (msg["event"] != "ILLEGAL_SPEEDING") continue;

        int car_id     = msg["id"];
        float speed    = msg["speed"];
        long long ts   = msg["timestamp"];
        std::string sensor = msg.value("sensor", "");

        // 1. 번호판/이미지 추출
        json result = detect_license_by_id(car_id);
        if (result.is_null() || result["shm_name"].empty()) {
            std::cerr << "[ERROR] No snapshot/plate for id=" << car_id << "\n";
            continue;
        }
        std::string shm_name = result["shm_name"];
        std::string plate    = result.value("plate", "");
        if (plate.empty()) plate = "0000";   // ← 여기 추가!
        // 2. TCP 서버 연결
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) { std::cerr << "[ERROR] socket()\n"; continue; }
        sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port   = htons(SERVER_PORT);
        inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr);

        if (connect(sock, (sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            std::cerr << "[ERROR] connect()\n";
            close(sock);
            continue;
        }

        // 3. 스냅샷 파일 경로 계산
        std::string snapshot_path = "/dev/shm/" + shm_name;
        std::ifstream infile(snapshot_path, std::ios::binary | std::ios::ate);
        if (!infile) { std::cerr << "[ERROR] file open\n"; close(sock); continue; }
        size_t filesize = infile.tellg();
        infile.seekg(0);

        // [중요] 파일명만 따로 추출
        std::string just_name = shm_name;

        // 4. UPLOAD 명령 (경로X, 파일명만)
        std::string upload_cmd = "UPLOAD " + just_name + ".jpg " + std::to_string(filesize) + "\n";
        send(sock, upload_cmd.c_str(), upload_cmd.size(), 0);

        // 5. 파일 전송
        if (!send_file(sock, snapshot_path)) {
            std::cerr << "[ERROR] file send\n"; close(sock); continue;
        }
        

       // 6. ADD_HISTORY 명령
std::string date = get_now_date();

std::ostringstream oss;
oss
  << "ADD_HISTORY "
  << date << " "
  << just_name << " "
  << plate << " "
  << "1 - - "
  << std::fixed << std::setprecision(2)  // 소수점 2자리
  << speed
  << "\n";

std::string add_history_cmd = oss.str();
send(sock, add_history_cmd.c_str(), add_history_cmd.size(), 0);
       

        // (선택) 응답 읽기/로깅 등

        close(sock);
        std::cout << "[UPLOAD][SPEED] id=" << car_id << " plate=" << plate << " sent.\n";
    }
    return 0;
}
