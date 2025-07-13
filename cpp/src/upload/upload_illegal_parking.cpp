// src/upload/upload_illegal_parking.cpp

#include "detector.hpp"     // detect_illegal_parking, detect_license
#include "config.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <filesystem>

using json = nlohmann::json;

// find_latest_meta(), load_meta()는 detector.cpp에서 동일하게 쓰는 걸 복사해오거나
// detector.hpp에 선언해 두시면 됩니다.
static std::string find_latest_meta();  
static bool load_meta(const std::string& path, json& out);

// 소켓으로 문자열 메시지 전송
static bool send_message(int sock, const std::string& msg) {
    return send(sock, msg.c_str(), msg.size(), 0) == (ssize_t)msg.size();
}

// 파일 업로드 (UPLOAD + 바이너리)
static bool upload_file(int sock, const std::string& filepath) {
    std::ifstream ifs(filepath, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto sz = ifs.tellg();
    ifs.seekg(0);

    std::vector<char> buf(sz);
    if (!ifs.read(buf.data(), sz)) return false;

    std::string fname = std::filesystem::path(filepath).filename().string();
    std::ostringstream cmd;
    cmd << "UPLOAD " << fname << " " << sz << "\n";
    if (!send_message(sock, cmd.str())) return false;

    return send(sock, buf.data(), sz, 0) == sz;
}

int main(){
    
    // 1) 불법주정차 감지 (내부에서 메타파일을 찾아 처리)
     if (!detect_illegal_parking(nlohmann::json::object())) {
        std::cout << "[INFO] 불법주정차 아님 → 종료\n";
        return 0;
    }

        // 2) OCR 포함 번호판 인식
    json lic = detect_license(nlohmann::json::object());
    if (lic.is_null()) {
        std::cerr << "[ERROR] 번호판 OCR 실패\n";
        return 1;
    }

    std::string snap = lic["file"];
    std::string plate = lic["plate"];

    // 5) TCP 서버 연결
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port   = htons(TCP_SERVER_PORT);
    inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr);
    if (connect(sock, (sockaddr*)&serv, sizeof(serv)) < 0) {
        perror("connect"); close(sock); return 1;
    }

    // 6) 스냅샷 업로드
    std::string fullpath = "/dev/shm/" + snap;
    if (!upload_file(sock, fullpath)) {
        std::cerr << "[ERROR] 스냅샷 업로드 실패\n";
        close(sock);
        return 1;
    }

    // 7) ADD_HISTORY 전송
    auto now = std::chrono::system_clock::now();
    std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t_c, &tm);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%F %T", &tm);

    std::ostringstream ah;
    ah << "ADD_HISTORY "
       << buf << " "
       << "images/" << snap << " "
       << plate << " "
       << EventCode::ILLEGAL_PARKING << "\n";

    if (!send_message(sock, ah.str())) {
        std::cerr << "[ERROR] ADD_HISTORY 전송 실패\n";
        close(sock);
        return 1;
    }

    std::cout << "[INFO] 불법주정차 업로드 및 이력 추가 완료\n";
    close(sock);
    return 0;
}
