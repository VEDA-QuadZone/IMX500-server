// src/upload/upload_illegal_parking.cpp

#include "detector.hpp"     // detect_illegal_parking_ids(), detect_license()
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
#include <thread>
#include <sys/mman.h>   // for shm_open, mmap, etc.
#include <sys/stat.h>   // for struct stat
#include <fcntl.h>      // for O_RDONLY
// 소켓으로 문자열 메시지 전송
static bool send_message(int sock, const std::string& msg) {
    return send(sock, msg.c_str(), msg.size(), 0) == (ssize_t)msg.size();
}

// SHM 세그먼트(이름)에서 바로 읽어 업로드
static bool upload_shm(int sock, const std::string& shm_name) {
    // shm_open + fstat 읽어서 버퍼에 담기
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return false; }
    std::vector<char> buf(st.st_size);
    if (::read(fd, buf.data(), buf.size()) != (ssize_t)buf.size()) {
        close(fd);
        return false;
    }
    close(fd);

    // UPLOAD 명령: 이름은 그대로 shm_name 으로
    std::ostringstream cmd;
    cmd << "UPLOAD " << shm_name << " " << buf.size() << "\n";
    if (!send_message(sock, cmd.str())) return false;

    // 바이너리 전송
    return send(sock, buf.data(), buf.size(), 0) == (ssize_t)buf.size();
}

int main(){
    std::cerr << "[DEBUG] Starting upload_illegal_parking (polling mode)\n";

    // 미리 소켓 연결
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port   = htons(TCP_SERVER_PORT);
    if (inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr) <= 0) {
        perror("inet_pton"); return 1;
    }
    if (connect(sock, (sockaddr*)&serv, sizeof(serv)) < 0) {
        perror("connect"); return 1;
    }
    std::cerr << "[DEBUG] Connected to TCP server\n";

    while (true) {
        // 1) 불법주정차 ID 목록 조회
        auto illegal_ids = detect_illegal_parking_ids();
        if (!illegal_ids.empty()) {
            std::cerr << "[DEBUG] Illegal parking IDs:";
            for (int id : illegal_ids) std::cerr << " " << id;
            std::cerr << "\n";

            // 2) 번호판 OCR → JSON 배열
            auto lic_list = detect_license(nlohmann::json::object());
            std::cerr << "[DEBUG] detect_license returned " << lic_list.size()
                      << " entries\n";

            // 3) 각 스냅샷별로 업로드 & ADD_HISTORY
            for (auto& lic : lic_list) {
                auto snap  = lic["shm_name"].get<std::string>();
                auto plate = lic["plate"].get<std::string>();
                std::string fullpath = "/dev/shm/" + snap;

                            // SHM에서 바로 업로드
            if (!upload_shm(sock, snap)) {
                std::cerr << "[ERROR] upload_shm failed: " << snap << "\n";
                continue;
            }

                // ADD_HISTORY 전송
                auto now = std::chrono::system_clock::now();
                std::time_t t_c = std::chrono::system_clock::to_time_t(now);
                std::tm tm{}; localtime_r(&t_c, &tm);
                char buf_time[20];
                std::strftime(buf_time, sizeof(buf_time), "%F %T", &tm);

                std::ostringstream ah;
                ah << "ADD_HISTORY "
                   << buf_time << " "
                   << "images/" << snap << " "
                   << plate << " "
                   << EventCode::ILLEGAL_PARKING
                   << "\n";
                send_message(sock, ah.str());
                std::cerr << "[DEBUG] Sent ADD_HISTORY for " << snap << "\n";
            }
        }
        else {
            std::cerr << "[DEBUG] No illegal parking detected\n";
        }

        // 1초 간격 폴링
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 언젠가 종료할 일 있으면 이쪽으로
    close(sock);
    return 0;
}
