// src/upload/upload_person_detection.cpp

#include <iostream>
#include <filesystem>
#include <fstream>
#include <set>
#include <vector>
#include <thread>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "detector.hpp"   // std::vector<int> detect_persons(const json&);
#include "config.hpp"     // TCP_SERVER_IP, TCP_SERVER_PORT

namespace fs = std::filesystem;

// 1) 현재 시각을 "YYYY-MM-DD HH:MM:SS" 로 반환
static std::string now_str() {
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t_c, &tm);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%F %T", &tm);
    return std::string(buf);
}

// 2) "shm_snapshot_<slot>_<id>_<ts>.jpg" 패턴 파싱
static bool parse_snapshot_name(const std::string& name,
                                int& out_slot,
                                int& out_id,
                                std::string& out_ts) {
    const std::string prefix = "shm_snapshot_";
    if (name.rfind(prefix, 0) != 0) return false;

    // core = "<slot>_<id>_<ts>"
    auto core = name.substr(prefix.size());
    auto p1 = core.find('_');
    auto p2 = core.find('_', p1+1);
    if (p1 == std::string::npos || p2 == std::string::npos) return false;

    out_slot = std::stoi(core.substr(0, p1));
    out_id   = std::stoi(core.substr(p1+1, p2-(p1+1)));
    out_ts   = core.substr(p2+1);
    return true;
}

// 3) target_id용 SHM 스냅샷을 읽어 Mat(BGR) + 메타정보 반환 (fallback 포함)
static bool load_snapshot(int target_id,
                          cv::Mat& out_bgr,
                          std::string& out_ts) {
    const fs::path shm_dir = "/dev/shm";
    for (auto& e : fs::directory_iterator(shm_dir)) {
        auto fn = e.path().filename().string();
        int slot, id;
        std::string ts;
        if (!parse_snapshot_name(fn, slot, id, ts)) continue;
        if (id != target_id) continue;

        // read all bytes
        std::ifstream in(e.path(), std::ios::binary | std::ios::ate);
        if (!in.is_open()) continue;
        auto sz = in.tellg();
        in.seekg(0);
        std::vector<unsigned char> buf(sz);
        in.read(reinterpret_cast<char*>(buf.data()), sz);

        // JPEG → BGR
        out_bgr = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (out_bgr.empty()) {
            std::cerr << "[!] JPEG decode failed: " << fn << "\n";
            continue;
        }
        out_ts = ts;
        return true;
    }
    return false;
}

// 4) Mat을 파일로 저장 (id, ts, w×h 포함)
static std::string save_image(const cv::Mat& img,
                              int id,
                              const std::string& ts
) {
    fs::create_directories("images");
    std::string safe_ts = ts;
    std::replace(safe_ts.begin(), safe_ts.end(), ' ', '_');
    std::replace(safe_ts.begin(), safe_ts.end(), ':', '-');

    std::string fname = "person_"
                      + std::to_string(id) + "_"
                      + safe_ts + ".jpg";
    std::string path = "images/" + fname;

    if (!cv::imwrite(path, img)) {
        std::cerr << "[ERROR] 이미지 저장 실패: " << path << "\n";
        return "";
    }
    return path;
}


static bool send_upload_image(const std::string& img_path) {
    // 1) open file & get its size
    std::error_code ec;
    size_t size = fs::file_size(img_path, ec);
    if (ec) {
        std::cerr << "[ERROR] file_size error: " << ec.message() << "\n";
        return false;
    }

    // 2) connect
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return false; }
    sockaddr_in serv{AF_INET, htons(TCP_SERVER_PORT), {0}};
    if (inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr) <= 0 ||
        connect(sock, (sockaddr*)&serv, sizeof(serv)) < 0) {
        perror("connect"); close(sock); return false;
    }

    // 3) send header
    std::string filename = fs::path(img_path).filename().string();
    std::string header   = "UPLOAD " + filename + " " + std::to_string(size) + "\n";
    if (send(sock, header.data(), header.size(), 0) < 0) {
        perror("send header"); close(sock); return false;
    }

    // 4) send body
    std::ifstream in(img_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[ERROR] cannot open " << img_path << "\n";
        close(sock);
        return false;
    }
    const size_t BUF_SZ = 4096;
    std::vector<char> buf(BUF_SZ);
    size_t total = 0;
    while (in.good() && total < size) {
        in.read(buf.data(), std::min(BUF_SZ, size - total));
        std::streamsize r = in.gcount();
        if (r <= 0) break;
        ssize_t s = send(sock, buf.data(), r, 0);
        if (s < 0) { perror("send data"); in.close(); close(sock); return false; }
        total += s;
    }
    in.close();
    std::cout << "[DEBUG] Declared size=" << size
              << ", actually sent=" << total << std::endl;

    // 5) ⚠️ tell server “no more body” so it can finish reading
    if (shutdown(sock, SHUT_WR) < 0) {
        perror("shutdown UPLOAD SHUT_WR");
    }

    // 6) read entire JSON response (until we see the closing '}')
    std::string resp;
    char rbuf[512];
    while (true) {
        ssize_t n = recv(sock, rbuf, sizeof(rbuf), 0);
        if (n <= 0) break;
        resp.append(rbuf, n);
        if (resp.find('}') != std::string::npos) break;
    }

    if (resp.empty()) {
        std::cerr << "[ERROR] no response from server\n";
        close(sock);
        return false;
    }
    std::cout << "[TCP] Server response to UPLOAD: " << resp << std::endl;

    // 7) close & return
    close(sock);
    return true;
}




static bool send_add_history(const std::string& date,
                             const std::string& img_path,
                             const std::string& plate,
                             int event_type) {
    // 1) 소켓 생성 & 서버 연결
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return false; }
    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port   = htons(TCP_SERVER_PORT);
    if (inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr) <= 0) {
        perror("inet_pton"); close(sock); return false;
    }
    if (connect(sock, (sockaddr*)&serv, sizeof(serv)) < 0) {
        perror("connect"); close(sock); return false;
    }

    // 2) ADD_HISTORY 명령 전송
    std::string cmd = "ADD_HISTORY "
                    + date + " "
                    + img_path + " "
                    + plate + " "
                    + std::to_string(event_type)
                    + "\n";
    if (send(sock, cmd.c_str(), cmd.size(), 0) < 0) {
        perror("send"); close(sock); return false;
    }

    // 3) 더 보낼 데이터 없음을 서버에 알림 (EOF)
    if (shutdown(sock, SHUT_WR) < 0) {
        perror("shutdown ADD_HISTORY SHUT_WR");
    }

    // 4) 서버 응답 수신 (성공 혹은 reset 도 정상으로 처리)
    char resp_buf[1024];
    int n = recv(sock, resp_buf, sizeof(resp_buf) - 1, 0);
    if (n > 0) {
        resp_buf[n] = '\0';
        std::cout << "[TCP] Server response to ADD_HISTORY: " 
                  << resp_buf << std::endl;
    } else if (n == 0) {
        std::cout << "[TCP] Server closed connection after ADD_HISTORY" 
                  << std::endl;
    } else {
        if (errno == ECONNRESET) {
            std::cout << "[TCP] Connection reset by peer after ADD_HISTORY; assuming success"
                      << std::endl;
        } else {
            perror("recv");
        }
    }

    // 5) 소켓 닫기
    close(sock);
    std::cout << "[TCP] Sent ADD_HISTORY: " << cmd;
    return true;
}



int main() {
    std::set<int> prev_ids, reported;

    while (true) {
        auto ids = detect_persons();
        std::set<int> current(ids.begin(), ids.end());

        // 사라진 ID만 처리
        for (int id : prev_ids) {
            if (reported.count(id)) continue;

            cv::Mat img;
            std::string ts;
            if (!load_snapshot(id, img, ts))
                continue;

            // 1) 파일로 저장
            std::string path = save_image(img, id, ts);
            if (path.empty())
                continue;

            // 2) 이미지 UPLOAD
            if (!send_upload_image(path))
                continue;

            // 3) ADD_HISTORY
            if (!ts.empty()) {
                // ts = "YYYYMMDD_HHMMSS" → "YYYY-MM-DD HH:MM:SS"
                std::string date = ts.substr(0,4) + "-"   // YYYY-
                                 + ts.substr(4,2) + "-"   // MM-
                                 + ts.substr(6,2) + "_"   // DD_
                                 + ts.substr(9,2) + ":"   // HH:
                                 + ts.substr(11,2) + ":"  // MM:
                                 + ts.substr(13,2);       // SS
                if (send_add_history(date, path, "-", 2)) {
                    reported.insert(id);
                }
            }
        }

        // 반복 준비: 이번 current를 다음 prev_ids로
        prev_ids = std::move(current);

        // 디버깅: 루프 한 사이클 지났음을 찍어보면 좋습니다.
        std::cout << "[DEBUG] Loop tick, prev_ids size=" << prev_ids.size() << "\n";

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}

