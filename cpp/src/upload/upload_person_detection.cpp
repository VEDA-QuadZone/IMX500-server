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
#include <nlohmann/json.hpp>

#include "detector.hpp"   // std::vector<int> detect_persons();
#include "config.hpp"     // TCP_SERVER_IP, TCP_SERVER_PORT

namespace fs = std::filesystem;
using json = nlohmann::json;

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

    auto core = name.substr(prefix.size());
    auto p1 = core.find('_');
    auto p2 = core.find('_', p1 + 1);
    if (p1 == std::string::npos || p2 == std::string::npos) return false;

    out_slot = std::stoi(core.substr(0, p1));
    out_id   = std::stoi(core.substr(p1 + 1, p2 - (p1 + 1)));
    out_ts   = core.substr(p2 + 1);
    return true;
}

// 3) SHM에서 snapshot을 읽어 Mat(BGR)과 timestamp 반환
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

        std::ifstream in(e.path(), std::ios::binary | std::ios::ate);
        if (!in.is_open()) continue;
        auto sz = in.tellg();
        in.seekg(0);
        std::vector<unsigned char> buf(sz);
        in.read(reinterpret_cast<char*>(buf.data()), sz);
        in.close();

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

// 4) 메모리 버퍼(buf)와 filename으로 서버에 바로 전송
static bool send_upload_image(const std::vector<uchar>& buf,
                               const std::string& filename) {
    size_t size = buf.size();
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return false; }
    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port   = htons(TCP_SERVER_PORT);
    if (inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr) <= 0 ||
        connect(sock, (sockaddr*)&serv, sizeof(serv)) < 0) {
        perror("connect"); close(sock); return false;
    }

    std::string header = "UPLOAD " + filename + " " + std::to_string(size) + "\n";
    if (send(sock, header.data(), header.size(), 0) < 0) {
        perror("send header"); close(sock); return false;
    }

    size_t sent = 0;
    while (sent < size) {
        ssize_t s = send(sock, buf.data() + sent, size - sent, 0);
        if (s <= 0) { perror("send data"); close(sock); return false; }
        sent += s;
    }

    if (shutdown(sock, SHUT_WR) < 0) perror("shutdown UPLOAD SHUT_WR");

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

    try {
        auto j = json::parse(resp);
        if (j.value("status", "error") != "success" || j.value("code", 0) != 200) {
            std::cerr << "[ERROR] UPLOAD failed: " << j.value("message", "unknown error") << std::endl;
            close(sock);
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] JSON parse error: " << e.what() << std::endl;
        close(sock);
        return false;
    }

    close(sock);
    return true;
}

// 5) ADD_HISTORY
static bool send_add_history(const std::string& date,
                             const std::string& img_path,
                             const std::string& plate,
                             int event_type) {
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

    std::string cmd = "ADD_HISTORY "
                    + date + " "
                    + img_path + " "
                    + plate + " "
                    + std::to_string(event_type) + "\n";
    if (send(sock, cmd.c_str(), cmd.size(), 0) < 0) {
        perror("send"); close(sock); return false;
    }
    if (shutdown(sock, SHUT_WR) < 0) perror("shutdown ADD_HISTORY SHUT_WR");

    char resp_buf[1024];
    int n = recv(sock, resp_buf, sizeof(resp_buf)-1, 0);
    if (n > 0) {
        resp_buf[n] = '\0';
        std::cout << "[TCP] Server response to ADD_HISTORY: " << resp_buf << std::endl;
    } else if (n == 0) {
        std::cout << "[TCP] Server closed connection after ADD_HISTORY" << std::endl;
    } else if (errno == ECONNRESET) {
        std::cout << "[TCP] Connection reset by peer after ADD_HISTORY; assuming success" << std::endl;
    } else {
        perror("recv");
    }
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
            if (!load_snapshot(id, img, ts)) continue;

            // 1) 메모리에서 JPEG로 인코딩 않고 바로 서버 업로드
            std::vector<uchar> buf;
            if (!cv::imencode(".jpg", img, buf)) {
                std::cerr << "[ERROR] imencode failed for ID=" << id << "\n";
                continue;
            }
            std::string safe_ts = ts;
            std::replace(safe_ts.begin(), safe_ts.end(), ' ', '_');
            std::replace(safe_ts.begin(), safe_ts.end(), ':', '-');
            std::string filename = "person_" + std::to_string(id) + "_" + safe_ts + ".jpg";

            if (!send_upload_image(buf, filename)) continue;

            // 3) ADD_HISTORY
            if (!ts.empty()) {
                std::string date = ts.substr(0,4) + "-" + ts.substr(4,2) + "-" + ts.substr(6,2) + "_"
                                 + ts.substr(9,2) + ":" + ts.substr(11,2) + ":" + ts.substr(13,2);
                if (send_add_history(date, filename, "-", 2)) {
                    reported.insert(id);
                }
            }
        }

        prev_ids = std::move(current);
        std::cout << "[DEBUG] Loop tick, prev_ids size=" << prev_ids.size() << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}
