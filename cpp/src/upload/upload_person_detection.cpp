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

// 2) "shm_snapshot_<id>_<ts>_<w>x<h>" 패턴 파싱
static bool parse_snapshot_name(const std::string& name,
                                int& out_id,
                                std::string& out_ts,
                                int& out_w,
                                int& out_h) {
    const std::string prefix = "shm_snapshot_";
    if (name.rfind(prefix, 0) != 0) return false;
    auto rest = name.substr(prefix.size());
    auto p1 = rest.find('_');
    auto p2 = rest.find('_', p1 + 1);
    if (p1 == std::string::npos || p2 == std::string::npos) return false;
    out_id = std::stoi(rest.substr(0, p1));
    out_ts = rest.substr(p1 + 1, p2 - (p1 + 1));
    auto wxh = rest.substr(p2 + 1);
    auto x = wxh.find('x');
    if (x == std::string::npos) return false;
    out_w = std::stoi(wxh.substr(0, x));
    out_h = std::stoi(wxh.substr(x + 1));
    return true;
}

// 3) target_id용 SHM 스냅샷을 읽어 Mat(BGR) + 메타정보 반환 (fallback 포함)
static bool load_snapshot(int target_id,
                          cv::Mat& out_bgr,
                          std::string& out_ts,
                          int& out_w,
                          int& out_h) {
    const fs::path shm_dir = "/dev/shm";
    for (auto& e : fs::directory_iterator(shm_dir)) {
        std::string fn = e.path().filename().string();
        int id, w, h;
        std::string ts;
        if (!parse_snapshot_name(fn, id, ts, w, h)) continue;
        if (id != target_id) continue;

        std::ifstream in(e.path(), std::ios::binary | std::ios::ate);
        if (!in.is_open()) continue;
        auto size = in.tellg();
        in.seekg(0);
        std::vector<unsigned char> buf(size);
        in.read(reinterpret_cast<char*>(buf.data()), size);
        if (in.gcount() != size) continue;

        // 1) 제목에 명시된 해상도일 때
        std::size_t expected = std::size_t(w) * h * 4;
        if (size == std::streamoff(expected)) {
            cv::Mat bgra(h, w, CV_8UC4, buf.data());
            cv::cvtColor(bgra, out_bgr, cv::COLOR_BGRA2BGR);
            out_ts = ts; out_w = w; out_h = h;
            return true;
        }

        // 2) fallback 1280×720 시도
        constexpr int fw = 1280, fh = 720;
        std::size_t fallback_sz = std::size_t(fw) * fh * 4;
        if (size == std::streamoff(fallback_sz)) {
            cv::Mat bgra(fh, fw, CV_8UC4, buf.data());
            cv::cvtColor(bgra, out_bgr, cv::COLOR_BGRA2BGR);
            out_ts = ts; out_w = fw; out_h = fh;
            std::cerr << "[!] " << fn << " → fallback(" 
                      << fw << "x" << fh << ")\n";
            return true;
        }

        // 3) 둘 다 아니면 건너뛰기
    }
    return false;
}

// 4) Mat을 파일로 저장 (id, ts, w×h 포함)
static std::string save_image(const cv::Mat& img,
                              int id,
                              const std::string& ts,
                              int w,
                              int h) {
    fs::create_directories("images");
    std::string safe_ts = ts;
    std::replace(safe_ts.begin(), safe_ts.end(), ' ', '_');
    std::replace(safe_ts.begin(), safe_ts.end(), ':', '-');

    std::string fname = "person_"
                      + std::to_string(id) + "_"
                      + safe_ts + "_"
                      + std::to_string(w) + "x"
                      + std::to_string(h) + ".jpg";
    std::string path = "images/" + fname;

    if (!cv::imwrite(path, img)) {
        std::cerr << "[ERROR] 이미지 저장 실패: " << path << "\n";
        return "";
    }
    return path;
}

// 5) TCP로 ADD_HISTORY 전송
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
                    + std::to_string(event_type)
                    + "\n";
    if (send(sock, cmd.c_str(), cmd.size(), 0) < 0) {
        perror("send"); close(sock); return false;
    }
    close(sock);
    std::cout << "[TCP] Sent: " << cmd;
    return true;
}

// 6) TCP로 UPLOAD <filename> <filesize>\n + 바이너리
static bool send_upload_image(const std::string& img_path) {
    std::error_code ec;
    auto size = fs::file_size(img_path, ec);
    if (ec) {
        std::cerr<<"[ERROR] file_size 오류: "<<ec.message()<<"\n";
        return false;
    }

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

    std::string header = "UPLOAD "
                       + img_path + " "
                       + std::to_string(size)
                       + "\n";
    if (send(sock, header.c_str(), header.size(), 0) < 0) {
        perror("send header"); close(sock); return false;
    }

    std::ifstream in(img_path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr<<"[ERROR] 이미지 파일 오픈 실패: "<<img_path<<"\n";
        close(sock);
        return false;
    }
    const size_t BUF_SZ = 4096;
    std::vector<char> buf(BUF_SZ);
    while (in.good()) {
        in.read(buf.data(), BUF_SZ);
        std::streamsize r = in.gcount();
        if (r > 0) {
            if (send(sock, buf.data(), r, 0) < 0) {
                perror("send data");
                in.close();
                close(sock);
                return false;
            }
        }
    }
    in.close();
    close(sock);

    std::cout<<"[TCP] UPLOAD sent: "<<header;
    return true;
}

int main() {
    std::set<int> prev_ids, reported;
    while (true) {
        auto ids = detect_persons({}); 
        std::set<int> current(ids.begin(), ids.end());

        // 사라진 ID만 처리
        for (int id : prev_ids) {
            if (reported.count(id)) continue;

            cv::Mat img;
            std::string ts;
            int w, h;
            if (!load_snapshot(id, img, ts, w, h))
                continue;

            // 1) 파일로 저장
            std::string path = save_image(img, id, ts, w, h);
            if (path.empty())
                continue;

            // 2) 이미지 UPLOAD
            if (!send_upload_image(path))
                continue;

            // 3) ADD_HISTORY
            if (send_add_history(ts, path, "-", 2)) {
                reported.insert(id);
            }
        }

        prev_ids = std::move(current);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}
