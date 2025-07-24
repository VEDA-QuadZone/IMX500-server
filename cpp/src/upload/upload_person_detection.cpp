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
#include <openssl/ssl.h>
#include <openssl/err.h>
#include "detector.hpp"   // std::vector<int> detect_persons();
#include "config.hpp"     // TCP_SERVER_IP, TCP_SERVER_PORT

namespace fs = std::filesystem;
using json = nlohmann::json;
//openssl
static SSL_CTX* clientCtx = nullptr;

static void initClientCtx() {
    SSL_library_init();
    OpenSSL_add_all_algorithms();
    SSL_load_error_strings();

    const SSL_METHOD* method = TLS_client_method();
    clientCtx = SSL_CTX_new(method);
    if (!clientCtx) {
        ERR_print_errors_fp(stderr);
        std::exit(1);
    }

    // 클라이언트 인증서 + 키 로드
    if (SSL_CTX_use_certificate_file(clientCtx,
            "/home/sejin/myCA/pi_client/certs/pi.cert.pem",
            SSL_FILETYPE_PEM) <= 0)
        ERR_print_errors_fp(stderr);

    if (SSL_CTX_use_PrivateKey_file(clientCtx,
            "/home/sejin/myCA/pi_client/private/pi.key.pem",
            SSL_FILETYPE_PEM) <= 0)
        ERR_print_errors_fp(stderr);

    if (!SSL_CTX_check_private_key(clientCtx)) {
        std::cerr << "Private key does not match the certificate\n";
        std::exit(1);
    }

    // CA 로드 (서버 인증서 검증용)
    if (!SSL_CTX_load_verify_locations(clientCtx,
            "/home/sejin/myCA/certs/ca.cert.pem", nullptr))
        ERR_print_errors_fp(stderr);

    SSL_CTX_set_verify(clientCtx, SSL_VERIFY_PEER, nullptr);
    SSL_CTX_set_verify_depth(clientCtx, 4);
}
// 1) 현재 시각을 "YYYY-MM-DD HH:MM:SS" 로 반환
static std::string now_str() {
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t_c, &tm);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%F_%T", &tm);
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
static bool send_upload_image(SSL* ssl,
                              const std::vector<uchar>& buf,
                              const std::string& filename) {
    size_t size = buf.size();
    // 헤더 전송
    std::string header = "UPLOAD " + filename + " " + std::to_string(size) + "\n";
    if (SSL_write(ssl, header.data(), header.size()) <= 0) {
        perror("SSL_write header");
        return false;
    }
    // 바디 전송
    size_t sent = 0;
    while (sent < size) {
        int w = SSL_write(ssl, buf.data()+sent, size-sent);
        if (w <= 0) {
            perror("SSL_write data");
            return false;
        }
        sent += w;
    }
    // 응답 수신
    std::string resp;
    char rbuf[512];
    while (true) {
        int n = SSL_read(ssl, rbuf, sizeof(rbuf));
        if (n <= 0) break;
        resp.append(rbuf, n);
        if (resp.find('}') != std::string::npos) break;
    }
    std::cout << "[TCP] UPLOAD resp: " << resp << "\n";
    // 결과 파싱
    try {
        auto j = json::parse(resp);
        return (j.value("status","") == "success" && j.value("code",0) == 200);
    } catch (...) {
        return false;
    }
}

// 5) ADD_HISTORY
static bool send_add_history(SSL* ssl,
                             const std::string& date,
                             const std::string& img,
                             const std::string& plate,
                             int event_type) {
    std::string cmd = "ADD_HISTORY " + date
                        + " images/" + img
                        + " " + plate
                        + " " + std::to_string(event_type)
                        + "\n";
    if (SSL_write(ssl, cmd.c_str(), cmd.size()) <= 0) {
        perror("SSL_write ADD_HISTORY");
        return false;
    }
    char buf2[1024];
    int n = SSL_read(ssl, buf2, sizeof(buf2)-1);
    if (n > 0) {
        buf2[n] = '\0';
        std::cout << "[TCP] ADD_HISTORY resp: " << buf2 << "\n";
    }
    return true;
}

int main() {
    initClientCtx();

    // 1) TCP 연결 + TLS 핸드쉐이크 (한 번만!)
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port   = htons(TCP_SERVER_PORT);
    inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr);
    if (connect(sock, (sockaddr*)&serv, sizeof(serv)) < 0) {
        perror("connect"); return 1;
    }
    SSL* ssl = SSL_new(clientCtx);
    SSL_set_fd(ssl, sock);
    if (SSL_connect(ssl) <= 0) {
        ERR_print_errors_fp(stderr);
        return 1;
    }
    std::cout << "[+] TLS connection established\n";

    std::set<int> prev_ids, reported;

    while (true) {
        auto ids = detect_persons();
        std::set<int> current(ids.begin(), ids.end());

        for (int id : prev_ids) {
            if (reported.count(id)) continue;

            cv::Mat img; std::string ts;
            if (!load_snapshot(id, img, ts)) continue;

            // encode
            std::vector<uchar> buf;
            if (!cv::imencode(".jpg", img, buf)) continue;
            std::string safe_ts = ts;
            std::replace(safe_ts.begin(), safe_ts.end(), ' ', '_');
            std::replace(safe_ts.begin(), safe_ts.end(), ':', '-');
            std::string filename = "person_" + std::to_string(id) + "_" + safe_ts + ".jpg";

            if (!send_upload_image(ssl, buf, filename)) continue;

                      // ADD_HISTORY: 지금 시간 사용
            std::string date = now_str();
            if (send_add_history(ssl, date, filename, "-", 2))
                reported.insert(id);
        }

        prev_ids = std::move(current);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // cleanup (도달하지 않지만)
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(sock);
    SSL_CTX_free(clientCtx);
    return 0;
}