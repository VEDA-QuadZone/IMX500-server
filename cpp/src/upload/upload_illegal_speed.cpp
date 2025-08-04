#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <zmq.hpp>
#include <arpa/inet.h>
#include <unistd.h>
#include "config.hpp"
#include "detector.hpp" // detect_license_by_id()

using json = nlohmann::json;
namespace fs = std::filesystem;

// 인증서 경로
constexpr const char* CLIENT_CERT = "/home/sejin/myCA/pi_client/certs/pi.cert.pem";
constexpr const char* CLIENT_KEY  = "/home/sejin/myCA/pi_client/private/pi.key.pem";
constexpr const char* CA_CERT     = "/home/sejin/myCA/certs/ca.cert.pem";

// 전역 SSL 컨텍스트
static SSL_CTX* clientCtx = nullptr;

// 1) TLS 컨텍스트 초기화
void initClientCtx() {
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();
    clientCtx = SSL_CTX_new(TLS_client_method());
    if (!clientCtx) {
        ERR_print_errors_fp(stderr);
        std::exit(1);
    }

    if (SSL_CTX_use_certificate_file(clientCtx, CLIENT_CERT, SSL_FILETYPE_PEM) <= 0)
        ERR_print_errors_fp(stderr);
    if (SSL_CTX_use_PrivateKey_file(clientCtx, CLIENT_KEY, SSL_FILETYPE_PEM) <= 0)
        ERR_print_errors_fp(stderr);
    if (!SSL_CTX_check_private_key(clientCtx)) {
        std::cerr << "[ERROR] Private key does not match certificate\n";
        std::exit(1);
    }
    if (!SSL_CTX_load_verify_locations(clientCtx, CA_CERT, nullptr))
        ERR_print_errors_fp(stderr);

    SSL_CTX_set_verify(clientCtx, SSL_VERIFY_PEER, nullptr);
    SSL_CTX_set_verify_depth(clientCtx, 4);
}

// 2) 현재 시각 문자열 (예: "2025-07-25_11-45-30")
std::string now_str() {
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&t_c, &tm);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%F_%T", &tm);
    return std::string(buf);
}

// 3) SSL 연결 열기 (소켓 + 핸드셰이크)
bool connectSSL(SSL*& ssl, int& sock) {
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket");
        return false;
    }

    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port   = htons(TCP_SERVER_PORT);
    if (inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr) != 1) {
        std::cerr << "[ERROR] Invalid server IP\n";
        close(sock);
        return false;
    }

    if (connect(sock, (sockaddr*)&serv, sizeof(serv)) < 0) {
        perror("connect");
        close(sock);
        return false;
    }

    ssl = SSL_new(clientCtx);
    SSL_set_fd(ssl, sock);
    if (SSL_connect(ssl) <= 0) {
        ERR_print_errors_fp(stderr);
        SSL_free(ssl);
        close(sock);
        return false;
    }

    return true;
}

// 4) 파일 업로드
bool send_upload_image(SSL* ssl, const std::vector<uchar>& buf, const std::string& filename) {
    size_t size = buf.size();
    std::string header = "UPLOAD " + filename + " " + std::to_string(size) + "\n";
    if (SSL_write(ssl, header.c_str(), header.size()) <= 0) {
        std::cerr << "[ERROR] SSL_write header\n";
        return false;
    }

    size_t sent = 0;
    while (sent < size) {
        int w = SSL_write(ssl, buf.data() + sent, size - sent);
        if (w <= 0) {
            std::cerr << "[ERROR] SSL_write body\n";
            return false;
        }
        sent += w;
    }

    // 서버 응답(JSON) 수신
    std::string resp;
    char rbuf[512];
    while (true) {
        int n = SSL_read(ssl, rbuf, sizeof(rbuf));
        if (n <= 0) break;
        resp.append(rbuf, n);
        if (resp.find('}') != std::string::npos) break;
    }
    std::cout << "[TCP] UPLOAD resp: " << resp << "\n";

    try {
        auto j = json::parse(resp);
        return (j.value("status", "") == "success");
    } catch (...) {
        return false;
    }
}

// 5) ADD_HISTORY 전송
bool send_add_history(SSL* ssl,
                      const std::string& date,
                      const std::string& filename,
                      const std::string& plate,
                      float speed) {
    std::ostringstream oss;
    oss << "ADD_HISTORY " << date
        << " images/" << filename
        << " " << plate
        << " 1 - - "
        << std::fixed << std::setprecision(2) << speed << "\n";

    std::string cmd = oss.str();
    if (SSL_write(ssl, cmd.c_str(), cmd.size()) <= 0) {
        std::cerr << "[ERROR] SSL_write ADD_HISTORY\n";
        return false;
    }

    // 서버 응답(JSON) 수신
    std::string resp;
    char buf[512];
    while (true) {
        int n = SSL_read(ssl, buf, sizeof(buf));
        if (n <= 0) {
            std::cerr << "[ERROR] SSL_read failed after ADD_HISTORY\n";
            return false;
        }
        resp.append(buf, n);
        if (resp.find('}') != std::string::npos) break;
    }
    std::cout << "[TCP] ADD_HISTORY resp: " << resp << "\n";

    try {
        auto j = json::parse(resp);
        return (j.value("status", "") == "success");
    } catch (...) {
        std::cerr << "[ERROR] Invalid JSON in ADD_HISTORY resp\n";
        return false;
    }
}

int main() {
    // 1) SSL 컨텍스트 생성
    initClientCtx();

    // 2) 최초 연결
    SSL* ssl = nullptr;
    int sock = -1;
    if (!connectSSL(ssl, sock)) {
        std::cerr << "[FATAL] Could not establish initial SSL connection\n";
        return 1;
    }

    // 3) ZMQ SUB 설정
    zmq::context_t zmq_ctx(1);
    zmq::socket_t sub(zmq_ctx, zmq::socket_type::sub);
    sub.connect("ipc:///tmp/speed_detector.ipc");
    sub.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    // 4) 메인 루프
    while (true) {
        zmq::message_t frame;
        sub.recv(frame, zmq::recv_flags::none);
        auto msg = json::parse(frame.to_string());
        if (msg["event"] != "ILLEGAL_SPEEDING") continue;

        int car_id = msg["id"];
        float speed = msg["speed"];

        // 번호판 인식
        json result = detect_license_by_id(car_id);
        if (result.is_null() || result["shm_name"].empty()) {
            std::cerr << "[ERROR] No snapshot/plate for id=" << car_id << "\n";
            continue;
        }

        std::string shm_name = result["shm_name"];
        std::string plate    = result.value("plate", "0000");

        // SHM에서 JPEG 읽기
        std::string snapshot_path = "/dev/shm/" + shm_name;
        std::ifstream in(snapshot_path, std::ios::binary | std::ios::ate);
        if (!in.is_open()) {
            std::cerr << "[ERROR] Cannot open " << snapshot_path << "\n";
            continue;
        }
        auto sz = in.tellg();
        in.seekg(0);
        std::vector<uchar> jpeg_buf(sz);
        in.read(reinterpret_cast<char*>(jpeg_buf.data()), sz);
        in.close();
        // 변경: 데모용 이미지 경로 생성 및 읽기
        // std::ostringstream demo_path_oss;
        // demo_path_oss << "/home/sejin/demo/speed_" << car_id << ".jpg";
        // std::string demo_path = demo_path_oss.str();

        // std::ifstream in(demo_path, std::ios::binary | std::ios::ate);
        // if (!in.is_open()) {
        //     std::cerr << "[ERROR] Cannot open " << demo_path << "\n";
        //     continue;
        // }
        // auto sz = in.tellg();
        // in.seekg(0);
        // std::vector<uchar> jpeg_buf(sz);
        // in.read(reinterpret_cast<char*>(jpeg_buf.data()), sz);
        // in.close();


        // 파일 이름/날짜 생성
        std::string timestamp = now_str();
       std::replace(timestamp.begin(), timestamp.end(), ':', '-');
        std::string filename  = shm_name + "_" + timestamp + ".jpg";
       std::string date      = timestamp;
//std::string filename = shm_name + ".jpg";  
        // 5) UPLOAD 시도 (실패 시 재연결 & 재시도)
        if (!send_upload_image(ssl, jpeg_buf, filename)) {
            std::cerr << "[WARN] UPLOAD failed, reconnecting...\n";
            SSL_shutdown(ssl);
            SSL_free(ssl);
            close(sock);
            if (connectSSL(ssl, sock)) {
                send_upload_image(ssl, jpeg_buf, filename);
            } else {
                std::cerr << "[ERROR] Reconnect failed, skipping this event\n";
                continue;
            }
        }

        // 6) ADD_HISTORY 시도
        if (!send_add_history(ssl, date, filename, plate, speed)) {
            std::cerr << "[WARN] ADD_HISTORY failed, reconnecting...\n";
            SSL_shutdown(ssl);
            SSL_free(ssl);
            close(sock);
            if (connectSSL(ssl, sock)) {
                send_add_history(ssl, date, filename, plate, speed);
            } else {
                std::cerr << "[ERROR] Reconnect failed, skipping history\n";
            }
        }

        std::cout << "[UPLOAD][SPEED] id=" << car_id
                  << " plate=" << plate << " sent.\n";
    }

    // 종료 시 정리
    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(sock);
    SSL_CTX_free(clientCtx);
    return 0;
}
