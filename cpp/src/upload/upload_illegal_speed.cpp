#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <openssl/ssl.h>
#include <openssl/err.h>

#include "detector.hpp"    // detect_license_by_id()
#include "config.hpp"      // extern const std::string TCP_SERVER_IP; extern const int TCP_SERVER_PORT;

using json = nlohmann::json;

// ——— TLS 초기화 —————————————————————————————————

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

    // 고객 인증서 + 키 (pi_client)
    SSL_CTX_use_certificate_file(clientCtx,
        "/home/sejin/myCA/pi_client/certs/pi.cert.pem",
        SSL_FILETYPE_PEM);
    SSL_CTX_use_PrivateKey_file(clientCtx,
        "/home/sejin/myCA/pi_client/private/pi.key.pem",
        SSL_FILETYPE_PEM);
    if (!SSL_CTX_check_private_key(clientCtx)) {
        std::cerr << "Private key does not match certificate\n";
        std::exit(1);
    }

    // CA 로드 (서버 인증서 검증)
    SSL_CTX_load_verify_locations(clientCtx,
        "/home/sejin/myCA/certs/ca.cert.pem", nullptr);
    SSL_CTX_set_verify(clientCtx, SSL_VERIFY_PEER, nullptr);
    SSL_CTX_set_verify_depth(clientCtx, 4);
}

// ——— 현재 시각 포맷 함수 ———————————————————————————

static std::string get_now_date() {
    time_t t = time(nullptr);
    struct tm tm;
    localtime_r(&t, &tm);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d_%H:%M:%S", &tm);
    return buf;
}

// ——— 파일 전송 (SSL_write 버전) ——————————————————————

bool send_file_ssl(SSL* ssl, const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file) return false;
    size_t filesize = file.tellg();
    file.seekg(0);

    std::vector<char> buf(4096);
    while (filesize > 0) {
        size_t chunk = std::min(buf.size(), filesize);
        file.read(buf.data(), chunk);
        int sent = SSL_write(ssl, buf.data(), chunk);
        if (sent <= 0) return false;
        filesize -= sent;
    }
    return true;
}

// ——— 메인 루프 ——————————————————————————————————

int main() {
    initClientCtx();

    // ZeroMQ SUB 초기화
    zmq::context_t ctx(1);
    zmq::socket_t sub(ctx, zmq::socket_type::sub);
    sub.connect("ipc:///tmp/speed_detector.ipc");
    sub.setsockopt(ZMQ_SUBSCRIBE, "", 0);

    const char* SERVER_IP = TCP_SERVER_IP.c_str();
    const int   SERVER_PORT = TCP_SERVER_PORT;

    while (true) {
        zmq::message_t frame;
        sub.recv(frame, zmq::recv_flags::none);

        auto msg = json::parse(frame.to_string());
        if (msg["event"] != "ILLEGAL_SPEEDING") continue;

        int car_id   = msg["id"];
        float speed  = msg["speed"];
        // 번호판/스냅샷 찾기
        json result = detect_license_by_id(car_id);
        if (result.is_null() || result["shm_name"].empty()) {
            std::cerr << "[ERROR] No snapshot for id=" << car_id << "\n";
            continue;
        }
        std::string shm_name = result["shm_name"];
        std::string plate    = result.value("plate", "");
        if (plate.empty()) plate = "0000";

        // 1) TCP 소켓 + TLS 핸드쉐이크
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) { perror("socket"); continue; }
        sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port   = htons(SERVER_PORT);
        inet_pton(AF_INET, SERVER_IP, &serv_addr.sin_addr);
        if (connect(sock, (sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
            perror("connect"); close(sock); continue;
        }
        SSL* ssl = SSL_new(clientCtx);
        SSL_set_fd(ssl, sock);
        if (SSL_connect(ssl) <= 0) {
            ERR_print_errors_fp(stderr);
            SSL_free(ssl);
            close(sock);
            continue;
        }

        // 2) UPLOAD 명령
        std::string snapshot_path = "/dev/shm/" + shm_name;
        std::ifstream infile(snapshot_path, std::ios::binary | std::ios::ate);
        if (!infile) { std::cerr << "[ERROR] open shm\n"; SSL_shutdown(ssl); SSL_free(ssl); close(sock); continue; }
        size_t filesize = infile.tellg();
        infile.close();

        std::string upload_cmd = "UPLOAD " + shm_name + ".jpg " + std::to_string(filesize) + "\n";
        SSL_write(ssl, upload_cmd.c_str(), upload_cmd.size());

        // 3) 파일 바디 전송
        if (!send_file_ssl(ssl, snapshot_path)) {
            std::cerr << "[ERROR] file send\n";
            SSL_shutdown(ssl); SSL_free(ssl); close(sock);
            continue;
        }

        // 4) ADD_HISTORY 명령
        std::ostringstream oss;
        oss << "ADD_HISTORY "
            << get_now_date() << " "
            << "images/" << shm_name << ".jpg "
            << plate << " 1 - - "
            << std::fixed << std::setprecision(2) << speed
            << "\n";
        std::string ah = oss.str();
        SSL_write(ssl, ah.c_str(), ah.size());

        // (선택) 서버 응답 수집/로깅 생략

        // 정리
        SSL_shutdown(ssl);
        SSL_free(ssl);
        close(sock);

        std::cout << "[UPLOAD][SPEED] id=" << car_id
                  << " plate=" << plate << " speed=" << speed << "\n";
    }

    SSL_CTX_free(clientCtx);
    return 0;
}
