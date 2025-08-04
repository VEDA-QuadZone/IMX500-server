// src/upload/upload_illegal_parking.cpp (with OpenSSL)

#include "detector.hpp"     // detect_illegal_parking_ids(), detect_license_by_id()
#include "config.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <sys/stat.h>
#include <fcntl.h>
#include <unordered_set>
#include <cerrno>
#include <openssl/ssl.h>
#include <openssl/err.h>

// OpenSSL 인증서 경로
constexpr const char* CLIENT_CERT = "/home/sejin/myCA/pi_client/certs/pi.cert.pem";
constexpr const char* CLIENT_KEY  = "/home/sejin/myCA/pi_client/private/pi.key.pem";
constexpr const char* CA_CERT     = "/home/sejin/myCA/certs/ca.cert.pem";
static SSL_CTX* sslCtx = nullptr;

// TLS 초기화
void init_ssl_ctx() {
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();
    const SSL_METHOD* method = TLS_client_method();
    sslCtx = SSL_CTX_new(method);
    if (!sslCtx) {
        ERR_print_errors_fp(stderr);
        exit(1);
    }
    SSL_CTX_use_certificate_file(sslCtx, CLIENT_CERT, SSL_FILETYPE_PEM);
    SSL_CTX_use_PrivateKey_file(sslCtx, CLIENT_KEY, SSL_FILETYPE_PEM);
    if (!SSL_CTX_check_private_key(sslCtx)) {
        std::cerr << "[ERROR] TLS private key mismatch\n";
        exit(1);
    }
    SSL_CTX_load_verify_locations(sslCtx, CA_CERT, nullptr);
    SSL_CTX_set_verify(sslCtx, SSL_VERIFY_PEER, nullptr);
    SSL_CTX_set_verify_depth(sslCtx, 4);
}

// ----------------- 설정 상수 ----------------- //
static const std::string DEFAULT_SHM_DIR         = "/dev/shm";
static const std::string DEFAULT_ONNX_MODEL_PATH = "/home/sejin/myproject/cpp/src/detect/assets/model/best.onnx";
static const std::string DEFAULT_TFLITE_MODEL    = "/home/sejin/myproject/cpp/src/detect/assets/model/model.tflite";
static const std::string DEFAULT_LABEL_PATH      = "/home/sejin/myproject/cpp/src/detect/assets/model/labels.names";

// 커맨드 타입
enum class CmdType { UPLOAD, ADD_HISTORY };

// 큐에 담길 커맨드 구조체
struct Command {
    CmdType     type;
    std::string arg;  // UPLOAD: segment name (no path, no extension), ADD_HISTORY: full command
};

static std::queue<Command>     cmd_queue;
static std::mutex              q_mutex;
static std::condition_variable q_cv;

static std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{}; localtime_r(&t_c, &tm);
    char buf[20]; std::strftime(buf, sizeof(buf), "%F_%T", &tm);
    return buf;
}

static bool send_message(SSL* ssl, const std::string& msg) {
    int sent = SSL_write(ssl, msg.c_str(), msg.size());
    std::cerr << "[DEBUG] send_message: sent " << sent << ": `" << msg.substr(0, msg.find('\n')) << "`\n";
    return sent == (int)msg.size();
}

static std::string ensure_jpg(const std::string& name) {
    if (name.size()>=4 && name.substr(name.size()-4)==".jpg") return name;
    return name + ".jpg";
}

static std::string find_latest_bestshot_for_id(int id) {
    std::string latest;
    std::string pat = "_" + std::to_string(id) + "_";
    for (auto& e : std::filesystem::directory_iterator(DEFAULT_SHM_DIR)) {
        auto fn = e.path().filename().string();
        if (fn.rfind("shm_snapshot_",0)==0 && fn.find(pat)!=std::string::npos) {
            if (fn > latest) latest = fn;
        }
    }
    std::cerr << "[DEBUG] latest bestshot for id=" << id << ": " << latest << "\n";
    return latest;
}

static std::pair<std::string,std::string> find_latest_snapshots_for_id(int id) {
    std::string latest_s, latest_e;
    std::string ps = "shm_startshot_" + std::to_string(id) + "_";
    std::string pe = "shm_endshot_"   + std::to_string(id) + "_";
    for (auto& e : std::filesystem::directory_iterator(DEFAULT_SHM_DIR)) {
        auto fn = e.path().filename().string();
        if (fn.rfind(ps,0)==0 && fn>latest_s) latest_s=fn;
        if (fn.rfind(pe,0)==0 && fn>latest_e) latest_e=fn;
    }
    std::cerr << "[DEBUG] latest for id="<<id<<": start="<<latest_s<<", end="<<latest_e<<"\n";
    return {latest_s, latest_e};
}

static bool upload_shm(SSL* ssl, const std::string& base_name) {
    std::string path = DEFAULT_SHM_DIR + "/" + base_name;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { std::cerr << "[ERROR] open " << path << ": " << strerror(errno) << "\n"; return false; }
    struct stat st;
    if (fstat(fd, &st) < 0) { std::cerr << "[ERROR] fstat " << path << ": " << strerror(errno) << "\n"; close(fd); return false; }
    std::vector<char> buf(st.st_size);
    if (::read(fd, buf.data(), buf.size()) != (ssize_t)buf.size()) { close(fd); return false; }
    close(fd);
    std::string seg = ensure_jpg(base_name);
    std::ostringstream cmd; cmd << "UPLOAD " << seg << " " << buf.size() << "\n";
    if (!send_message(ssl, cmd.str())) return false;
    
    // **이 부분을 반드시 반복으로!**
    size_t total_sent = 0;
    while (total_sent < buf.size()) {
        int n = SSL_write(ssl, buf.data() + total_sent, buf.size() - total_sent);
        if (n <= 0) {
            std::cerr << "[ERROR] SSL_write failed during upload_shm\n";
            return false;
        }
        total_sent += n;
    }
    char ack[256]; int r = SSL_read(ssl, ack, sizeof(ack)-1); ack[r] = '\0';
    std::cerr << "[DEBUG] Received ACK: `" << ack << "`\n";
    return std::string(ack).find("\"status\":\"success\"") != std::string::npos;
}

static void sender_thread(SSL* ssl) {
    std::unique_lock<std::mutex> lk(q_mutex);
    while (true) {
        q_cv.wait(lk, []{ return !cmd_queue.empty(); });
        Command cmd = std::move(cmd_queue.front());
        cmd_queue.pop();
        lk.unlock();
        if (cmd.type == CmdType::UPLOAD) {
            upload_shm(ssl, cmd.arg);
        } else {
            send_message(ssl, cmd.arg);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        lk.lock();
    }
}

int main(){
    std::cerr << "[DEBUG] Starting upload_illegal_parking client\n";
    init_ssl_ctx();

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in serv{}; serv.sin_family = AF_INET;
    serv.sin_port = htons(TCP_SERVER_PORT);
    inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr);
    connect(sock, (sockaddr*)&serv, sizeof(serv));

    SSL* ssl = SSL_new(sslCtx);
    SSL_set_fd(ssl, sock);
    SSL_connect(ssl);

    std::thread(sender_thread, ssl).detach();

    nlohmann::json cfg = {
        {"shm_dir", DEFAULT_SHM_DIR},
        {"onnx_path", DEFAULT_ONNX_MODEL_PATH},
        {"tflite_path", DEFAULT_TFLITE_MODEL},
        {"label_path", DEFAULT_LABEL_PATH}
    };

    std::unordered_set<int> processed_ids;

    while (true) {
        auto ids = detect_illegal_parking_ids();
        for (int id : ids) {
            if (!processed_ids.insert(id).second) continue;
            auto lic = detect_license_by_id(id, cfg);
            std::string plate = (lic.empty()) ? "0000" : lic["plate"].get<std::string>();
            std::string best = find_latest_bestshot_for_id(id);
            auto [start_snap,end_snap] = find_latest_snapshots_for_id(id);
            if (best.empty() || start_snap.empty() || end_snap.empty()) continue;
            {
                std::lock_guard<std::mutex> g(q_mutex);
                cmd_queue.push({CmdType::UPLOAD, best});
                cmd_queue.push({CmdType::UPLOAD, start_snap});
                cmd_queue.push({CmdType::UPLOAD, end_snap});
                std::ostringstream ah;
                ah << "ADD_HISTORY " << current_timestamp() << " images/" << best << ".jpg "
                   << plate << " " << EventCode::ILLEGAL_PARKING << " images/"
                   << start_snap << " images/" << end_snap << "\n";
                cmd_queue.push({CmdType::ADD_HISTORY, ah.str()});
            }
            q_cv.notify_one();
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    SSL_shutdown(ssl);
    SSL_free(ssl);
    close(sock);
    SSL_CTX_free(sslCtx);
    return 0;
}
