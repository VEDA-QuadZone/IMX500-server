// src/upload/upload_illegal_parking.cpp

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


// ----------------- 설정 상수 ----------------- //
static const std::string DEFAULT_SHM_DIR         = "/dev/shm";
static const std::string DEFAULT_ONNX_MODEL_PATH = "../src/detect/assets/model/best.onnx";
static const std::string DEFAULT_TFLITE_MODEL    = "../src/detect/assets/model/model.tflite";
static const std::string DEFAULT_LABEL_PATH      = "../src/detect/assets/model/labels.names";

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

// 현재 시각 "YYYY-MM-DD_HH:MM:SS" 반환
static std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm{}; localtime_r(&t_c, &tm);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%F_%T", &tm);
    return buf;
}

// 소켓으로 문자열 메시지 전송 (디버그 포함)
static bool send_message(int sock, const std::string& msg) {
    ssize_t sent = send(sock, msg.c_str(), msg.size(), 0);
    std::cerr << "[DEBUG] send_message: sent " << sent
              << " bytes: `" << msg.substr(0, msg.find('\n')) << "`\n";
    return sent == (ssize_t)msg.size();
}

// ".jpg" 보장
static std::string ensure_jpg(const std::string& name) {
    if (name.size()>=4 && name.substr(name.size()-4)==".jpg")
        return name;
    return name + ".jpg";
}

// 가장 최신 best샷 파일명 찾기 ("shm_snapshot_<slot>_<id>_<ts>")
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

// 최신 startshot/endshot 한 쌍 찾기
static std::pair<std::string,std::string> find_latest_snapshots_for_id(int id) {
    std::string latest_s, latest_e;
    std::string ps = "shm_startshot_" + std::to_string(id) + "_";
    std::string pe = "shm_endshot_"   + std::to_string(id) + "_";
    for (auto& e : std::filesystem::directory_iterator(DEFAULT_SHM_DIR)) {
        auto fn = e.path().filename().string();
        if (fn.rfind(ps,0)==0 && fn>latest_s) latest_s=fn;
        if (fn.rfind(pe,0)==0 && fn>latest_e) latest_e=fn;
    }
    std::cerr << "[DEBUG] latest for id="<<id
              <<": start="<<latest_s
              <<", end="<<latest_e<<"\n";
    return {latest_s, latest_e};
}

// UPLOAD + 바이너리 + ACK 대기
static bool upload_shm(int sock, const std::string& base_name) {
    // 1) 파일 열기 (base_name 에 확장자 없음)
    std::string path = DEFAULT_SHM_DIR + "/" + base_name;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "[ERROR] open failed for " << path
                  << ": " << strerror(errno) << "\n";
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        std::cerr << "[ERROR] fstat failed for " << path
                  << ": " << strerror(errno) << "\n";
        close(fd);
        return false;
    }

    // 2) 데이터 읽기
    std::vector<char> buf(st.st_size);
    if (::read(fd, buf.data(), buf.size()) != (ssize_t)buf.size()) {
        std::cerr << "[ERROR] read failed for " << path << "\n";
        close(fd);
        return false;
    }
    close(fd);

    // 3) UPLOAD 명령 전송 (seg = base_name+".jpg")
    std::string seg = ensure_jpg(base_name);
    {
        std::ostringstream cmd;
        cmd << "UPLOAD " << seg << " " << buf.size() << "\n";
        if (!send_message(sock, cmd.str())) return false;
    }

    // 4) 바이너리 데이터 전송
    ssize_t sent = send(sock, buf.data(), buf.size(), 0);
    if (sent != (ssize_t)buf.size()) {
        std::cerr << "[ERROR] send binary failed for " << seg
                  << ": sent " << sent << "/" << buf.size() << "\n";
        return false;
    }
    std::cerr << "[DEBUG] Sent binary for " << seg
              << " (" << sent << " bytes)\n";

    // 5) ACK 타임아웃 설정
    struct timeval tv{5,0};
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    // 6) ACK 대기
    std::cerr << "[DEBUG] waiting for ACK for " << seg << "\n";
    std::string ack;
    char buf_ack[256];
    while (true) {
        ssize_t r = recv(sock, buf_ack, sizeof(buf_ack)-1, 0);
        if (r < 0) {
            if (errno==EAGAIN||errno==EWOULDBLOCK)
                std::cerr<<"[ERROR] ACK timeout for "<<seg<<"\n";
            else
                std::cerr<<"[ERROR] recv failed for "<<seg
                         <<": "<<strerror(errno)<<"\n";
            return false;
        }
        if (r == 0) {
            std::cerr<<"[ERROR] connection closed waiting ACK\n";
            return false;
        }
        buf_ack[r] = '\0';
        ack += buf_ack;
        if (ack.find('}')!=std::string::npos || ack.find('\n')!=std::string::npos)
            break;
    }
    std::cerr<<"[DEBUG] Received ACK: `"<<ack<<"`\n";
    if (ack.find("\"status\":\"success\"")==std::string::npos) {
        std::cerr<<"[ERROR] negative ACK for "<<seg<<"\n";
        return false;
    }
    return true;
}

// 전송 전용 스레드
static void sender_thread(int sock) {
    std::unique_lock<std::mutex> lk(q_mutex);
    while (true) {
        q_cv.wait(lk, []{ return !cmd_queue.empty(); });
        Command cmd = std::move(cmd_queue.front());
        cmd_queue.pop();
        lk.unlock();

        if (cmd.type == CmdType::UPLOAD) {
            upload_shm(sock, cmd.arg);
        } else {
            send_message(sock, cmd.arg);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        lk.lock();
    }
}

int main(){
    std::cerr << "[DEBUG] Starting upload_illegal_parking client\n";

    // 1) TCP 서버 연결
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return 1; }
    sockaddr_in serv{};
    serv.sin_family = AF_INET;
    serv.sin_port   = htons(TCP_SERVER_PORT);
    inet_pton(AF_INET, TCP_SERVER_IP.c_str(), &serv.sin_addr);
    if (connect(sock, (sockaddr*)&serv, sizeof(serv)) < 0) {
        perror("connect"); return 1;
    }
    std::cerr << "[DEBUG] Connected to " << TCP_SERVER_IP << ":" << TCP_SERVER_PORT << "\n";

    // 2) OCR 설정 JSON
    nlohmann::json cfg = {
        {"shm_dir",    DEFAULT_SHM_DIR},
        {"onnx_path",  DEFAULT_ONNX_MODEL_PATH},
        {"tflite_path",DEFAULT_TFLITE_MODEL},
        {"label_path", DEFAULT_LABEL_PATH}
    };

    // sender 스레드 시작
    std::thread(sender_thread, sock).detach();

    std::unordered_set<int> processed_ids;

    while (true) {
        std::cerr << "[DEBUG] detect_illegal_parking_ids()...\n";
        auto ids = detect_illegal_parking_ids();
        for (int id : ids) {
            if (!processed_ids.insert(id).second) continue;

            // plate 정보는 OCR로
            auto lic = detect_license_by_id(id, cfg);
            // if (lic.empty()) continue;
            // std::string plate = lic["plate"].get<std::string>();
            std::string plate;
if (lic.empty()) {
    plate = "0000";          // 못 읽었을 때 0000으로 대체
} else {
    plate = lic["plate"].get<std::string>();
}
            // best, start, end 샷 찾기
            std::string best        = find_latest_bestshot_for_id(id);
            auto [start_snap,end_snap] = find_latest_snapshots_for_id(id);
            if (best.empty() || start_snap.empty() || end_snap.empty()) continue;

            {
                std::lock_guard<std::mutex> g(q_mutex);
                cmd_queue.push({CmdType::UPLOAD,      best});
                cmd_queue.push({CmdType::UPLOAD,      start_snap});
                cmd_queue.push({CmdType::UPLOAD,      end_snap});

                std::ostringstream ah;
                ah << "ADD_HISTORY "
                   << current_timestamp() << " "
                   << "images/" << best << ".jpg" << " "
                   << plate << " "
                   << EventCode::ILLEGAL_PARKING << " "
                   << "images/" << start_snap << " "
                   << "images/" << end_snap << "\n";
                cmd_queue.push({CmdType::ADD_HISTORY, ah.str()});
            }
            q_cv.notify_one();
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    close(sock);
    return 0;
}
