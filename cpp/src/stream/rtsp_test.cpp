#include <iostream>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <csignal>

std::thread mediamtx_thread;
std::thread stream_thread;
bool running = true;

// MediaMTX 실행 (백그라운드)
void start_mediamtx() {
    std::cout << "[INFO] Starting MediaMTX..." << std::endl;
    int ret = std::system("./mediamtx ./mediamtx.yml &");
    if (ret != 0) {
        std::cerr << "[ERROR] Failed to launch MediaMTX (code: " << ret << ")" << std::endl;
    }
}

// libcamera-vid → stdout → ffmpeg → RTSP 송출
void start_streaming() {
    const std::string command =
        "libcamera-vid -t 0 --codec h264 --inline --flush "
        "--width 1920 --height 1080 --framerate 30 --output - "
        "| ffmpeg -re -i - -vcodec copy -f rtsp rtsp://localhost:8554/mystream";

    std::cout << "[INFO] Starting video stream via libcamera-vid + ffmpeg..." << std::endl;
    int ret = std::system(command.c_str());

    if (ret != 0) {
        std::cerr << "[ERROR] Streaming command failed with code: " << ret << std::endl;
    }
}

// SIGINT (Ctrl+C) 처리용
void signal_handler(int) {
    std::cout << "\n[INFO] Caught SIGINT. Exiting..." << std::endl;
    running = false;
    std::exit(0);
}

int main() {
    std::signal(SIGINT, signal_handler);

    // 1) MediaMTX 실행
    mediamtx_thread = std::thread(start_mediamtx);

    // 2) 약간 대기 후 ffmpeg + libcamera-vid 실행 (MediaMTX 먼저 준비되도록)
    std::this_thread::sleep_for(std::chrono::seconds(2));
    stream_thread = std::thread(start_streaming);

    // 3) 메인 루프: 상태 출력
    while (running) {
        std::cout << "[INFO] RTSP server running at rtsp://<IP>:8554/mystream" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(60));
    }

    stream_thread.join();
    mediamtx_thread.join();
    return 0;
}
