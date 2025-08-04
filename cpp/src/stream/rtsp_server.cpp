#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <cstring>
#include <sys/stat.h>

#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720
#define FRAME_CHANNELS 3  // BGR
#define FRAME_SIZE (FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS)

#define BUFFER_SLOTS 8
#define INDEX_SHM "/dev/shm/shm_index"
#define FRAME_SHM_BASE "/dev/shm/shm_frame_"
#define RTSP_FPS_SHM "/dev/shm/shm_rtsp_fps"

using namespace std::chrono;

uint32_t readIndex() {
    int fd = open(INDEX_SHM, O_RDONLY);
    if (fd < 0) {
        perror("open (index)");
        return 0;
    }
    uint32_t index = 0;
    read(fd, &index, sizeof(uint32_t));
    close(fd);
    return index % BUFFER_SLOTS;
}

cv::Mat readFrameFromSlot(int slot) {
    std::string shm_name = FRAME_SHM_BASE + std::to_string(slot);
    int fd = open(shm_name.c_str(), O_RDONLY);
    if (fd < 0) {
        perror(("open " + shm_name).c_str());
        return cv::Mat();
    }

    void* ptr = mmap(nullptr, FRAME_WIDTH * FRAME_HEIGHT * 4, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return cv::Mat();
    }

    cv::Mat bgra(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC4, ptr);
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    munmap(ptr, FRAME_WIDTH * FRAME_HEIGHT * 4);
    close(fd);

    return bgr;
}

void writeRtspFpsToShm(uint32_t fps) {
    int fd = shm_open("/shm_rtsp_fps", O_CREAT | O_RDWR, 0666);
    if (fd >= 0) {
        ftruncate(fd, sizeof(uint32_t));
        write(fd, &fps, sizeof(uint32_t));
        close(fd);
    }
}

int main() {
       const char* cmd =
        "ffmpeg -loglevel error "
        "-f rawvideo -pixel_format bgr24 -video_size 1280x720 -framerate 30 "
        "-i - "
        "-c:v h264_v4l2m2m -b:v 2M "
        "-rtsp_transport tcp "
        "-f rtsp rtsps://localhost:8555/test";

    std::cout << "[INFO] Starting FFmpeg subprocess..." << std::endl;
    FILE* pipe = popen(cmd, "w");
    if (!pipe) {
        std::cerr << "[ERROR] Failed to launch FFmpeg" << std::endl;
        return 1;
    }

    auto nextFrameTime = steady_clock::now();
    int frameCount = 0;
    auto lastFpsReport = steady_clock::now();

    while (true) {
        nextFrameTime += milliseconds(33);  // 30fps 목표

        int slot = readIndex();
        cv::Mat frame = readFrameFromSlot(slot);
        if (frame.empty()) {
            std::cerr << "[WARN] Empty frame, skipping..." << std::endl;
            std::this_thread::sleep_until(nextFrameTime);
            continue;
        }

        size_t written = fwrite(frame.data, 1, frame.total() * frame.elemSize(), pipe);
        if (written != frame.total() * frame.elemSize()) {
            std::cerr << "[ERROR] Failed to write frame to FFmpeg stdin" << std::endl;
            break;
        }

        frameCount++;
        auto now = steady_clock::now();
        auto elapsedSec = duration_cast<seconds>(now - lastFpsReport).count();

        if (elapsedSec >= 1) {
            uint32_t fps = frameCount / elapsedSec;
            writeRtspFpsToShm(fps);
            std::cout << "[INFO] RTSP FPS: " << fps << std::endl;
            frameCount = 0;
            lastFpsReport = now;
        }

        std::this_thread::sleep_until(nextFrameTime);
    }

    pclose(pipe);
    return 0;
}

//g++ -o rtsp_sender rtsp_sender.cpp `pkg-config --cflags --libs opencv4` -std=c++17

