#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>

#define FRAME_WIDTH 1920
#define FRAME_HEIGHT 1080
#define FRAME_CHANNELS 3  // BGR
#define FRAME_SIZE (FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS)

#define BUFFER_SLOTS 8
#define INDEX_SHM "/dev/shm/shm_index"
#define FRAME_SHM_BASE "/dev/shm/shm_frame_"

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

int main() {
    const char* cmd =
        "ffmpeg -f rawvideo -pixel_format bgr24 -video_size 1920x1080 -framerate 30 "
        "-i - -c:v h264_v4l2m2m -b:v 2M -f rtsp rtsp://localhost:8554/test";

    std::cout << "[INFO] Starting FFmpeg subprocess..." << std::endl;
    FILE* pipe = popen(cmd, "w");
    if (!pipe) {
        std::cerr << "[ERROR] Failed to launch FFmpeg" << std::endl;
        return 1;
    }

    auto nextFrameTime = steady_clock::now();

    while (true) {
        nextFrameTime += milliseconds(33);  // 30fps 목표 (1000ms / 30 ≈ 33.3ms)

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

        std::this_thread::sleep_until(nextFrameTime);
    }

    pclose(pipe);
    return 0;
}
