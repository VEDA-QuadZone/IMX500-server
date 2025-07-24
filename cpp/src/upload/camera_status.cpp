#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <ctime>
#include <thread>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 1. 카메라 모델명 획득
std::string get_camera_model() {
    FILE* fp = popen("udevadm info --query=all --name=/dev/video0 | grep ID_MODEL=", "r");
    if (!fp) return "Unknown";
    char buffer[256];
    std::string model;
    while (fgets(buffer, sizeof(buffer), fp)) {
        std::string line(buffer);
        if (line.find("ID_MODEL=") != std::string::npos) {
            model = line.substr(line.find("=") + 1);
            model.erase(model.find_last_not_of(" \n\r\t") + 1);
            break;
        }
    }
    pclose(fp);
    return model.empty() ? "Unknown" : model;
}

// 2. 현재 시간 HH:MM:SS 문자열로
std::string get_time_str() {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char buf[10];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&now));
    return std::string(buf);
}

// 3. CPU 온도 수집
std::string get_temperature() {
    std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
    int temp;
    file >> temp;
    return std::to_string(temp / 1000.0) + "°C";
}

// 4. AI 모델 파일명
std::string get_ai_model() {
    std::string path = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json";
    std::ifstream file(path);
    return file.good() ? "imx500_mobilenet_ssd.json" : "Unknown";
}

int main() {
    cv::VideoCapture cap(0);
    bool camera_available = cap.isOpened();
    double frame_width = 0, frame_height = 0, fps = 0;

    if (camera_available) {
        // 미리 한번 읽어서 속성 값 추출
        frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        fps = cap.get(cv::CAP_PROP_FPS);
    }

    while (true) {
        std::string last_frame_ts = "None";
        std::string error_msg = "Camera not available";
        bool frame_ok = false;

        if (camera_available) {
            cv::Mat frame;
            frame_ok = cap.read(frame);
            if (frame_ok) {
                last_frame_ts = get_time_str();
                error_msg = "None";
            } else {
                error_msg = "Frame read failed";
            }
        }

        json status = {
            {"camera_model", get_camera_model()},
            {"device", "/dev/video0"},
            {"status", camera_available ? "Connected" : "Disconnected"},
            {"resolution", camera_available ? std::to_string((int)frame_width) + "x" + std::to_string((int)frame_height) : "N/A"},
            {"fps", camera_available ? fps : 0},
            {"last_frame", last_frame_ts},
            {"ai_model", get_ai_model()},
            {"ai_status", "Active"},
            {"temperature", get_temperature()},
            {"error", error_msg}
        };

        std::string json_str = status.dump(4); // Pretty print
        std::cout << "[STATUS REPORT]\n" << json_str << "\n\n";

        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    return 0;
}
