#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <ctime>
#include <thread>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <arpa/inet.h>
#include <unistd.h>

using json = nlohmann::json;

// 1. 카메라 모델명 획득 (udevadm 이용)
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

// 4. AI 모델 파일명 (예: mobilenet_ssd.json)
std::string get_ai_model() {
    std::string path = "/usr/share/rpi-camera-assets/imx500_mobilenet_ssd.json";
    std::ifstream file(path);
    return file.good() ? "imx500_mobilenet_ssd.json" : "Unknown";
}

int main() {
    const char* server_ip = "192.168.0.32"; // 수정 필요
    const int server_port = 5000;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in serv_addr{};
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(server_port);
    inet_pton(AF_INET, server_ip, &serv_addr.sin_addr);

    if (connect(sock, (sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("connect");
        return 1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Camera open failed\n";
        return 1;
    }

    while (true) {
        cv::Mat frame;
        bool ok = cap.read(frame);

        json status = {
            {"camera_model", get_camera_model()},
            {"device", "/dev/video0"},
            {"status", cap.isOpened() ? "Connected" : "Disconnected"},
            {"resolution", std::to_string((int)cap.get(cv::CAP_PROP_FRAME_WIDTH)) + "x" +
                           std::to_string((int)cap.get(cv::CAP_PROP_FRAME_HEIGHT))},
            {"fps", cap.get(cv::CAP_PROP_FPS)},
            {"last_frame", ok ? get_time_str() : "None"},
            {"ai_model", get_ai_model()},
            {"ai_status", "Active"},
            {"temperature", get_temperature()},
            {"error", ok ? "None" : "Frame read failed"}
        };

        std::string json_str = status.dump() + "\n";

        // 콘솔 출력
        std::cout << "[SEND] " << json_str;

        send(sock, json_str.c_str(), json_str.size(), 0);
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    close(sock);
    return 0;
}
