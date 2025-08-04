#include "detector.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <signal.h>
#include <atomic>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

#define LCD_DEVICE "/dev/lcd_notify"
#define WAV_DEVICE "/dev/wav_notify"
#define ACK_DEVICE "/dev/alert_trigger"

constexpr int POLLING_INTERVAL_MS    = 500;
constexpr int DEBOUNCE_CYCLES        = 1;   // 필요시 2~3으로 조정 가능
constexpr int ALERT_COOLDOWN_SECONDS = 30;

volatile bool running = true;
std::atomic<bool> alert_in_progress = false;

void handle_sigint(int) { running = false; }

void trigger_device(const char* device_path) {
    int fd = open(device_path, O_WRONLY);
    if (fd >= 0) {
        write(fd, "1", 1);
        close(fd);
    } else {
        perror("open device");
    }
}

bool wait_for_ack() {
    int fd = open(ACK_DEVICE, O_RDONLY);
    if (fd < 0) {
        perror("open ack");
        return false;
    }

    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLOUT;

    int ret = poll(&pfd, 1, 10000); // 최대 10초 대기
    if (ret > 0 && (pfd.revents & POLLOUT)) {
        char dummy;
        read(fd, &dummy, 1); // flag reset
        close(fd);
        return true;
    }

    close(fd);
    return false;
}

int main() {
    signal(SIGINT, handle_sigint);

    int consecutive_detects = 0;
    bool last_person = false;

    using clock = std::chrono::steady_clock;
    clock::time_point last_alert_time =
        clock::now() - std::chrono::seconds(ALERT_COOLDOWN_SECONDS);

    while (running) {
        auto ids = detect_persons();
        if (!ids.empty()) {
            consecutive_detects++;
        } else {
            consecutive_detects = 0;
        }

        bool now_person = (consecutive_detects >= DEBOUNCE_CYCLES);

        // === 디버깅 로그 추가 ===
        std::cout << "[DEBUG] ids.size(): " << ids.size()
                  << " consecutive_detects: " << consecutive_detects
                  << " now_person: " << now_person
                  << " last_person: " << last_person
                  << " alert_in_progress: " << alert_in_progress.load()
                  << std::endl;

        if (!last_person && now_person && !alert_in_progress.load()) {
            auto now = clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_alert_time).count();

            if (elapsed >= ALERT_COOLDOWN_SECONDS) {
                alert_in_progress = true;
                std::cout << "[ALERT] 사람 감지됨! 이벤트 트리거\n";

                // LCD + 사운드 트리거
                trigger_device(LCD_DEVICE);
                trigger_device(WAV_DEVICE);

                // ack 신호 대기
                if (!wait_for_ack()) {
                    std::cerr << "[WARN] ack timeout\n";
                }

                alert_in_progress = false;
                last_alert_time = now;  // 마지막 알림 시간 갱신
            } else {
                std::cout << "[INFO] 쿨다운 중, 남은 시간: " << (ALERT_COOLDOWN_SECONDS - elapsed) << "s\n";
            }
        }

        last_person = now_person;
        std::this_thread::sleep_for(std::chrono::milliseconds(POLLING_INTERVAL_MS));
    }

    return 0;
}
