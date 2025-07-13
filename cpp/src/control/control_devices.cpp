// control_devices.cpp

#include "detector.hpp"            // detect_persons()
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#include <string.h>
#include <cstdlib>
#include <lgpio.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <chrono>
#include <signal.h>

// === 구성 분리 & 상수화 ===
constexpr const char* SPI_DEV             = "/dev/spidev0.0";
constexpr uint32_t   SPI_SPEED            = 40000000;
constexpr int        DC_PIN               = 25;
constexpr int        RST_PIN              = 22;
constexpr int        BL_PIN               = 24;

constexpr const char* DEFAULT_IMAGE       = "assets/images/kid-in-cross.png";
constexpr const char* ALERT_IMAGE         = "assets/images/slow_sig.png";
constexpr const char* AUDIO_FILE          = "assets/audio/warning.wav";

constexpr int POLLING_INTERVAL_MS         = 500;      // 폴링 주기 (ms)
constexpr int DEBOUNCE_CYCLES             = 3;        // 연속 감지 프레임 수
constexpr int ALERT_FLASH_COUNT           = 6;        // 플래시 횟수
constexpr int FLASH_DELAY_US              = 250000;   // 플래시 딜레이 (us)

// === 전역 변수 ===
volatile bool running = true;
int gpiochip = -1;

// SIGINT 핸들러: 안전한 종료 플래그 설정
void handle_sigint(int) {
    running = false;
}

// === SPI & GPIO 헬퍼 ===

void spi_write(int fd, const uint8_t* buf, size_t len) {
    write(fd, buf, len);
}

void lcd_cmd(int fd, uint8_t cmd) {
    lgGpioWrite(gpiochip, DC_PIN, 0);
    spi_write(fd, &cmd, 1);
}

void lcd_data(int fd, const uint8_t* data, size_t len) {
    lgGpioWrite(gpiochip, DC_PIN, 1);
    spi_write(fd, data, len);
}

void lcd_set_window(int fd, int x0, int y0, int x1, int y1) {
    uint8_t data[4];
    lcd_cmd(fd, 0x2A);
    data[0] = x0 >> 8; data[1] = x0 & 0xFF;
    data[2] = x1 >> 8; data[3] = x1 & 0xFF;
    lcd_data(fd, data, 4);

    lcd_cmd(fd, 0x2B);
    data[0] = y0 >> 8; data[1] = y0 & 0xFF;
    data[2] = y1 >> 8; data[3] = y1 & 0xFF;
    lcd_data(fd, data, 4);

    lcd_cmd(fd, 0x2C);
}

uint16_t rgb888_to_rgb565(uint8_t r, uint8_t g, uint8_t b) {
    return ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
}

// === LCD 초기화 & 드로잉 ===

void lcd_init(int fd) {
    lgGpioWrite(gpiochip, RST_PIN, 0); usleep(100000);
    lgGpioWrite(gpiochip, RST_PIN, 1); usleep(100000);

    lcd_cmd(fd, 0x36); uint8_t madctl = 0x00; lcd_data(fd, &madctl, 1);
    lcd_cmd(fd, 0x3A); uint8_t colmod = 0x05; lcd_data(fd, &colmod, 1);
    lcd_cmd(fd, 0x21);
    lcd_cmd(fd, 0x11); usleep(120000);
    lcd_cmd(fd, 0x29);
}

void lcd_draw_image(int spi_fd, const cv::Mat& img) {
    lcd_set_window(spi_fd, 0, 0, 239, 319);
    lgGpioWrite(gpiochip, DC_PIN, 1);

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3b bgr = img.at<cv::Vec3b>(y, x);
            uint16_t color = rgb888_to_rgb565(bgr[2], bgr[1], bgr[0]);
            uint8_t data[2] = { (uint8_t)(color >> 8), (uint8_t)(color & 0xFF) };
            spi_write(spi_fd, data, 2);
        }
    }
}

void lcd_fill_color(int spi_fd, uint16_t color) {
    lcd_set_window(spi_fd, 0, 0, 239, 319);
    lgGpioWrite(gpiochip, DC_PIN, 1);
    uint8_t pixel[2] = { (uint8_t)(color >> 8), (uint8_t)(color & 0xFF) };

    for (int i = 0; i < 240 * 320; i++) {
        spi_write(spi_fd, pixel, 2);
    }
}

// === 이미지 전처리 ===

cv::Mat preprocess_image(const std::string& path) {
    cv::Mat src = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (src.empty()) {
        fprintf(stderr, "이미지 로드 실패: %s\n", path.c_str());
        return {};
    }

    if (src.channels() == 4)
        cv::cvtColor(src, src, cv::COLOR_BGRA2BGR);

    int target_w = 240, target_h = 320;
    double scale = std::min(target_w / (double)src.cols, target_h / (double)src.rows);
    int new_w = src.cols * scale;
    int new_h = src.rows * scale;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h));

    cv::Mat canvas(target_h, target_w, CV_8UC3, cv::Scalar(0, 0, 0));
    int x_offset = (target_w - new_w) / 2;
    int y_offset = (target_h - new_h) / 2;
    resized.copyTo(canvas(cv::Rect(x_offset, y_offset, new_w, new_h)));

    return canvas;
}

// === 경고음 재생 ===

void play_sound_async(const std::string& file_path) {
    std::thread([file_path]() {
        std::string cmd = "aplay -D plughw:0,0 " + file_path + " > /dev/null 2>&1";
        system(cmd.c_str());
    }).detach();
}

// === 보행자 경고 트리거 ===

void trigger_pedestrian_alert(int spi_fd, const cv::Mat& alert, const cv::Mat& default_img) {
    play_sound_async(AUDIO_FILE);

    for (int i = 0; i < ALERT_FLASH_COUNT; i++) {
        lcd_draw_image(spi_fd, alert);
        usleep(FLASH_DELAY_US);
        lcd_fill_color(spi_fd, 0x0000);
        usleep(FLASH_DELAY_US);
    }

    lcd_draw_image(spi_fd, default_img);
}

// === 메인 ===

int main() {
    // SIGINT 핸들러 등록
    signal(SIGINT, handle_sigint);

    // 1) 이미지 로드
    cv::Mat image = preprocess_image(DEFAULT_IMAGE);
    if (image.empty()) return 1;
    cv::Mat alert = preprocess_image(ALERT_IMAGE);
    if (alert.empty()) return 1;

    // 2) SPI & GPIO 초기화
    int spi_fd = open(SPI_DEV, O_WRONLY);
    if (spi_fd < 0) {
        perror("SPI open");
        return 1;
    }
    uint8_t mode = 0, bits = 8;
    uint32_t speed = SPI_SPEED;
    if (ioctl(spi_fd, SPI_IOC_WR_MODE, &mode) < 0 ||
        ioctl(spi_fd, SPI_IOC_WR_BITS_PER_WORD, &bits) < 0 ||
        ioctl(spi_fd, SPI_IOC_WR_MAX_SPEED_HZ, &speed) < 0) {
        perror("SPI ioctl");
        close(spi_fd);
        return 1;
    }

    gpiochip = lgGpiochipOpen(0);
    if (gpiochip < 0) {
        fprintf(stderr, "GPIO chip open failed\n");
        close(spi_fd);
        return 1;
    }
    lgGpioClaimOutput(gpiochip, 0, DC_PIN, 0);
    lgGpioClaimOutput(gpiochip, 0, RST_PIN, 1);
    lgGpioClaimOutput(gpiochip, 0, BL_PIN, 1);
    lgGpioWrite(gpiochip, BL_PIN, 1);

    lcd_init(spi_fd);
    lcd_draw_image(spi_fd, image);

    // 3) 디바운싱 & 상태 추적
    int consecutive_detects = 0;
    bool last_person = false;

    // 4) 폴링 루프
    while (running) {
        auto ids = detect_persons();
        if (!ids.empty()) {
            consecutive_detects++;
        } else {
            consecutive_detects = 0;
        }

        bool now_person = (consecutive_detects >= DEBOUNCE_CYCLES);

        // OFF → ON 전환 시 알림 트리거
        if (!last_person && now_person) {
            trigger_pedestrian_alert(spi_fd, alert, image);
        }
        last_person = now_person;

        std::this_thread::sleep_for(std::chrono::milliseconds(POLLING_INTERVAL_MS));
    }

    // 5) 자원 정리
    close(spi_fd);
    lgGpiochipClose(gpiochip);
    return 0;
}
