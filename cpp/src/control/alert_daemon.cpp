#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <cstdlib>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#include <lgpio.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>

#define LCD_DEVICE "/dev/lcd_notify"
#define WAV_DEVICE "/dev/wav_notify"
#define ACK_DEVICE "/dev/alert_trigger"

constexpr const char* SPI_DEV = "/dev/spidev0.0";
constexpr uint32_t SPI_SPEED = 40000000;
constexpr int DC_PIN  = 25;
constexpr int RST_PIN = 22;
constexpr int BL_PIN  = 24;

constexpr const char* NORMAL_IMG = "assets/images/normal_90.png";
constexpr const char* ALERT_IMG  = "assets/images/kids_90.png";
constexpr const char* WAV_FILE   = "assets/audio/warning.wav";

volatile bool running = true;
int gpiochip = -1;

std::atomic<bool> lcd_in_progress(false);
std::atomic<bool> wav_in_progress(false);

void handle_sigint(int) { running = false; }

void spi_write(int fd, const uint8_t* buf, size_t len) { write(fd, buf, len); }

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

cv::Mat preprocess_image(const std::string& path) {
    cv::Mat src = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (src.empty()) return {};
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

void send_ack() {
    int fd = open(ACK_DEVICE, O_WRONLY);
    if (fd >= 0) {
        write(fd, "1", 1);
        close(fd);
    }
}

void run_lcd_alert(int spi_fd, const cv::Mat& alert_img, const cv::Mat& normal_img) {
    if (lcd_in_progress.exchange(true)) return;
    for (int i = 0; i < 5; i++) {
        lcd_draw_image(spi_fd, alert_img);
        usleep(300000);
        lcd_draw_image(spi_fd, normal_img);
        usleep(300000);
    }
    lcd_in_progress = false;
}

void run_wav_alert() {
    if (wav_in_progress.exchange(true)) return;
    std::string play_cmd = "aplay -D plughw:3,0 " + std::string(WAV_FILE) + " > /dev/null 2>&1";
    system(play_cmd.c_str());
    wav_in_progress = false;
}

int main() {
    signal(SIGINT, handle_sigint);

    int fd_lcd = open(LCD_DEVICE, O_RDONLY);
    int fd_wav = open(WAV_DEVICE, O_RDONLY);
    if (fd_lcd < 0 || fd_wav < 0) {
        perror("open device");
        return 1;
    }

    // 시작 시 남은 이벤트 초기화
    char flush_buf[8];
    read(fd_lcd, flush_buf, sizeof(flush_buf));
    read(fd_wav, flush_buf, sizeof(flush_buf));

    int spi_fd = open(SPI_DEV, O_WRONLY);
    if (spi_fd < 0) {
        perror("SPI open");
        return 1;
    }

    gpiochip = lgGpiochipOpen(0);
    lgGpioClaimOutput(gpiochip, 0, DC_PIN, 0);
    lgGpioClaimOutput(gpiochip, 0, RST_PIN, 1);
    lgGpioClaimOutput(gpiochip, 0, BL_PIN, 1);
    lgGpioWrite(gpiochip, BL_PIN, 1);

    lcd_init(spi_fd);

    cv::Mat normal_img = preprocess_image(NORMAL_IMG);
    cv::Mat alert_img  = preprocess_image(ALERT_IMG);

    lcd_draw_image(spi_fd, normal_img);

    struct pollfd pfds[2];
    pfds[0].fd = fd_lcd; pfds[0].events = POLLIN;
    pfds[1].fd = fd_wav; pfds[1].events = POLLIN;

    std::cout << "[AlertDaemon] LCD + WAV 모니터링 시작\n";

    while (running) {
        int ret = poll(pfds, 2, -1);
        if (ret < 0) break;

        bool triggered = false;
        bool lcd_event = false;
        bool wav_event = false;

        if (pfds[0].revents & POLLIN) {
            char dummy;
            read(fd_lcd, &dummy, 1);
            lcd_event = true;
            triggered = true;
        }

        if (pfds[1].revents & POLLIN) {
            char dummy[8] = {0};
            read(fd_wav, dummy, sizeof(dummy));
            wav_event = true;
            triggered = true;
        }

        // 이벤트가 발생하면 각각 스레드에서 동시에 실행
        if (lcd_event) {
            std::thread([&]() {
                std::cout << "[LCD] 이벤트 → 이미지 깜빡임\n";
                run_lcd_alert(spi_fd, alert_img, normal_img);
            }).detach();
        }

        if (wav_event) {
            std::thread([]() {
                std::cout << "[WAV] 이벤트 → 사운드 재생\n";
                std::string play_cmd = "aplay -D plughw:3,0 " + std::string(WAV_FILE) + " > /dev/null 2>&1";
                system(play_cmd.c_str());
            }).detach();
        }

        if (triggered) {
            send_ack();
        }
    }

    close(spi_fd);
    close(fd_lcd);
    close(fd_wav);
    lgGpiochipClose(gpiochip);

    return 0;
}
