#include "filter.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include <cmath>
#include <opencv2/photo.hpp>  // for detailEnhance if needed

using json = nlohmann::json;

EnhanceMode get_mode_from_config() {
    try {
        std::ifstream file("/dev/shm/overlay_config");
        if (!file.is_open()) return EnhanceMode::Original;

        json config;
        file >> config;

        std::string mode_str = config.value("mode", "original");

        static const std::map<std::string, EnhanceMode> mode_map = {
            {"original", EnhanceMode::Original},
            {"day", EnhanceMode::Day},
            {"night", EnhanceMode::Night},
            {"sharp", EnhanceMode::Sharp}
        };

        auto it = mode_map.find(mode_str);
        return (it != mode_map.end()) ? it->second : EnhanceMode::Original;

    } catch (...) {
        return EnhanceMode::Original;
    }
}

// 감마 보정 함수
cv::Mat apply_gamma(const cv::Mat &src, double gamma) {
    CV_Assert(gamma >= 0);
    cv::Mat lut(1, 256, CV_8UC1);
    uchar *p = lut.ptr();
    for (int i = 0; i < 256; i++) {
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::Mat dst;
    cv::LUT(src, lut, dst);
    return dst;
}

cv::Mat apply_enhancement(const cv::Mat &input) {
    EnhanceMode mode = get_mode_from_config();
    cv::Mat result = input.clone();

    try {
        std::ifstream file("/dev/shm/overlay_config");
        json config;
        file >> config;

        switch (mode) {
            case EnhanceMode::Day: {
                int brightness = config.value("day_brightness", 10); // default: 10
                int contrast   = config.value("day_contrast", 15);   // default: 15

                // 자연스럽게 밝기와 대비 조정
                result.convertTo(result, -1, 1.0 + contrast / 100.0, brightness);

                // 부드러운 톤업 효과 추가
                cv::Mat overlay;
                cv::GaussianBlur(result, overlay, cv::Size(5, 5), 0);
                cv::addWeighted(result, 0.8, overlay, 0.2, 0, result);
                break;
            }

            case EnhanceMode::Night: {
                result = apply_gamma(result, 1.5);
                result.convertTo(result, -1, 1.2, 30);
                break;
            }

            case EnhanceMode::Sharp: {
                int sharpness_level = config.value("sharpness_level", 50); // 0~100
                sharpness_level = std::clamp(sharpness_level, 0, 100);

                double sharpness_factor = sharpness_level / 100.0;
                double alpha = 1.0 + sharpness_factor * 1.0; // 1.0 ~ 2.0
                double beta  = sharpness_factor * 0.5;       // 0.0 ~ 0.5

                cv::Mat blurred;
                cv::GaussianBlur(result, blurred, cv::Size(0, 0), 3);
                cv::addWeighted(result, alpha, blurred, -beta, 0, result);
                break;
            }

            case EnhanceMode::Original:
            default:
                break;
        }

    } catch (...) {
        // 오류 발생 시 원본 그대로
    }

    return result;
}
