#include "filter.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <map>
#include <cmath>
#include <opencv2/photo.hpp>  // for detailEnhance, fastNlMeansDenoisingColored

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

    switch (mode) {
        case EnhanceMode::Day: {
            // CLAHE로 밝은 영역 강조
            cv::Mat lab;
            cv::cvtColor(result, lab, cv::COLOR_BGR2Lab);
            std::vector<cv::Mat> lab_planes;
            cv::split(lab, lab_planes);
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
            clahe->apply(lab_planes[0], lab_planes[0]);
            cv::merge(lab_planes, lab);
            cv::cvtColor(lab, result, cv::COLOR_Lab2BGR);
            break;
        }
        case EnhanceMode::Night: {
            result = apply_gamma(result, 1.5);
            result.convertTo(result, -1, 1.2, 30);
            break;
        }
        case EnhanceMode::Sharp: {
            cv::Mat blurred;
            cv::GaussianBlur(result, blurred, cv::Size(0, 0), 3);
            cv::addWeighted(result, 1.5, blurred, -0.5, 0, result);
            break;
        }
        case EnhanceMode::Original:
        default:
            break;
    }

    return result;
}

