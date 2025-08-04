#ifndef FILTER_HPP
#define FILTER_HPP

#include <opencv2/opencv.hpp>
#include <string>

enum class EnhanceMode {
    Original,
    Day,
    Night,
    Sharp
};

// 설정파일(/dev/shm/overlay_config)에서 모드 문자열 읽고 enum으로 반환
EnhanceMode get_mode_from_config();

// 영상 보정 수행 함수 (모드에 따라 내부 알고리즘 적용)
cv::Mat apply_enhancement(const cv::Mat &input);

#endif // FILTER_HPP
