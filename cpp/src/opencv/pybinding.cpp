#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include "filter.hpp"

namespace py = pybind11;

// numpy → cv::Mat 변환 (3채널 또는 4채널 지원)
cv::Mat numpy_to_mat(py::array_t<uint8_t>& input) {
    py::buffer_info buf = input.request();
    if (buf.ndim != 3)
        throw std::runtime_error("Input must be an HxWxC uint8 image");

    int h = buf.shape[0];
    int w = buf.shape[1];
    int c = buf.shape[2];

    if (c != 3 && c != 4)
        throw std::runtime_error("Only 3 or 4 channel images supported (BGR or BGRA)");

    int type = (c == 3) ? CV_8UC3 : CV_8UC4;
    return cv::Mat(h, w, type, buf.ptr);
}

// cv::Mat → numpy 변환 (채널 수 유지)
py::array_t<uint8_t> mat_to_numpy(const cv::Mat& mat) {
    int h = mat.rows;
    int w = mat.cols;
    int c = mat.channels();

    auto shape = std::vector<ssize_t>{h, w, c};
    auto strides = std::vector<ssize_t>{
        static_cast<ssize_t>(mat.step),
        static_cast<ssize_t>(mat.elemSize()),
        static_cast<ssize_t>(mat.elemSize1())
    };

    return py::array_t<uint8_t>(shape, strides, mat.data);
}

PYBIND11_MODULE(brighten, m) {
    m.doc() = "Image enhancement module using OpenCV and Pybind11";

    m.def("apply_enhancement", [](py::array_t<uint8_t>& input) {
        cv::Mat img = numpy_to_mat(input);
        cv::Mat enhanced = apply_enhancement(img);
        return mat_to_numpy(enhanced);
    }, "Apply enhancement based on /dev/shm/overlay_config (supports 3 or 4 channels)");
}
