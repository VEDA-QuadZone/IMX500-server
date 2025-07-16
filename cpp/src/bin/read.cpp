#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <thread>
#include <chrono>

#include "detector.hpp"
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

// ----------- 상수 ----------- //
const int INPUT_WIDTH = 320;
const int INPUT_HEIGHT = 320;
const float CONF_THRESH = 0.5f;
const float NMS_THRESH = 0.45f;

// ----------- SHM에서 최신 이미지 로드 ----------- //
// 2) shm에서 최신 JPEG 스냅샷을 찾아서 디코딩
bool load_latest_shm_image(const std::string& shm_dir,
                           cv::Mat& out_image,
                           std::string& out_filename) {
    std::regex pattern(R"(shm_snapshot_(\d+)_(\d+)_(\d{8}_\d{6}))");

    fs::path best_file;
    uint64_t best_ts = 0;

    // 🔎 불법주정차 ID 목록
    std::vector<int> target_ids = detect_illegal_parking_ids();

    for (const auto& entry : fs::directory_iterator(shm_dir)) {
        std::string fn = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_match(fn, m, pattern)) continue;

        int slot = std::stoi(m[1].str());
        int id   = std::stoi(m[2].str());
        std::string ts_str = m[3].str();

        // ⛔ 대상 ID가 아니면 무시
        if (std::find(target_ids.begin(), target_ids.end(), id) == target_ids.end())
            continue;

        // timestamp를 숫자로 변환
        uint64_t ts = std::stoull(ts_str);
        if (ts <= best_ts) continue;

        best_ts   = ts;
        best_file = entry.path();
    }

    if (best_file.empty()) {
        std::cerr << "[ERROR] No valid snapshot found for detected IDs\n";
        return false;
    }

    // JPEG 로드
    out_filename = best_file.filename().string();
    std::ifstream in(best_file, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        std::cerr << "[ERROR] Failed to open file: " << best_file << "\n";
        return false;
    }

    auto sz = in.tellg();
    in.seekg(0);
    std::vector<uchar> buf(sz);
    in.read(reinterpret_cast<char*>(buf.data()), sz);
    in.close();

    out_image = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (out_image.empty()) {
        std::cerr << "[ERROR] JPEG decode failed: " << out_filename << "\n";
        return false;
    }

    std::cout << "[INFO] Loaded snapshot: " << out_filename
              << " (size=" << sz << " bytes)\n";
    return true;
}



// ----------- ONNX 입력 전처리 ----------- //
Ort::Value prepare_input_tensor(const cv::Mat& frame, std::vector<float>& input_tensor_values, int& top_pad, int& left_pad, float& ratio) {
    int img_w = frame.cols, img_h = frame.rows;
    ratio = std::min((float)INPUT_WIDTH / img_w, (float)INPUT_HEIGHT / img_h);
    int new_w = static_cast<int>(img_w * ratio);
    int new_h = static_cast<int>(img_h * ratio);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));
    top_pad = (INPUT_HEIGHT - new_h) / 2;
    left_pad = (INPUT_WIDTH - new_w) / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top_pad, INPUT_HEIGHT - new_h - top_pad,
                       left_pad, INPUT_WIDTH - new_w - left_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    padded.convertTo(padded, CV_32FC3, 1.0 / 255);

    input_tensor_values.clear();
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < INPUT_HEIGHT; ++y)
            for (int x = 0; x < INPUT_WIDTH; ++x)
                input_tensor_values.push_back(padded.at<cv::Vec3f>(y, x)[c]);

    std::vector<int64_t> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(mem_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());
}

// ----------- 번호판 박스 감지 ----------- //
std::vector<cv::Rect> detect_plates(Ort::Session& session, Ort::Value& input_tensor, const char* input_name, const char* output_name, const cv::Mat& original, int top_pad, int left_pad, float ratio) {
    auto output = session.Run(Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, &output_name, 1);
    float* out_data = output[0].GetTensorMutableData<float>();
    size_t num_detections = output[0].GetTensorTypeAndShapeInfo().GetShape()[1];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    for (size_t i = 0; i < num_detections; ++i) {
        float* row = out_data + i * 6;
        float cx = row[0], cy = row[1], w = row[2], h = row[3], conf = row[4];
        if (conf < CONF_THRESH) continue;

        int x1 = static_cast<int>((cx - w / 2 - left_pad) / ratio);
        int y1 = static_cast<int>((cy - h / 2 - top_pad) / ratio);
        int x2 = static_cast<int>((cx + w / 2 - left_pad) / ratio);
        int y2 = static_cast<int>((cy + h / 2 - top_pad) / ratio);

        x1 = std::max(0, x1); y1 = std::max(0, y1);
        x2 = std::min(original.cols - 1, x2);
        y2 = std::min(original.rows - 1, y2);

        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.push_back(conf);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, indices);

    std::vector<cv::Rect> final_boxes;
    for (int idx : indices) {
        final_boxes.push_back(boxes[idx]);
    }

    return final_boxes;
}

// ----------- OCR용 이미지 전처리 ----------- //
cv::Mat preprocess_ocr(const cv::Mat& plate, int height, int width) {
    cv::Mat gray, resized;
    cv::cvtColor(plate, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(width, height));
    resized.convertTo(resized, CV_32FC1, 1.0 / 255);
    return resized;
}

// ----------- CTC 디코딩 ----------- //
std::vector<int> ctc_decode(const float* logits, int time, int classes, int blank = 0) {
    std::vector<int> result;
    int prev = -1;
    for (int t = 0; t < time; ++t) {
        int max_id = 0;
        float max_val = logits[t * classes];
        for (int c = 1; c < classes; ++c) {
            if (logits[t * classes + c] > max_val) {
                max_val = logits[t * classes + c];
                max_id = c;
            }
        }
        if (max_id != blank && max_id != prev)
            result.push_back(max_id);
        prev = max_id;
    }
    return result;
}

// ----------- TFLite OCR 실행 ----------- //
std::string run_tflite_ocr(const cv::Mat& plate_img, const std::string& tflite_model_path, const std::string& label_path) {
    auto model = tflite::FlatBufferModel::BuildFromFile(tflite_model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();

    auto* input = interpreter->input_tensor(0);
    int h = input->dims->data[1], w = input->dims->data[2];
    cv::Mat processed = preprocess_ocr(plate_img, h, w);
    memcpy(input->data.f, processed.data, sizeof(float) * h * w);

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "TFLite inference failed\n";
        return "";
    }

    auto* output = interpreter->output_tensor(0);
    const float* logits = output->data.f;
    int t = output->dims->data[1];
    int cls = output->dims->data[2];

    std::vector<int> decoded = ctc_decode(logits, t, cls);

    std::ifstream file(label_path);
    std::map<int, std::string> labels;
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        line.erase(std::remove_if(line.begin(), line.end(), [](char c) {
            return c == '\n' || c == '\r' || c == ' ';
        }), line.end());
        labels[index++] = line;
    }

    std::string result;
    for (int id : decoded) {
        result += (labels.count(id) ? labels[id] : "[UNK]");
    }

    return result;
}

// ----------- Main ----------- //
int main() {
    std::string shm_dir = "/dev/shm";
    std::string onnx_model_path = "best.onnx";
    std::string tflite_model_path = "model.tflite";
    std::string label_path = "labels.names";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "plate");
    Ort::SessionOptions session_options;
    Ort::Session session(env, onnx_model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;

    Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const char* input_name = input_name_ptr.get();
    Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_ptr.get();

    std::string last_filename = "";

    std::cout << "[START] 실시간 불법 주정차 감지 및 번호판 OCR 루프 시작\n";

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));  // CPU 낭비 방지

        nlohmann::json dummy;
        if (!detect_illegal_parking_ids(dummy)) {
            std::cout << "."; std::cout.flush();
            continue;
        }

        cv::Mat image;
        std::string current_filename;

        // ✅ load_latest_shm_image() 내부에서 파일 이름도 반환하도록 수정
        if (!load_latest_shm_image(shm_dir, image, current_filename)) {
            std::cerr << "[ERROR] 유효한 SHM 이미지 없음\n";
            continue;
        }

        // ✅ 이전과 동일한 파일이면 skip
        if (current_filename == last_filename) {
            std::cout << "[SKIP] 이전과 동일한 이미지: " << current_filename << "\n";
            continue;
        }

        last_filename = current_filename;
        std::cout << "\n[ALERT] 불법 주정차 차량 감지됨! → " << current_filename << "\n";

        int top_pad, left_pad;
        float ratio;
        std::vector<float> input_tensor_values;
        Ort::Value input_tensor = prepare_input_tensor(image, input_tensor_values, top_pad, left_pad, ratio);

        std::vector<cv::Rect> plates = detect_plates(session, input_tensor, input_name, output_name, image, top_pad, left_pad, ratio);
        if (plates.empty()) {
            std::cerr << "[WARN] 번호판 탐지 실패\n";
            continue;
        }

        cv::Mat plate_img = image(plates[0]);
        std::string result = run_tflite_ocr(plate_img, tflite_model_path, label_path);

        if (!result.empty()) {
            std::cout << "[OCR RESULT] " << result << "\n";
            cv::imwrite("ocr_plate_" + result + ".jpg", plate_img);
        } else {
            std::cout << "[INFO] OCR 결과 없음\n";
        }
    }

    return 0;
}

/*
g++ read_plate_and_upload.cpp ../detect/parking_detector.cpp \
    -std=c++17 -O2 \
    -I../../include \
    -I/home/yuna/onnxruntime-linux-aarch64-1.16.3/include \
    -I/home/yuna/tensorflow \
    -L/home/yuna/onnxruntime-linux-aarch64-1.16.3/lib -lonnxruntime \
    -L. -ltensorflowlite \
    `pkg-config --cflags --libs opencv4` \
    -o read_plate_shm

*/

/*
LD_LIBRARY_PATH=/home/yuna/onnxruntime-linux-aarch64-1.16.3/lib:. ./read_plate_shm
*/
