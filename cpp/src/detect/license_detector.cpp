// read_plate_and_upload.cpp (라이브러리화된 전체 코드 with 설정 상수, snapshot 파싱, main 제거)
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
#include <unordered_set>

#include "detector.hpp"
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ----------------- 설정 상수 ----------------- //
static const std::string DEFAULT_SHM_DIR         = "/dev/shm";
static const std::string DEFAULT_ONNX_MODEL_PATH = "../src/detect/assets/model/best.onnx";
static const std::string DEFAULT_TFLITE_MODEL    = "../src/detect/assets/model/model.tflite";
static const std::string DEFAULT_LABEL_PATH      = "../src/detect/assets/model/labels.names";

static const int    INPUT_WIDTH   = 320;
static const int    INPUT_HEIGHT  = 320;
static const float  CONF_THRESH   = 0.5f;
static const float  NMS_THRESH    = 0.45f;

// ----------------- snapshot 이름 파싱 ----------------- //
bool parse_shm_snapshot(const std::string& name,
                        int& slot, int& id, std::string& ts) {
    static const std::regex pat(R"(^shm_snapshot_(\d+)_(\d+)_(\d{8}_\d{6})$)");
    std::smatch m;
    if (!std::regex_match(name, m, pat)) return false;
    slot = std::stoi(m[1].str());
    id   = std::stoi(m[2].str());
    ts   = m[3].str();
    return true;
}

// 최신 shm 스냅샷 파일 디코딩
bool load_snapshot_from_shm(const std::string& filename, cv::Mat& out_img) {
    fs::path full_path = fs::path(DEFAULT_SHM_DIR) / filename;
    std::ifstream in(full_path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) return false;
    auto sz = in.tellg();
    in.seekg(0);
    std::vector<unsigned char> buf(sz);
    in.read(reinterpret_cast<char*>(buf.data()), sz);
    in.close();

    out_img = cv::imdecode(buf, cv::IMREAD_COLOR);
    return !out_img.empty();
}

Ort::Value make_ort_tensor(const cv::Mat& frame, std::vector<float>& input_tensor_values, int& top_pad, int& left_pad, float& ratio) {
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

std::vector<cv::Rect> detect_boxes(Ort::Session& session, Ort::Value& input_tensor, const char* input_name, const char* output_name, const cv::Mat& original, int top_pad, int left_pad, float ratio) {
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

cv::Mat preprocess_ocr(const cv::Mat& plate, int height, int width) {
    cv::Mat gray, resized;
    cv::cvtColor(plate, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(width, height));
    resized.convertTo(resized, CV_32FC1, 1.0 / 255);
    return resized;
}

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
/*
json detect_license(const json& cfg) {
    std::string shm_dir    = cfg.value("shm_dir",    DEFAULT_SHM_DIR);
    std::string onnx_path  = cfg.value("onnx_path",  DEFAULT_ONNX_MODEL_PATH);
    std::string tflite_mod = cfg.value("tflite_path",DEFAULT_TFLITE_MODEL);
    std::string labels_fp  = cfg.value("label_path", DEFAULT_LABEL_PATH);

    auto ids = detect_illegal_parking_ids();
    if (ids.empty()) return json::array();

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "license");
    Ort::SessionOptions opts;
    Ort::Session session(env, onnx_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_ptr  = session.GetInputNameAllocated(0, alloc);
    auto out_name_ptr = session.GetOutputNameAllocated(0, alloc);
    std::string in_name  = in_name_ptr.get();
    std::string out_name = out_name_ptr.get();

    json results = json::array();

    for (auto& entry : fs::directory_iterator(shm_dir)) {
        std::string name = entry.path().filename().string();
        int slot, id; std::string ts;
        if (!parse_shm_snapshot(name, slot, id, ts)) continue;
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) continue;

        cv::Mat img;
        if (!load_snapshot_from_shm(name, img)) continue;

        int top_pad, left_pad; float scale;
        std::vector<float> buf;
        auto tensor = make_ort_tensor(img, buf, top_pad, left_pad, scale);
        auto boxes = detect_boxes(session, tensor, in_name.c_str(), out_name.c_str(), img, top_pad, left_pad, scale);

        std::string plate = "";
        if (!boxes.empty()) {
            cv::Mat plate_roi = img(boxes[0]);
            plate = run_tflite_ocr(plate_roi, tflite_mod, labels_fp);
        }

        results.push_back({
            {"shm_name", name},
            {"plate",    plate}
        });
    }

    return results;
}
*/
nlohmann::json detect_license_by_id(int id, const nlohmann::json& cfg)  {
    std::string shm_dir    = cfg.value("shm_dir",    DEFAULT_SHM_DIR);
    std::string onnx_path  = cfg.value("onnx_path",  DEFAULT_ONNX_MODEL_PATH);
    std::string tflite_mod = cfg.value("tflite_path",DEFAULT_TFLITE_MODEL);
    std::string labels_fp  = cfg.value("label_path", DEFAULT_LABEL_PATH);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "license");
    Ort::SessionOptions opts;
    Ort::Session session(env, onnx_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    auto in_name_ptr  = session.GetInputNameAllocated(0, alloc);
    auto out_name_ptr = session.GetOutputNameAllocated(0, alloc);
    std::string in_name  = in_name_ptr.get();
    std::string out_name = out_name_ptr.get();

    for (auto& entry : std::filesystem::directory_iterator(shm_dir)) {
        std::string name = entry.path().filename().string();
        int slot, cur_id; std::string ts;
        if (!parse_shm_snapshot(name, slot, cur_id, ts)) continue;
        if (cur_id != id) continue; // 여기만 id로 필터

        cv::Mat img;
        if (!load_snapshot_from_shm(name, img)) continue;

        int top_pad, left_pad; float scale;
        std::vector<float> buf;
        auto tensor = make_ort_tensor(img, buf, top_pad, left_pad, scale);
        auto boxes = detect_boxes(session, tensor, in_name.c_str(), out_name.c_str(), img, top_pad, left_pad, scale);

        std::string plate = "";
        if (!boxes.empty()) {
            cv::Mat plate_roi = img(boxes[0]);
            plate = run_tflite_ocr(plate_roi, tflite_mod, labels_fp);
        }
        // 반환: 스냅샷 이름/번호판만
        return {
            {"shm_name", name},
            {"plate", plate}
        };
    }
    // 못 찾은 경우
    return {};
}
