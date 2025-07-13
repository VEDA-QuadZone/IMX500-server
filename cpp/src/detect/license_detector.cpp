// src/detect/license_detector.cpp

#include "detector.hpp"

#include <nlohmann/json.hpp>
#include <filesystem>
#include <regex>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>

// ----------------- 설정 상수 ----------------- //
static const std::string DEFAULT_SHM_DIR        = "/dev/shm";
static const std::string DEFAULT_ONNX_MODEL_PATH = "../src/detect/assets/model/best.onnx";
static const std::string DEFAULT_TFLITE_MODEL    = "../src/detect/assets/model/model.tflite";
static const std::string DEFAULT_LABEL_PATH      = "../src/detect/assets/model/labels.names";


static const int    INPUT_WIDTH   = 320;
static const int    INPUT_HEIGHT  = 320;
static const float  CONF_THRESH   = 0.5f;
static const float  NMS_THRESH    = 0.45f;

// -------- 최신 메타파일 찾기 (정확한 패턴 매칭) -------- //
 static std::string find_latest_meta() {
     // 정확히 shm_meta_<slot>_<id>_<YYYYMMDD_HHMMSS>.json 만 매칭
     static const std::regex pat(R"(^shm_meta_\d+_\d+_\d{8}_\d{6}\.json$)");
     std::string latest_path;
     std::filesystem::file_time_type latest_time;
     bool found = false;

     for (auto& entry : std::filesystem::directory_iterator("/dev/shm")) {
         auto fn = entry.path().filename().string();
         if (!std::regex_match(fn, pat)) 
             continue;
         auto t = std::filesystem::last_write_time(entry.path());
         if (!found || t > latest_time) {
             found = true;
             latest_time = t;
             latest_path = entry.path().string();
         }
     }
     return latest_path;  // found==false 이면 빈 문자열
 }

// ---- 메타파일명 → 스냅샷 로드 ---- //
static bool load_from_meta(const std::string& meta_path,
                           cv::Mat& out_img,
                           std::string& out_filename)
{
    // 예: shm_meta_3_124_20250714_031200.json
    std::regex pat(R"(shm_meta_(\d+)_(\d+)_(\d{8}_\d{6})\.json)");
    std::smatch m;
    std::string fname = std::filesystem::path(meta_path).filename().string();
    if (!std::regex_match(fname, m, pat)) {
        std::cerr << "[ERROR] meta filename parse failed: " << fname << "\n";
        return false;
    }
    std::string slot = m[1].str();
    std::string id   = m[2].str();
    std::string ts   = m[3].str();

    out_filename = "shm_snapshot_" + slot + "_" + id + "_" + ts + ".jpg";
    std::filesystem::path img_path = std::filesystem::path(DEFAULT_SHM_DIR) / out_filename;
    if (!std::filesystem::exists(img_path)) {
        std::cerr << "[ERROR] snapshot not found: " << img_path << "\n";
        return false;
    }
    out_img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (out_img.empty()) {
        std::cerr << "[ERROR] failed to load image: " << img_path << "\n";
        return false;
    }
    std::cout << "[INFO] Loaded snapshot: " << out_filename << "\n";
    return true;
}

// ---- ONNX 입력 텐서 생성 ---- //
static Ort::Value make_ort_tensor(const cv::Mat& img,
                                  std::vector<float>& buffer,
                                  int& top_pad,
                                  int& left_pad,
                                  float& scale)
{
    int w = img.cols;
    int h = img.rows;
    scale = std::min((float)INPUT_WIDTH / w, (float)INPUT_HEIGHT / h);
    int new_w = int(w * scale);
    int new_h = int(h * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    top_pad  = (INPUT_HEIGHT - new_h) / 2;
    left_pad = (INPUT_WIDTH - new_w) / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
                       top_pad, INPUT_HEIGHT - new_h - top_pad,
                       left_pad, INPUT_WIDTH - new_w - left_pad,
                       cv::BORDER_CONSTANT, cv::Scalar(114,114,114));

    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    padded.convertTo(padded, CV_32FC3, 1.0f/255.0f);

    buffer.resize(3 * INPUT_WIDTH * INPUT_HEIGHT);
    std::memcpy(buffer.data(), padded.data,
                sizeof(float) * buffer.size());

    std::array<int64_t,4> shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(
        mem_info,
        buffer.data(),
        buffer.size(),
        shape.data(),
        shape.size());
}

// ---- ONNX 박스 검출 ---- //
static std::vector<cv::Rect> detect_boxes(Ort::Session& session,
                                          Ort::Value& input_tensor,
                                          const char* input_name,
                                          const char* output_name,
                                          const cv::Mat& orig,
                                          int top_pad,
                                          int left_pad,
                                          float scale)
{
    auto outputs = session.Run(
        Ort::RunOptions{nullptr},
        &input_name, &input_tensor, 1,
        &output_name, 1);

    float* data = outputs[0].GetTensorMutableData<float>();
    size_t count = outputs[0]
        .GetTensorTypeAndShapeInfo()
        .GetShape()[1];

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    for (size_t i = 0; i < count; ++i) {
        float cx   = data[i*6 + 0];
        float cy   = data[i*6 + 1];
        float w    = data[i*6 + 2];
        float h    = data[i*6 + 3];
        float conf = data[i*6 + 4];
        if (conf < CONF_THRESH) {
            continue;
        }
        int x1 = int((cx - w/2 - left_pad) / scale);
        int y1 = int((cy - h/2 - top_pad) / scale);
        int x2 = int((cx + w/2 - left_pad) / scale);
        int y2 = int((cy + h/2 - top_pad) / scale);

        x1 = std::clamp(x1, 0, orig.cols-1);
        y1 = std::clamp(y1, 0, orig.rows-1);
        x2 = std::clamp(x2, 0, orig.cols-1);
        y2 = std::clamp(y2, 0, orig.rows-1);

        boxes.emplace_back(x1, y1, x2-x1, y2-y1);
        scores.push_back(conf);
    }

    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, idx);

    std::vector<cv::Rect> result;
    for (int i : idx) {
        result.push_back(boxes[i]);
    }
    return result;
}

// -------- CTC 디코딩 -------- //
static std::vector<int> ctc_decode(const float* logits,
                                   int T,
                                   int C,
                                   int blank=0)
{
    std::vector<int> seq;
    int prev = -1;
    for (int t = 0; t < T; ++t) {
        int best_id = 0;
        float best_v = logits[t*C];
        for (int c = 1; c < C; ++c) {
            float v = logits[t*C + c];
            if (v > best_v) {
                best_v = v;
                best_id = c;
            }
        }
        if (best_id != blank && best_id != prev) {
            seq.push_back(best_id);
        }
        prev = best_id;
    }
    return seq;
}

// -------- TFLite OCR 실행 -------- //
static std::string run_tflite_ocr(const cv::Mat& plate,
                                  const std::string& model_path,
                                  const std::string& label_path)
{
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();

    int H = interpreter->tensor(interpreter->inputs()[0])
                ->dims->data[1];
    int W = interpreter->tensor(interpreter->inputs()[0])
                ->dims->data[2];

    cv::Mat gray;
    cv::cvtColor(plate, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(W, H));
    gray.convertTo(gray, CV_32FC1, 1.0f/255.0f);

    std::memcpy(interpreter->typed_input_tensor<float>(0),
                gray.data, sizeof(float)*H*W);

    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "[ERROR] TFLite inference failed\n";
        return "";
    }

    float* out_logits = interpreter->typed_output_tensor<float>(0);
    int   T           = interpreter->tensor(interpreter->outputs()[0])
                          ->dims->data[1];
    int   C           = interpreter->tensor(interpreter->outputs()[0])
                          ->dims->data[2];

    std::vector<int> ids = ctc_decode(out_logits, T, C);

    std::ifstream lf(label_path);
    std::map<int,std::string> labels;
    std::string line;
    int idx = 0;
    while (std::getline(lf, line)) {
        line.erase(std::remove_if(
            line.begin(), line.end(),
            [](char c){ return c=='\n'||c=='\r'||c==' '; }),
            line.end());
        labels[idx++] = line;
    }

    std::string result;
    for (int id : ids) {
        auto it = labels.find(id);
        if (it != labels.end()) {
            result += it->second;
        }
    }
    return result;
}

// -------- 공개 인터페이스 -------- //
nlohmann::json detect_license(const nlohmann::json& cfg) {
    // 설정
    std::string shm_dir    = cfg.value("shm_dir",    DEFAULT_SHM_DIR);
    std::string onnx_path  = cfg.value("onnx_path",  DEFAULT_ONNX_MODEL_PATH);
    std::string tflite_mod = cfg.value("tflite_path",DEFAULT_TFLITE_MODEL);
    std::string labels_fp  = cfg.value("label_path", DEFAULT_LABEL_PATH);

    // 1) 메타 → 스냅샷
    std::string meta_fp = find_latest_meta();
    if (meta_fp.empty()) {
        return nlohmann::json();
    }

    cv::Mat frame;
    std::string filename;
    if (!load_from_meta(meta_fp, frame, filename)) {
        return nlohmann::json();
    }

    // 2) ONNX Runtime 세션
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "license");
    Ort::SessionOptions opts;
    Ort::Session session(env, onnx_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions allocator;

    const char* in_name  = session.GetInputNameAllocated(0, allocator).get();
    const char* out_name = session.GetOutputNameAllocated(0, allocator).get();

    // 3) 입력 텐서 준비 및 박스 검출
    int top_pad, left_pad;
    float scale;
    std::vector<float> tensor_buf;
    Ort::Value input_tensor = make_ort_tensor(
        frame, tensor_buf, top_pad, left_pad, scale);

    auto boxes = detect_boxes(
        session, input_tensor, in_name, out_name,
        frame, top_pad, left_pad, scale);
    if (boxes.empty()) {
        return nlohmann::json();
    }

    // 4) 첫 박스 OCR
    cv::Mat plate = frame(boxes[0]);
    std::string plate_text = run_tflite_ocr(
        plate, tflite_mod, labels_fp);

    // 5) 결과 반환
    return nlohmann::json{
        {"file",  filename},
        {"plate", plate_text}
    };
}
