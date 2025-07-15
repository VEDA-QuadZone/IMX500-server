// src/detect/license_detector.cpp

#include "detector.hpp"                  // detect_illegal_parking_ids()
#include <nlohmann/json.hpp>
#include <filesystem>
#include <regex>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <map>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ----------------- 설정 상수 ----------------- //
static const std::string DEFAULT_SHM_DIR        = "/dev/shm";
static const std::string DEFAULT_ONNX_MODEL_PATH = "../src/detect/assets/model/best.onnx";
static const std::string DEFAULT_TFLITE_MODEL    = "../src/detect/assets/model/model.tflite";
static const std::string DEFAULT_LABEL_PATH      = "../src/detect/assets/model/labels.names";

static const int    INPUT_WIDTH   = 320;
static const int    INPUT_HEIGHT  = 320;
static const float  CONF_THRESH   = 0.5f;
static const float  NMS_THRESH    = 0.45f;

// ---- SHM 이름에서 slot, id, timestamp 파싱 ----
static bool parse_shm_snapshot(const std::string& name,
                               int& slot, int& id, std::string& ts)
{
    // 이름 예: shm_snapshot_3_124_20250714_031200
    static const std::regex pat(R"(^shm_snapshot_(\d+)_(\d+)_(\d{8}_\d{6})$)");
    std::smatch m;
    if (!std::regex_match(name, m, pat)) return false;
    slot = std::stoi(m[1].str());
    id   = std::stoi(m[2].str());
    ts   = m[3].str();
    return true;
}

// ---- SHM에서 JPEG 바이트 읽어와 cv::Mat으로 디코딩 ----
static bool load_snapshot_from_shm(const std::string& shm_name, cv::Mat& out_img) {
    int fd = shm_open(shm_name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
        std::cerr << "[ERROR] shm_open failed: " << shm_name << "\n";
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return false;
    }
    std::vector<uchar> buf(st.st_size);
    if (::read(fd, buf.data(), buf.size()) != (ssize_t)buf.size()) {
        close(fd);
        return false;
    }
    close(fd);
    out_img = cv::imdecode(buf, cv::IMREAD_COLOR);
    if (out_img.empty()) {
        std::cerr << "[ERROR] cv::imdecode failed for " << shm_name << "\n";
        return false;
    }
    return true;
}

// ---- ONNX 입력 텐서 생성 ----
static Ort::Value make_ort_tensor(const cv::Mat& img,
                                  std::vector<float>& buffer,
                                  int& top_pad, int& left_pad,
                                  float& scale)
{
    int w = img.cols, h = img.rows;
    scale = std::min((float)INPUT_WIDTH / w, (float)INPUT_HEIGHT / h);
    int new_w = int(w * scale), new_h = int(h * scale);

    cv::Mat resized; cv::resize(img, resized, {new_w, new_h});
    top_pad  = (INPUT_HEIGHT - new_h) / 2;
    left_pad = (INPUT_WIDTH  - new_w) / 2;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded,
        top_pad, INPUT_HEIGHT - new_h - top_pad,
        left_pad, INPUT_WIDTH  - new_w - left_pad,
        cv::BORDER_CONSTANT, {114,114,114});
    cv::cvtColor(padded, padded, cv::COLOR_BGR2RGB);
    padded.convertTo(padded, CV_32FC3, 1/255.0f);

    buffer.resize(3 * INPUT_WIDTH * INPUT_HEIGHT);
    std::memcpy(buffer.data(), padded.data,
                buffer.size() * sizeof(float));

    std::array<int64_t,4> shape = {1,3,INPUT_HEIGHT,INPUT_WIDTH};
    auto mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(
        mem_info, buffer.data(), buffer.size(),
        shape.data(), shape.size());
}

// ---- ONNX 박스 검출 ----
static std::vector<cv::Rect> detect_boxes(Ort::Session& session,
    Ort::Value& input_tensor,
    const char* in_name, const char* out_name,
    const cv::Mat& orig, int top_pad, int left_pad, float scale)
{
    Ort::RunOptions run_opts;

// 모델이 기대하는 입력 이름
const char* input_names[]  = { in_name };
// 실제 넣을 텐서
Ort::Value input_tensors[] = { input_tensor };

// 모델이 반환할 출력 이름
const char* output_names[] = { out_name };

// 이렇게 호출하세요
auto outputs = session.Run(
    run_opts,
    input_names,     // const char*[] 형태
    input_tensors,   // Ort::Value[] 형태
    1,               // 입력 개수
    output_names,    // const char*[] 형태
    1                // 출력 개수
);

    float* data = outputs[0].GetTensorMutableData<float>();
    size_t count = outputs[0]
        .GetTensorTypeAndShapeInfo()
        .GetShape()[1];

    std::vector<cv::Rect> boxes;
    std::vector<float>     scores;
    for (size_t i = 0; i < count; ++i) {
        float cx   = data[i*6 + 0];
        float cy   = data[i*6 + 1];
        float w    = data[i*6 + 2];
        float h    = data[i*6 + 3];
        float conf = data[i*6 + 4];
        if (conf < CONF_THRESH) continue;

        int x1 = int((cx - w/2  - left_pad)/scale);
        int y1 = int((cy - h/2  - top_pad )/scale);
        int x2 = int((cx + w/2  - left_pad)/scale);
        int y2 = int((cy + h/2  - top_pad )/scale);

        x1 = std::clamp(x1, 0, orig.cols-1);
        y1 = std::clamp(y1, 0, orig.rows-1);
        x2 = std::clamp(x2, 0, orig.cols-1);
        y2 = std::clamp(y2, 0, orig.rows-1);

        boxes.emplace_back(x1, y1, x2-x1, y2-y1);
        scores.push_back(conf);
    }

    std::vector<int> idx;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, idx);

    std::vector<cv::Rect> out;
    for (int i : idx) out.push_back(boxes[i]);
    return out;
}

// ---- CTC 디코딩 ----
static std::vector<int> ctc_decode(const float* logits,
                                   int T, int C, int blank=0)
{
    std::vector<int> seq; int prev=-1;
    for (int t=0; t<T; ++t) {
        int best=0; float bv=logits[t*C];
        for (int c=1; c<C; ++c) {
            float v=logits[t*C+c];
            if (v>bv){ bv=v; best=c;}
        }
        if (best!=blank && best!=prev) seq.push_back(best);
        prev=best;
    }
    return seq;
}

// ---- TFLite OCR 실행 ----
static std::string run_tflite_ocr(const cv::Mat& plate,
    const std::string& model_path,
    const std::string& label_path)
{
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interp;
    tflite::InterpreterBuilder(*model, resolver)(&interp);
    interp->AllocateTensors();

    int H = interp->tensor(interp->inputs()[0])->dims->data[1];
    int W = interp->tensor(interp->inputs()[0])->dims->data[2];

    cv::Mat gray; cv::cvtColor(plate, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, {W,H});
    gray.convertTo(gray, CV_32FC1, 1/255.0f);

    std::memcpy(interp->typed_input_tensor<float>(0),
                gray.data, H*W*sizeof(float));
    if (interp->Invoke() != kTfLiteOk) {
        std::cerr<<"[ERROR] TFLite inference failed\n";
        return "";
    }

    float* logits = interp->typed_output_tensor<float>(0);
    int   T = interp->tensor(interp->outputs()[0])->dims->data[1];
    int   C = interp->tensor(interp->outputs()[0])->dims->data[2];
    auto ids = ctc_decode(logits, T, C);

    std::ifstream lf(label_path);
    std::map<int,std::string> labels;
    std::string line; int idx=0;
    while (std::getline(lf,line)) {
        line.erase(std::remove_if(
            line.begin(), line.end(),
            [](char c){return c=='\r'||c=='\n'||c==' ';}),
            line.end());
        labels[idx++] = line;
    }

    std::string result;
    for (int i : ids) if (labels.count(i)) result += labels[i];
    return result;
}

// -------- 공개 인터페이스 -------- //
nlohmann::json detect_license(const nlohmann::json& cfg) {
    std::string shm_dir    = cfg.value("shm_dir",    DEFAULT_SHM_DIR);
    std::string onnx_path  = cfg.value("onnx_path",  DEFAULT_ONNX_MODEL_PATH);
    std::string tflite_mod = cfg.value("tflite_path",DEFAULT_TFLITE_MODEL);
    std::string labels_fp  = cfg.value("label_path", DEFAULT_LABEL_PATH);

    // 1) 불법주정차 ID 목록 조회
    auto ids = detect_illegal_parking_ids();
    if (ids.empty()) {
        return json::array();
    }

    // 2) ONNX 세션 초기화
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "license");
    Ort::SessionOptions opts;
    Ort::Session session(env, onnx_path.c_str(), opts);
    Ort::AllocatorWithDefaultOptions alloc;
    const char* in_name  = session.GetInputNameAllocated(0, alloc).get();
    const char* out_name = session.GetOutputNameAllocated(0, alloc).get();

    json results = json::array();

    // 3) /dev/shm 내 SHM 세그먼트 검색
    for (auto& entry : fs::directory_iterator(shm_dir)) {
        std::string name = entry.path().filename().string();
        int slot, id; std::string ts;
        if (!parse_shm_snapshot(name, slot, id, ts)) continue;
        // 불법주정차 ID 목록에 있는지 확인
        if (std::find(ids.begin(), ids.end(), id) == ids.end()) continue;

        // 4) SHM에서 JPEG 읽어오기
        cv::Mat img;
        if (!load_snapshot_from_shm(name, img)) continue;

        // 5) ONNX 박스 검출
        int top_pad, left_pad; float scale;
        std::vector<float> buf;
        auto tensor = make_ort_tensor(img, buf, top_pad, left_pad, scale);
        auto boxes = detect_boxes(
            session, tensor, in_name, out_name,
            img, top_pad, left_pad, scale);

        // 6) OCR 수행
        std::string plate = "";
        if (!boxes.empty()) {
            cv::Mat plate_roi = img(boxes[0]);
            plate = run_tflite_ocr(plate_roi, tflite_mod, labels_fp);
        }

        // 7) 결과 추가
        results.push_back({
            {"shm_name", name},
            {"plate",    plate}
        });
    }

    return results;
}
