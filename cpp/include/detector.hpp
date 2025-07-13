// include/detector.hpp

#pragma once

#include <nlohmann/json.hpp>
#include <vector>

// ─── 감지 함수 인터페이스 ────────────────────────

// 불법 주정차 감지
//   meta: find_latest_meta() 로 읽어들인 JSON 객체
//   return: true 이면 불법 주정차 발생
bool detect_illegal_parking(const nlohmann::json& meta);

// 과속 감지
//   meta: find_latest_meta() 로 읽어들인 JSON 객체
//   return: true 이면 과속 차량 발생
bool detect_illegal_speeding(const nlohmann::json& meta);

// 보행자 감지
//   meta: find_latest_meta() 로 읽어들인 JSON 객체
//   return: 감지된 사람 ID 목록
std::vector<int> detect_persons(const nlohmann::json& meta);

// 번호판 OCR 포함 스냅샷 로드 + ONNX 박스 검출 + TFLite OCR
//   cfg 에는 {
//     "shm_dir":    "/dev/shm",
//     "onnx_path":  "src/detect/assets/model/best.onnx",
//     "tflite_path":"src/detect/assets/model/model.tflite",
//     "label_path": "src/detect/assets/model/labels.names"
//   } 와 같이 넘겨줄 수 있습니다.
//   return: { "file":"<snapshot.jpg>", "plate":"<인식된 문자열>" }
nlohmann::json detect_license(const nlohmann::json& cfg = nlohmann::json::object());
