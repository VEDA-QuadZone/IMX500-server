#pragma once

#include <nlohmann/json.hpp>
#include <vector>
// 감지 함수 인터페이스
// 공유메모리로부터 메타데이터를 읽어와 감지 결과를 bool로 반환합니다.

// 불법 주정차 감지: 반환값이 true면 불법 주정차 발생
bool detect_illegal_parking(const nlohmann::json& meta);

// 과속 감지: 반환값이 true면 과속차량 발생
bool detect_illegal_speeding();

// 보행자 감지: 반환값이 true면 보행자 감지됨
std::vector<int> detect_persons(const nlohmann::json&);
