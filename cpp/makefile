# Makefile

CXX      := g++
CXXFLAGS := -std=c++17 -Iinclude -Wall -g \
            $(shell pkg-config --cflags opencv4 libmosquitto)
LDFLAGS  := -pthread -lcurl \
            $(shell pkg-config --libs   opencv4 libmosquitto)

# 오브젝트 및 실행파일 정의
PERSON_OBJ        := build/person_detector.o
PARKING_OBJ       := build/parking_detector.o
UPLOAD_EXE        := build/upload_person_detection
UPLOAD_SRC        := src/upload/upload_person_detection.cpp
MQTT_ALERT_EXE    := build/mqtt_alert
MQTT_ALERT_SRC    := src/mqtt/mqtt_alert.cpp

all: $(UPLOAD_EXE) $(MQTT_ALERT_EXE)

# build 폴더 보장
build:
	@mkdir -p build

# person_detector.o
$(PERSON_OBJ): src/detect/person_detector.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# parking_detector.o
$(PARKING_OBJ): src/detect/parking_detector.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# upload_person_detection 링크
$(UPLOAD_EXE): $(PERSON_OBJ) | build
	$(CXX) $(CXXFLAGS) \
	$(UPLOAD_SRC) $(PERSON_OBJ) \
	-o $@ $(LDFLAGS)

# mqtt_alert 링크
# detect_illegal_parking() 은 parking_detector.cpp 에서, 
# publish_event() 는 mqtt_utils.hpp 안에 inline 구현이 있다고 가정
$(MQTT_ALERT_EXE): $(PARKING_OBJ) | build
	$(CXX) $(CXXFLAGS) \
	$(MQTT_ALERT_SRC) $(PARKING_OBJ) \
	-o $@ $(LDFLAGS)

clean:
	rm -rf build

.PHONY: all clean
