# Makefile

CXX           := g++
# ONNX Runtime include
ORT_INC       := -I/home/yuna/onnxruntime-linux-aarch64-1.16.3/include
# TensorFlow Lite include
TFLITE_INC    := -I/home/yuna/tensorflow

CXXFLAGS      := -std=c++17 -Iinclude -Wall -g \
                $(ORT_INC) $(TFLITE_INC) \
                $(shell pkg-config --cflags opencv4 libmosquitto)

# ONNX Runtime lib
ORT_LIBS      := -L/home/yuna/onnxruntime-linux-aarch64-1.16.3/lib -lonnxruntime
# TensorFlow Lite lib 위치와 이름 (libtensorflowlite.so)
TFLITE_LIB_DIR := -L/home/yuna/myproject/cpp/src/detect/assets/model
TFLITE_LIBS    := $(TFLITE_LIB_DIR) -ltensorflowlite

LDFLAGS       := -pthread -lcurl \
                $(ORT_LIBS) $(TFLITE_LIBS) \
				-Wl,-rpath,/home/yuna/onnxruntime-linux-aarch64-1.16.3/lib \
				-Wl,-rpath,/home/yuna/myproject/cpp/src/detect/assets/model \
                $(shell pkg-config --libs opencv4 libmosquitto) \
                -llgpio

# 오브젝트 및 실행파일 정의
PERSON_OBJ           := build/person_detector.o
PARKING_OBJ          := build/parking_detector.o
LICENSE_OBJ          := build/license_detector.o

UPLOAD_SRC           := src/upload/upload_person_detection.cpp
UPLOAD_EXE           := build/upload_person_detection

UPLOAD_ILLEGAL_SRC   := src/upload/upload_illegal_parking.cpp
UPLOAD_ILLEGAL_EXE   := build/upload_illegal_parking

MQTT_ALERT_SRC       := src/mqtt/mqtt_alert.cpp
MQTT_ALERT_EXE       := build/mqtt_alert

CONTROL_SRC          := src/control/control_devices.cpp
CONTROL_OBJS         := $(PERSON_OBJ)
CONTROL_EXE          := build/control_devices

ASSETS_IMG_DIR       := build/assets/images
ASSETS_AUD_DIR       := build/assets/audio

# ─────────────────────────────────────────────────
# all
# ─────────────────────────────────────────────────
all: $(UPLOAD_EXE) \
    $(UPLOAD_ILLEGAL_EXE) \
    $(MQTT_ALERT_EXE) \
    $(CONTROL_EXE) \
    control_assets

# build 디렉터리 보장
build:
	@mkdir -p build

# ─────────────────────────────────────────────────
# 컴파일 규칙
# ─────────────────────────────────────────────────

$(PERSON_OBJ): src/detect/person_detector.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(PARKING_OBJ): src/detect/parking_detector.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(LICENSE_OBJ): src/detect/license_detector.cpp | build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ─────────────────────────────────────────────────
# 링크 규칙
# ─────────────────────────────────────────────────

$(UPLOAD_EXE): $(PERSON_OBJ) | build
	$(CXX) $(CXXFLAGS) \
	  $(UPLOAD_SRC) $(PERSON_OBJ) \
	  -o $@ $(LDFLAGS)

# upload_illegal_parking (parking_detector.o + license_detector.o 링크)
$(UPLOAD_ILLEGAL_EXE): $(PARKING_OBJ) $(LICENSE_OBJ) | build
	$(CXX) $(CXXFLAGS) \
	  $(UPLOAD_ILLEGAL_SRC) $(PARKING_OBJ) $(LICENSE_OBJ) \
	  -o $@ $(LDFLAGS)

$(MQTT_ALERT_EXE): $(PARKING_OBJ) $(PERSON_OBJ) | build
	$(CXX) $(CXXFLAGS) \
	  $(MQTT_ALERT_SRC) $(PARKING_OBJ) $(PERSON_OBJ) \
	  -o $@ $(LDFLAGS)

$(CONTROL_EXE): $(CONTROL_SRC) $(CONTROL_OBJS) | build
	$(CXX) $(CXXFLAGS) \
	  $(CONTROL_SRC) $(CONTROL_OBJS) \
	  -o $@ $(LDFLAGS)

# ─────────────────────────────────────────────────
# assets 복사
# ─────────────────────────────────────────────────

$(ASSETS_IMG_DIR):
	mkdir -p $(ASSETS_IMG_DIR)

$(ASSETS_AUD_DIR):
	mkdir -p $(ASSETS_AUD_DIR)

control_assets: | $(ASSETS_IMG_DIR) $(ASSETS_AUD_DIR)
	cp src/control/assets/images/* $(ASSETS_IMG_DIR)/
	cp src/control/assets/audio/*  $(ASSETS_AUD_DIR)/

# ─────────────────────────────────────────────────
clean:
	rm -rf build

.PHONY: all clean control_assets
