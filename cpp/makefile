# Makefile (upload_person_detection 전용)

CXX      := g++
CXXFLAGS := -std=c++17 -Iinclude -Wall -g \
            $(shell pkg-config --cflags opencv4)
LDFLAGS  := -pthread -lcurl \
            $(shell pkg-config --libs   opencv4)

# 오브젝트 및 실행파일 정의
PERSON_OBJ  := build/person_detector.o
UPLOAD_EXE  := build/upload_person_detection
UPLOAD_SRC  := src/upload/upload_person_detection.cpp

all: $(UPLOAD_EXE)

# person_detector.cpp → .o
build/person_detector.o: src/detect/person_detector.cpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -c $< -o $@

# upload_person_detection 링크
$(UPLOAD_EXE): $(PERSON_OBJ)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) \
	  $(UPLOAD_SRC) $(PERSON_OBJ) \
	  -o $@ $(LDFLAGS)

clean:
	rm -rf build

.PHONY: all clean
