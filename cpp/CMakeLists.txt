cmake_minimum_required(VERSION 3.16)
project(SchoolZoneMonitoring LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -g")

# ───── 경로 설정 ─────
include_directories(include)
include_directories(/home/sejin/onnxruntime-linux-aarch64-1.16.3/include)
include_directories(/home/sejin/tensorflow)

link_directories(
    /home/sejin/onnxruntime-linux-aarch64-1.16.3/lib
    /home/sejin/myproject/cpp/src/detect/assets/model
)

# ───── 종속 라이브러리 ─────
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
pkg_check_modules(MOSQUITTO REQUIRED libmosquitto)

include_directories(
    include
    /home/sejin/onnxruntime-linux-aarch64-1.16.3/include
    /home/sejin/tensorflow
    ${OPENCV_INCLUDE_DIRS}
    ${MOSQUITTO_INCLUDE_DIRS}
)

link_libraries(
    onnxruntime
    tensorflowlite
    ${OPENCV_LIBRARIES}
    ${MOSQUITTO_LIBRARIES}
    pthread curl ssl crypto lgpio zmq
)

# ───── build 폴더 생성 ─────
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/assets/images)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/assets/audio)

# ───── 소스 및 타겟 정의 ─────

add_library(person_detector src/detect/person_detector.cpp)
add_library(parking_detector src/detect/parking_detector.cpp)
add_library(license_detector src/detect/license_detector.cpp)

# upload_person_detection
add_executable(upload_person_detection src/upload/upload_person_detection.cpp)
target_link_libraries(upload_person_detection person_detector)

# upload_illegal_parking
add_executable(upload_illegal_parking src/upload/upload_illegal_parking.cpp)
target_link_libraries(upload_illegal_parking parking_detector license_detector)

# upload_illegal_speed
add_executable(upload_illegal_speed src/upload/upload_illegal_speed.cpp)
target_link_libraries(upload_illegal_speed parking_detector license_detector)

# mqtt_alert_s
add_executable(mqtt_alert_s src/mqtt/mqtt_alert_s.cpp)
target_link_libraries(mqtt_alert_s parking_detector person_detector)

# control_devices
add_executable(control_devices src/control/control_devices.cpp)
target_link_libraries(control_devices person_detector)

# speed_detector_daemon
add_executable(speed_detector_daemon src/detect/speed_detector_daemon.cpp)

# ───── assets 복사 명령 ─────
add_custom_target(copy_assets ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/control/assets/images
        ${CMAKE_BINARY_DIR}/assets/images
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/control/assets/audio
        ${CMAKE_BINARY_DIR}/assets/audio
)

# 모든 실행파일은 이 타겟에 의존
add_dependencies(upload_person_detection copy_assets)
add_dependencies(upload_illegal_parking copy_assets)
add_dependencies(upload_illegal_speed copy_assets)
add_dependencies(mqtt_alert_s copy_assets)
add_dependencies(control_devices copy_assets)
add_dependencies(speed_detector_daemon copy_assets)

