# IMX500-server

본 프로젝트는 **VEDA-QuadZone**이라는 이름의 라즈베리파이 기반 CCTV AI 시스템으로,  
IMX500 AI 카메라를 이용해 실시간 객체 감지, 차량 번호판 인식, 속도 측정 등의 다양한 이벤트를 처리하고,  
RTSP를 통해 실시간 스트리밍과 이벤트 기반 알림을 제공하는 것이 목적입니다.

이 레포지토리인 **IMX500-server**는 전체 시스템의 **서버**로서,  
IMX500 카메라로부터 수신한 추론 결과와 프레임을 기반으로 다음 기능을 수행합니다:

- 차량 및 사람에 대한 감지 이벤트 후처리
- 차량 번호판 인식 (ONNX + TFLite)
- 이벤트 감지 및 분류 (불법 주정차, 과속 등)
- RTSP를 통한 실시간 영상 송출

최종적으로 감지 결과는 TCP 서버와 Qt 기반 모니터링 클라이언트로 전송되어 실시간 확인이 가능합니다.

---
## 📌 프로젝트 전체 구성도

- **카메라 서버 (IMX500)**
  - 객체 감지 및 프레임 추출  
    (_.rpk 추론, libcamera 기반_)

- **[IMX500-server]** ← 본 레포지토리
  - AI 감지 후처리
  - 번호판 인식 (ONNX + TFLite)
  - 이벤트 감지 및 처리
  - 실시간 RTSP 송출

- **[raspi-cctv-tcp-server]**
  - 차량 속도 이벤트 수신
  - TCP 기반 메시지 처리 서버

- **[QuadQT]**
  - Qt 기반 모니터링 클라이언트
  - RTSP 스트리밍 영상 표시
  - 이벤트 수신 및 UI 알림
---
## 🛠️ 기술 스택

| 구분               | 사용 환경 / 툴              | 버전                      | 용도                                 |
|--------------------|-----------------------------|----------------------------|--------------------------------------|
| 클라이언트 개발    | Qt (Windows)                | 6.9.1                      | 모니터링 UI 개발, RTSP 영상 재생     |
| 영상 처리 라이브러리 | OpenCV                      | 4.6.0                      | 이미지 프레임 처리 및 테스트          |
| 영상 인코딩/디코딩 | FFmpeg                      | 5.1.6                      | H.264 인코딩 및 스트리밍             |
| 서버 개발 (TCP)    | Raspberry Pi OS (Linux)     | Debian GNU/Linux 12       | 차량 속도 이벤트 수신 서버           |
| 카메라 서버        | Raspberry Pi OS (Linux)     | 동일                       | IMX500 카메라 연동                    |
| 커널               | Linux Kernel                | 6.1.21 aarch64             | 디바이스 제어, 네트워크               |
| 펌웨어 개발        | STM32CubeIDE                | 1.13.0                     | 속도 감지 센서 펌웨어 개발           |
| 언어               | C/C++                       | g++ 10.2.1 (Raspberry Pi)  | 전체 서버 코드                        |
| DB                 | SQLite                      | 3.40.1                     | 이벤트 히스토리 저장                  |
| 보안               | OpenSSL                     | 3.0.16                     | TLS 기반 통신 암호화                  |
| 객체 감지          | IMX500 내부 .rpk 모델       | Sony Neural Network        | 차량/사람 감지 (엣지 추론)           |
| 번호판 탐지        | ONNX Runtime (C++)          | 1.16.0                     | 차량 번호판 위치 감지                 |
| 번호판 인식        | TFLite (C++)                | 2.14.0                     | 번호판 문자 OCR                       |

---

## 📂 주요 디렉토리 구조

```bash
IMX500-server/
├── camera/                # start_camera.sh 스크립트
├── cpp/
│   ├── build/             # 빌드된 실행 파일
│   ├── include/           # 공통 헤더 파일
│   └── src/
│       ├── detect/        # 번호판/보행자/속도 감지
│       ├── mqtt/          # MQTT 전송
│       ├── upload/        # 스냅샷 전송 및 DB 등록
│       ├── stream/        # RTSP 송출
│       ├── control/       # LCD/스피커 디바이스 제어
├── run_all.sh             # 전체 AI 모듈 실행
├── start_camera.sh        # Picamera2 + IMX500 실행
└── runlogs/               # 로그 저장 폴더
```

---

## 🔧 시스템 실행 순서

### 1. TCP 이벤트 서버 빌드 및 실행 (`raspi-cctv-tcp-server`)
→ 차량 속도 및 이벤트 수신 서버 구동

```bash
cd ~/myproject/raspi-cctv-tcp-server
mkdir build && cd build
cmake ..
make
./raspi-cctv-server
````

---

### 2. IMX500 서버 실행 (`IMX500-server`)
→ 카메라 기반 AI 감지 및 이벤트 송출 처리

```bash
cd ~/myproject/camera
./start_camera.sh    # 카메라 및 공유메모리 연동 시작

cd ~/myproject/cpp/build
./run_all.sh         # 번호판 감지, OCR, 이벤트 감지 등 전체 모듈 실행
```

---

### 3. Qt 클라이언트 실행 (`QuadQT`)

→ 실시간 RTSP 영상 재생 + MQTT 이벤트 수신

```bash
./QuadQT
```

---



## 🧠 내부 처리 구조 (IMX500-server)

### [`start_camera.sh`]
- Picamera2 + IMX500 연동
- 실시간 프레임 → 공유 메모리 기록
- `.rpk` 결과 (ObjectDetectResult) 메타데이터 기록

### [`run_all.sh`]
- 번호판 탐지 (ONNX)
- 번호판 OCR (TFLite)
- 불법주정차, 과속, 보행자 감지
- RTSP 송출 (FFmpeg)
- 이벤트 MQTT 발행
- 스냅샷 및 메타데이터 기록 (`/dev/shm`)

---

## 🔗 참고 레포지토리

- **번호판 OCR 모델**: [KR_LPR_Jax (by noahzhy)](https://github.com/noahzhy/KR_LPR_Jax)  
  └─ 한국 번호판 인식에 특화된 CTC 기반 TFLite OCR 모델
