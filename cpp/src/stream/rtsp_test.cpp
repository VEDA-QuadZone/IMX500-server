#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <gst/app/gstappsrc.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <thread>

// SHM 설정 (Python 쪽과 반드시 일치시킬 것)
static constexpr int WIDTH         = 1280;
static constexpr int HEIGHT        = 720;
static constexpr int CHANNELS_BGRA = 4;
static constexpr int FRAME_SIZE    = WIDTH * HEIGHT * CHANNELS_BGRA;

// 공유메모리 이름
const char* INDEX_SHM      = "/dev/shm/shm_index";
const char* FRAME_SHM_BASE = "/dev/shm/shm_frame_";

// 프레임 카운터 (PTS 계산용)
static guint64 gst_frame_count = 0;

// mmap 해서 포인터 리턴 (읽기 전용)
static void* map_shared_ro(const char* path, size_t size) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        std::cerr << "[ERROR] open failed: " << path << std::endl;
        std::perror("open");
        return nullptr;
    }
    void* ptr = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (ptr == MAP_FAILED) {
        std::cerr << "[ERROR] mmap failed: " << path << std::endl;
        std::perror("mmap");
        return nullptr;
    }
    return ptr;
}

// appsrc need-data 콜백: 버퍼가 필요할 때마다 최신 프레임을 푸시
static void on_need_data(GstElement *appsrc, guint unused_size, gpointer user_data) {
    std::cout << "[DEBUG] on_need_data called" << std::endl;

    // 1) 최신 슬롯 인덱스 읽기
    void* idx_ptr = map_shared_ro(INDEX_SHM, sizeof(uint32_t));
    if (!idx_ptr) {
        std::cerr << "[ERROR] Failed to open shm_index" << std::endl;
        return;
    }
    uint32_t slot = *reinterpret_cast<uint32_t*>(idx_ptr);
    std::cout << "[DEBUG] shm_index(slot): " << slot << std::endl;
    munmap(idx_ptr, sizeof(uint32_t));

    // 2) 해당 슬롯의 프레임 SHM 열기
    char frame_name[64];
    std::snprintf(frame_name, sizeof(frame_name), "%s%u", FRAME_SHM_BASE, slot);
    std::cout << "[DEBUG] frame file: " << frame_name << std::endl;
    void* frame_ptr = map_shared_ro(frame_name, FRAME_SIZE);
    if (!frame_ptr) {
        std::cerr << "[ERROR] Failed to open frame shm: " << frame_name << std::endl;
        return;
    }

    // 3) GstBuffer 생성 & 데이터 복사
    GstBuffer *buffer = gst_buffer_new_allocate(nullptr, FRAME_SIZE, nullptr);
    if (!buffer) {
        std::cerr << "[ERROR] gst_buffer_new_allocate failed!" << std::endl;
        munmap(frame_ptr, FRAME_SIZE);
        return;
    }
    GstMapInfo map_info;
    if (!gst_buffer_map(buffer, &map_info, GST_MAP_WRITE)) {
        std::cerr << "[ERROR] gst_buffer_map failed!" << std::endl;
        gst_buffer_unref(buffer);
        munmap(frame_ptr, FRAME_SIZE);
        return;
    }
    std::memcpy(map_info.data, frame_ptr, FRAME_SIZE);
    gst_buffer_unmap(buffer, &map_info);

    // 4) PTS, Duration 설정 (30 FPS 가정)
    GST_BUFFER_PTS(buffer)      = gst_frame_count * (GST_SECOND / 30);
    GST_BUFFER_DURATION(buffer) = (GST_SECOND / 30);
    std::cout << "[DEBUG] PTS: " << GST_BUFFER_PTS(buffer) << " Duration: " << GST_BUFFER_DURATION(buffer) << std::endl;
    gst_frame_count++;

    // 5) appsrc에 푸시
    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc), buffer);
    if (ret != GST_FLOW_OK) {
        std::cerr << "[ERROR] gst_app_src_push_buffer failed! ret=" << ret << std::endl;
    } else {
        std::cout << "[DEBUG] Buffer pushed to appsrc" << std::endl;
    }

    // 6) 언맵
    munmap(frame_ptr, FRAME_SIZE);
}

// *** 핵심: 전역 함수 콜백! ***
extern "C"
void media_configure_cb(GstRTSPMediaFactory *f, GstRTSPMedia *media, gpointer user_data) {
    std::cout << "[DEBUG] media-configure_cb called" << std::endl;
    GstElement *element = gst_rtsp_media_get_element(media);
    if (!element) {
        std::cerr << "[ERROR] gst_rtsp_media_get_element failed!" << std::endl;
        return;
    }
    GstElement *appsrc = gst_bin_get_by_name_recurse_up(GST_BIN(element), "mysrc");
    if (!appsrc) {
        std::cerr << "[ERROR] gst_bin_get_by_name_recurse_up failed!" << std::endl;
        g_object_unref(element);
        return;
    }
    std::cout << "[DEBUG] Connecting need-data to appsrc..." << std::endl;
    g_signal_connect(appsrc, "need-data", G_CALLBACK(on_need_data), nullptr);
    g_object_unref(appsrc);
    g_object_unref(element);
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // 1) RTSP 서버 생성
    GstRTSPServer *server = gst_rtsp_server_new();
    gst_rtsp_server_set_service(server, "8554");  // 포트 8554 사용

    // 2) Mount points & Factory 생성
    GstRTSPMountPoints *mounts  = gst_rtsp_server_get_mount_points(server);
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();

    // 3) Launch 파이프라인: appsrc → videoconvert → 420 format → x264enc → rtph264pay
    const char *launch_desc =
        "( "
        " appsrc name=mysrc is-live=true block=true format=TIME stream-type=stream "
        " caps=video/x-raw,format=BGRA,width=1280,height=720,framerate=30/1 "
        " ! videoconvert "
        " ! video/x-raw,format=I420 "
        " ! x264enc tune=zerolatency speed-preset=superfast bitrate=3000 "
        " ! rtph264pay name=pay0 pt=96 "
        ")";
    gst_rtsp_media_factory_set_launch(factory, launch_desc);
    gst_rtsp_media_factory_set_shared(factory, TRUE);

    // 4) "/test" 경로에 바인딩
    gst_rtsp_mount_points_add_factory(mounts, "/test", factory);
    g_object_unref(mounts);

    // 5) 서버 attach
    guint attach_result = gst_rtsp_server_attach(server, nullptr);
    std::cout << "[DEBUG] gst_rtsp_server_attach result: " << attach_result << std::endl;

    // 6) media-configure 콜백을 함수포인터로 등록!
    std::cout << "[DEBUG] Connecting media-configure (function)..." << std::endl;
    g_signal_connect(factory, "media-configure", G_CALLBACK(media_configure_cb), nullptr);

    std::cout << "*** RTSP 서버 시작: rtsp://<IP>:8554/test\n";

    // 7) GMainLoop 실행 (blocking)
    GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
    g_main_loop_run(loop);

    return 0;
}
//g++ -Wall -std=c++11 rtsp_server.cpp -o rtsp_server $(pkg-config --cflags --libs gstreamer-1.0 gstreamer-rtsp-server-1.0 gstreamer-app-1.0)