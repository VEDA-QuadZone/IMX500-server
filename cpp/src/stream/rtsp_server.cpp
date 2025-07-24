#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <iostream>
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>

void start_gst_rtsp_server(int fd) {
    gst_init(nullptr, nullptr);

    GstRTSPServer* server = gst_rtsp_server_new();
    GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server);
    GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();

    // 핵심 옵션: do-timestamp, blocking
    gchar* launch_str = g_strdup_printf(
        "( fdsrc fd=%d do-timestamp=true blocking=true ! h264parse ! rtph264pay name=pay0 pt=96 config-interval=1 )",
        fd
    );

    gst_rtsp_media_factory_set_launch(factory, launch_str);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_mount_points_add_factory(mounts, "/video", factory);
    g_object_unref(mounts);

    gst_rtsp_server_attach(server, NULL);
    std::cout << "RTSP 서버가 rtsp://<IP>:8554/video 에서 실행 중입니다.\n";

    GMainLoop* loop = g_main_loop_new(NULL, FALSE);
    g_main_loop_run(loop);
}

int main() {
    int pipefd[2];
    if (pipe(pipefd) == -1) {
        perror("pipe 생성 실패");
        return 1;
    }

    pid_t cam_pid = fork();
    if (cam_pid == 0) {
        // 자식 프로세스: libcamera-vid 실행
        dup2(pipefd[1], STDOUT_FILENO); // stdout → 파이프 쓰기
        close(pipefd[0]); // 읽기 닫음 (자식)
        // ✅ 파이프 쓰기 끝은 닫지 말고 남겨둬야 스트림 유지됨

        const char* camera_cmd[] = {
            "libcamera-vid",
            "-t", "0",
            "--codec", "h264",          // 하드웨어 인코더
            "--inline", "--flush",      // SPS/PPS 포함, 버퍼링 제거
            "--width", "1280",
            "--height", "720",
            "--framerate", "30",
            "--output", "-",            // stdout으로 출력
            NULL
        };

        execvp(camera_cmd[0], (char* const*)camera_cmd);
        perror("libcamera-vid 실행 실패");
        exit(1);
    }

    // 부모 프로세스
    close(pipefd[1]); // 쓰기 닫기 (부모는 읽기만)
    int camera_output_fd = pipefd[0];

    start_gst_rtsp_server(camera_output_fd);
    return 0;
}
