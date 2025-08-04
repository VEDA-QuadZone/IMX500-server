#!/usr/bin/env bash
set -euo pipefail
# ── 경로 설정 ───────────────────────────────────────────────────────────────
CPP_BUILD=~/myproject/cpp/build
STREAM_DIR=~/myproject/cpp/src/stream
MEDIAMTX_BIN=${STREAM_DIR}/mediamtx
MEDIAMTX_CONF=${STREAM_DIR}/mediamtx.yml
RTSP_SERVER_BIN=${STREAM_DIR}/rtsp_server
# 로그 저장 경로
LOG_DIR=~/myproject/runlogs/$(date +%F_%H%M%S)
mkdir -p "$LOG_DIR"
# ── 유틸 함수 ───────────────────────────────────────────────────────────────
pids=()
CLEANED_UP=0
require_file() {
    [[ -e "$1" ]] || { echo "[ERR] Not found: $1"; exit 1; }
    [[ -f "$1" && ! -x "$1" ]] && chmod +x "$1" || true
}
start() {
    local name="$1"; shift
    local cmd=( "$@" )
    echo "[START] $name -> ${cmd[*]}"
    "${cmd[@]}" >>"$LOG_DIR/$name.out" 2>&1 &
    local pid=$!
    echo $pid > "$LOG_DIR/$name.pid"
    pids+=("$pid")
}
sleep_step() {
    local sec="$1"
    echo "[SLEEP] ${sec}s"
    sleep "$sec"
}
cleanup() {
    [[ $CLEANED_UP -eq 1 ]] && return
    CLEANED_UP=1
    trap - INT TERM EXIT
    echo
    echo "[STOP] Stopping all processes..."
    for (( idx=${#pids[@]}-1 ; idx>=0 ; idx-- )); do
        kill "${pids[$idx]}" 2>/dev/null || true
    done
    pkill -P $$ 2>/dev/null || true
    wait 2>/dev/null || true
    echo "[STOP] Done."
}
trap cleanup INT TERM EXIT
# ── 존재 확인 ───────────────────────────────────────────────────────────────
require_file "$CPP_BUILD/mqtt_alert_s"
require_file "$CPP_BUILD/control_dev"
require_file "$CPP_BUILD/alert_daemon"
require_file "$CPP_BUILD/upload_person_detection"
require_file "$CPP_BUILD/upload_illegal_parking"
require_file "$CPP_BUILD/upload_illegal_speed"
require_file "$CPP_BUILD/speed_detector_daemon"
require_file "$MEDIAMTX_BIN"
require_file "$MEDIAMTX_CONF"
require_file "$RTSP_SERVER_BIN"
# ── 실행 순서 ───────────────────────────────────────────────────────────────
start mediamtx              "$MEDIAMTX_BIN" "$MEDIAMTX_CONF"
sleep_step 1
start rtsp_server           "$RTSP_SERVER_BIN"
sleep_step 1
start mqtt_alert_s          "$CPP_BUILD/mqtt_alert_s"
sleep_step 1
start speed_detector_daemon "$CPP_BUILD/speed_detector_daemon"
sleep_step 1
start upload_person         "$CPP_BUILD/upload_person_detection"
sleep_step 1
start upload_illegal_park   "$CPP_BUILD/upload_illegal_parking"
sleep_step 1
start upload_illegal_speed  "$CPP_BUILD/upload_illegal_speed"
sleep_step 1
start alert_daemon       "$CPP_BUILD/alert_daemon"
sleep_step 1
start control_dev       "$CPP_BUILD/control_dev"
sleep_step 1
echo "[INFO] All processes started. Logs saved to: $LOG_DIR"
echo "[INFO] Press Ctrl+C to stop all processes cleanly."
wait
