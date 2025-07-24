#!/usr/bin/env python3
import argparse
import sys
import os
import json
from functools import lru_cache
from multiprocessing import shared_memory, resource_tracker
from collections import defaultdict, deque
import time
import cv2
import numpy as np
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection
import subprocess
import brighten


CAMERA_WIDTH      = 1280
CAMERA_HEIGHT     = 720
CAMERA_RESOLUTION = (CAMERA_WIDTH, CAMERA_HEIGHT)

prev_time = time.time()
fps = 0.0

# snapshot ring-buffer
MAX_SNAPSHOTS=400
#현재까지 몇 번째 스냅샷을 썼는지 카운트
snapshot_counter=0
# ----------------------------------------
# Shared Memory Utilities (크기 재생성 가능)
# ----------------------------------------
def create_or_attach(name: str, size: int):
    """
    name 으로 SHM을 열거나, 없으면 새로 만들고,
    이미 있지만 크기가 size보다 작으면 unlink 후 재생성합니다.
    """
    try:
        shm = shared_memory.SharedMemory(name=name)
        if shm.size < size:
            shm.close()
            shm.unlink()
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileNotFoundError:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    try:
        resource_tracker.unregister(shm._name, 'shared_memory')
    except Exception:
        pass
    return shm

def write_frame(name: str, frame: np.ndarray):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)

    shm = create_or_attach(name, frame.nbytes)
    buf = np.ndarray(frame.shape, dtype=frame.dtype, buffer=shm.buf)
    buf[:] = frame
    shm.close()

def write_metadata(name: str, metadata: dict, max_size: int = 65536):
    data = json.dumps(metadata).encode('utf-8')
    shm = create_or_attach(name, max_size)
    view = shm.buf.cast("B")
    if len(data) >= max_size:
        raise ValueError("Metadata too large for shm segment!")
    view[:len(data)] = data
    view[len(data)] = 0
    del view  # <- 이거 필수!
    shm.close()
def write_snapshot(name: str, snap: np.ndarray):
    shm = create_or_attach(name, snap.nbytes)
    buf = np.ndarray(snap.shape, dtype=snap.dtype, buffer=shm.buf)
    buf[:] = snap
    shm.close()

def write_index(name: str, index: int):
    """4바이트 정수로 슬롯 인덱스를 공유메모리에 기록"""
    shm = create_or_attach(name, 4)
    shm.buf[:4] = index.to_bytes(4, byteorder='little', signed=False)
    shm.close()

# snapshot unlink()로 메모리를 해제하기
def unlink_snapshot(prefix: str):
    #/dev/shm 디렉토리에서 prefix로 시작하는 모든 파일을 찾아서 unlink() 후 close()
    for fname in os.listdir("/dev/shm"):
        if not fname.startswith(prefix):
            continue
        try:
            shm = shared_memory.SharedMemory(name=fname)
            resource_tracker.unregister(shm._name, 'shared_memory')
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            # 이미 unlink된 경우 무시
            pass

# ----------------------------------------
# IoU Tracker
# ----------------------------------------
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return 0.0 if a == 0 or b == 0 else inter / float(a + b - inter)

class IoUTracker:
    def __init__(self, max_disappeared=20, iou_threshold=0.3):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disp = max_disappeared
        self.iou_th = iou_threshold

    def register(self, box):
        self.objects[self.next_id] = box
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, rects):
        if not rects:
            for oid in list(self.disappeared):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disp:
                    self.deregister(oid)
            return {}
        if not self.objects:
            for b in rects:
                self.register(b)
        else:
            oids = list(self.objects)
            obs  = list(self.objects.values())
            D = np.zeros((len(obs), len(rects)), dtype=np.float32)
            for i, ob in enumerate(obs):
                for j, nb in enumerate(rects):
                    D[i,j] = iou(ob, nb)
            rows, cols = set(), set()
            for _ in range(min(len(obs), len(rects))):
                i, j = np.unravel_index(np.argmax(D), D.shape)
                if D[i,j] < self.iou_th:
                    break
                oid = oids[i]
                self.objects[oid] = rects[j]
                self.disappeared[oid] = 0
                D[i,:] = -1; D[:,j] = -1
                rows.add(i); cols.add(j)
            for i in set(range(len(obs))) - rows:
                oid=oids[i]; self.disappeared[oid]+=1
                if self.disappeared[oid] > self.max_disp:
                    self.deregister(oid)
            for j in set(range(len(rects))) - cols:
                self.register(rects[j])
        return self.objects.copy()

class Detection:
    def __init__(self, coords, cat, conf, md, tid=None):
        x, y, w, h = imx500.convert_inference_coords(coords, md, picam2)
        self.box = (int(x), int(y), int(w), int(h))
        self.category = cat
        self.conf = conf
        self.id = tid

# ----------------------------------------
# Globals, Ring Buffer 설정
# ----------------------------------------
BUFFER_SLOTS     = 8
INDEX_SHM        = "shm_index"       # 현재 사용 중인 슬롯 인덱스
FRAME_SHM_BASE   = "shm_frame"
META_SHM_BASE    = "shm_meta"
SNAP_SHM_BASE    = "shm_snapshot"
tracker          = None
best_shots       = {}                 # id -> (conf, crop)
position_history = defaultdict(lambda: deque(maxlen=30))
frame_count      = 0
fps_history = deque(maxlen=10)

@lru_cache
def get_labels():
    labs = intrinsics.labels
    if intrinsics.ignore_dash_labels:
        labs = [l for l in labs if l and l != "-"]
    return labs

def parse_detections(metadata: dict):
    th, iou_th, maxd = args.threshold, args.nms_iou, args.max_detections
    norm, order = intrinsics.bbox_normalization, intrinsics.bbox_order

    out = imx500.get_outputs(metadata, add_batch=True)
    if out is None:
        return []
    if intrinsics.postprocess == "nanodet":
        raw, scores, classes = postprocess_nanodet_detection(
            outputs=out[0], conf=th, iou_thres=iou_th, max_out_dets=maxd
        )[0]
        from picamera2.devices.imx500.postprocess import scale_boxes
        boxes = scale_boxes(raw, 1, 1, *imx500.get_input_size(), False, False)
    else:
        b, s, c = out[0][0], out[1][0], out[2][0]
        if norm:
            b = b / imx500.get_input_size()[1]
        if order == "xy":
            b = b[:, [1, 0, 3, 2]]
        boxes = list(zip(*np.array_split(b, 4, axis=1)))
        scores, classes = s, c

    filtered = [
        (boxes[i], float(scores[i]), int(classes[i]))
        for i in range(len(boxes))
        if scores[i] >= th and get_labels()[int(classes[i])] in {"person", "car", "truck"}
    ]
    rects = [(b[0], b[1], b[2], b[3]) for b, _, _ in filtered]
    tracked = tracker.update(rects)

    dets = []
    for b, conf, cat in filtered:
        tid = None
        for oid, tb in tracked.items():
            if iou(b, tb) > iou_th:
                tid = oid
                break
        dets.append(Detection(b, cat, conf, metadata, tid))
    return dets

def read_overlay_config():
    try:
        # 1. attach or create
        try:
            shm = shared_memory.SharedMemory(name="overlay_config")
            resource_tracker.unregister(shm._name, 'shared_memory')
        except FileNotFoundError:
            shm = shared_memory.SharedMemory(name="overlay_config", create=True, size=1024)
            resource_tracker.unregister(shm._name, 'shared_memory')
            default = {
                "show_bbox": False,
                "show_timestamp": False,
                "mode": "original"
            }
            encoded = json.dumps(default).encode('utf-8')
            shm.buf[:len(encoded)] = encoded
            shm.buf[len(encoded)] = 0
            print("[INFO] overlay_config created with default values")

        # 2. load JSON
        config_json = bytes(shm.buf[:]).split(b'\0', 1)[0].decode().strip()
        shm.close()
        config = json.loads(config_json) if config_json else {}

        # 3. 병합: 누락된 key 있으면 default 병합
        default = {
            "show_bbox": False,
            "show_timestamp": False,
            "mode": "original"
        }
        updated = False
        for key, value in default.items():
            if key not in config:
                config[key] = value
                updated = True

        # 4. 병합된 config 다시 기록
        if updated:
            shm = create_or_attach("overlay_config", 1024)
            encoded = json.dumps(config).encode("utf-8")
            shm.buf[:len(encoded)] = encoded
            shm.buf[len(encoded)] = 0
            shm.close()
            print("[INFO] overlay_config updated with missing defaults")

        # 5. 리턴
        return (
            config["show_bbox"],
            config["show_timestamp"],
            config["mode"]
        )

    except Exception as e:
        print(f"[!] read_overlay_config error: {e}")
        return False, False, "original"



def get_camera_device_path():
    return f"/dev/video{imx500.camera_num}"

def get_camera_model():
    try:
        # 우선 다양한 키 시도 (IMX500에서 'Sensor' 또는 'CameraName'이 더 자주 쓰임)
        for key in ("Model", "Sensor", "CameraName"):
            val = picam2.camera_properties.get(key)
            if val:
                return str(val)
        return get_camera_model_from_libcamera()
    except:
        return get_camera_model_from_libcamera()


def get_camera_status():
    try:
        return "Connected" if picam2.started else "Disconnected"
    except:
        return "Disconnected"

def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = int(f.read())
        return f"{temp / 1000.0:.1f}°C"
    except:
        return "Unknown"

def read_external_rtsp_fps():
    try:
        shm = shared_memory.SharedMemory(name="shm_rtsp_fps")
        fps_bytes = bytes(shm.buf[:4])
        shm.close()
        return int.from_bytes(fps_bytes, byteorder='little')
    except:
        return 0

def write_status_shm():
    try:
        status = {
            "camera_model": get_camera_model(),
            "device": get_camera_device_path(),
            "status": get_camera_status(),
            "resolution": f"{CAMERA_WIDTH}x{CAMERA_HEIGHT}",
            "fps": read_external_rtsp_fps(),
            "ai_model": os.path.basename(args.model),
            "ai_status": "Active",
            "temperature": read_cpu_temp(),
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        }
        encoded = json.dumps(status).encode()
        shm = create_or_attach("shm_status", len(encoded) + 1)
        shm.buf[:len(encoded)] = encoded
        shm.buf[len(encoded)] = 0
        shm.close()
    except Exception as e:
        print(f"[!] Failed to write status shm: {e}")

def draw_bbox_only(img, dets):
    for det in dets:
        x, y, w, h = det.box
        label = get_labels()[det.category]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

def draw_timestamp_only(img):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cv2.putText(img, ts, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img

def draw_and_publish(request, stream="main"):
    global frame_count, best_shots, snapshot_counter

    # 1) detection & tracking
    dets = parse_detections(request.get_metadata())

    # 2) frame counter & ring‐buffer slot for raw frames/metadata
    frame_count += 1
    fid = frame_count
    slot = fid % BUFFER_SLOTS

    # 3) write raw frame into SHM ring buffer
    with MappedArray(request, stream) as m:
        raw = m.array.copy()

        img = raw.copy()
        img = brighten.apply_enhancement(img)
        show_bbox, show_timestamp, _ = read_overlay_config()

        if show_bbox and show_timestamp:
            img = draw_bbox_only(img, dets)
            img = draw_timestamp_only(img)
        elif show_bbox:
            img = draw_bbox_only(img, dets)
        elif show_timestamp:
            img = draw_timestamp_only(img)

        shm_name = f"{FRAME_SHM_BASE}_{slot}"
        write_frame(shm_name, img)

        # 4) update position history
        for det in dets:
            if det.id is not None:
                timestamp_sec = time.time()
                position_history[det.id].append(list(det.box) + [timestamp_sec])

        # 5) build and write metadata
        '''
        objs = defaultdict(list)
        for det in dets:
            lbl = get_labels()[det.category]
            objs[lbl].append({
                "id":      det.id if det.id is not None else -1,
                "conf":    det.conf,
                "box":     list(det.box),
                "history": list(position_history.get(det.id, []))
            })
        now = time.strftime("%FT%T", time.localtime())
        meta = {"frame_id": fid, "timestamp": now, **objs}
        write_metadata(f"{META_SHM_BASE}_{slot}", meta)
        '''

        # 5) build and write metadata WITH active flag
        objs = defaultdict(list)
        # 현재 프레임에 실제로 탐지된 ID 집합
        current_ids = {det.id for det in dets if det.id is not None}
        # dets 리스트를 id→Detection 맵으로
        det_map = {det.id: det for det in dets if det.id is not None}
        # tracker.objects 에 남아 있는 모든 id에 대해
        for oid, last_box in tracker.objects.items():
            # 탐지 정보가 있다면 해당 conf/box, 없다면 마지막 known box + conf=0
            if oid in det_map:
                det = det_map[oid]
                conf = det.conf
                box  = det.box
            else:
                conf = 0.0
                #box  = last_box
                box = tuple(float(v) for v in last_box)
            history = list(position_history.get(oid, []))
            active  = 1 if oid in current_ids else 0

            lbl = get_labels()[det_map[oid].category] if oid in det_map else "unknown"
            objs[lbl].append({
                "id":      oid,
                "conf":    conf,
                "box":     list(box),
                "history": history,
                "active":  active
            })

        now = time.strftime("%FT%T", time.localtime())
        meta = {"frame_id": fid, "timestamp": now, **objs}
        write_metadata(f"{META_SHM_BASE}_{slot}", meta)
        # 6) best‐shot → JPEG 압축 + SHM 200‐slot ring buffer

        MIN_CONF_DIFF = 0.05  # 최소 confidence 차이
        for det in dets:
            if det.id is None:
                continue

            prev_conf, _ = best_shots.get(det.id, (0.0, None))
            if det.conf <= prev_conf + MIN_CONF_DIFF:
                continue

            # crop or full frame
            x, y, w, h = det.box
            snap = raw if get_labels()[det.category] == "person" else raw[y:y+h, x:x+w]

            # RGBA→BGRA if needed
            try:
                if snap.ndim == 3 and snap.shape[2] == 4:
                    snap_bgr = cv2.cvtColor(snap, cv2.COLOR_RGBA2BGRA)
                else:
                    snap_bgr = snap
            except Exception as e:
                print(f"[!] snapshot conversion error: {e}")
                continue
            # (1) 이전 snapshot 중 동일 ID 삭제
            for fname in os.listdir("/dev/shm"):
                if fname.startswith(SNAP_SHM_BASE) and f"_{det.id}_" in fname:
                    try:
                        old = shared_memory.SharedMemory(name=fname)
                        old.unlink()
                        old.close()
                    except FileNotFoundError:
                        pass

            # a) compute slot & timestamp
            slot2     = det.id % MAX_SNAPSHOTS
            base_name = f"{SNAP_SHM_BASE}_{slot2}"
            ts        = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            shm_name  = f"{base_name}_{det.id}_{ts}"

            # b) remove any old segments in this slot
            for fname in os.listdir("/dev/shm"):
                if not fname.startswith(base_name + "_"):
                    continue
                try:
                    old = shared_memory.SharedMemory(name=fname)
                    old.unlink()
                    old.close()
                except FileNotFoundError:
                    pass

            # d) JPEG encode
            success, jpeg_buf = cv2.imencode(
                '.jpg', snap_bgr,
                [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            if not success:
                print(f"[!] JPEG encode failed for ID={det.id}")
            else:
                data = jpeg_buf.tobytes()
                shm = create_or_attach(shm_name, len(data))
                shm.buf[:len(data)] = data
                shm.close()
                best_shots[det.id] = (det.conf, snap_bgr.copy())

                # e) advance counter
                #snapshot_counter += 1

        # 7) overlay detections on preview
        for det in dets:
            x, y, w, h = det.box
            label = f"{get_labels()[det.category]} ID:{det.id} ({det.conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label,
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.5, 1)
            cv2.rectangle(m.array,
                        (x, y - th - 4), (x + tw + 4, y),
                        (255, 255, 255), cv2.FILLED)
            cv2.putText(m.array, label, (x + 2, y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
            cv2.rectangle(m.array, (x, y), (x + w, y + h),
                        (0, 255, 0), 2)

    # 8) write latest frame‐slot index
    write_index(INDEX_SHM, slot)
    
    '''# —— 콘솔에 FPS 출력 ——  
    global prev_time, fps
    now = time.time()
    dt = now - prev_time

    if dt > 0:
        current_fps = 1.0 / dt
        fps_history.append(current_fps)
        fps = sum(fps_history) / len(fps_history)

    prev_time = now

    print(f"\rFPS: {fps:.1f}", end="", flush=True)
    '''

    # 상태 공유 JSON 쓰기
    write_status_shm()

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           required=True)
    p.add_argument("--threshold",       type=float, default=0.55)
    p.add_argument("--nms-iou",         type=float, default=0.5,
                help="NanoDet NMS IoU threshold")
    p.add_argument("--iou",             type=float, default=0.3,
                help="Tracker matching IoU threshold")
    p.add_argument("--max-detections",  type=int,   default=10)
    p.add_argument("--postprocess",     choices=["","nanodet"], default=None)
    p.add_argument("-r","--preserve-aspect-ratio",
                action=argparse.BooleanOptionalAction)
    p.add_argument("--show-bbox", action="store_true",
                   help="Overlay bounding boxes on the frame")
    p.add_argument("--show-timestamp", action="store_true",
                   help="Overlay timestamp on the frame")
    
    return p.parse_args()

def main():
    global args, imx500, intrinsics, picam2, tracker
    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"
    intrinsics.update_with_defaults()

    tracker = IoUTracker(max_disappeared=50, iou_threshold=args.iou)

    picam2 = Picamera2(imx500.camera_num)
    cfg = picam2.create_preview_configuration(
        main={"size": CAMERA_RESOLUTION},
        controls={"FrameRate": intrinsics.inference_rate}
    )
    picam2.start(cfg, show_preview=True)
    picam2.pre_callback = draw_and_publish

    # 메타-프레임 링버퍼 유지 위해 빈 루프
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

