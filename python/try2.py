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
from scipy.optimize import linear_sum_assignment

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

'''
def write_metadata(name: str, metadata: dict, max_size: int = 8192):
    data = json.dumps(metadata).encode('utf-8')
    shm = create_or_attach(name, max_size)
    shm.buf[:len(data)] = data
    shm.buf[len(data)] = 0
    shm.close()
'''
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
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            # 이미 unlink된 경우 무시
            pass

# ----------------------------------------
# SORT Tracker (filterpy.KalmanFilter 사용)
# ----------------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = areaA + areaB - interArea
    return interArea / denom if denom > 0 else 0.0

def iou_batch(bb_test, bb_gt):
    bb_gt  = bb_gt[None]        # (1,M,4)
    bb_test= bb_test[:,None]    # (N,1,4)
    xx1 = np.maximum(bb_test[...,0], bb_gt[...,0])
    yy1 = np.maximum(bb_test[...,1], bb_gt[...,1])
    xx2 = np.minimum(bb_test[...,2], bb_gt[...,2])
    yy2 = np.minimum(bb_test[...,3], bb_gt[...,3])
    w = np.clip(xx2-xx1, 0, None)
    h = np.clip(yy2-yy1, 0, None)
    inter = w*h
    areaA = (bb_test[...,2]-bb_test[...,0])*(bb_test[...,3]-bb_test[...,1])
    areaB = (bb_gt[...,2]-bb_gt[...,0])*(bb_gt[...,3]-bb_gt[...,1])
    return inter/(areaA+areaB-inter+1e-6)

def convert_bbox_to_z(bbox):
    x1,y1,x2,y2 = bbox
    w, h = x2-x1, y2-y1
    x = x1 + w/2.;  y = y1 + h/2.
    s = w*h;        r = w/(h+1e-6)
    return np.array([x,y,s,r], dtype=np.float32).reshape(4,1)

def convert_x_to_bbox(x, score=None):
    w = np.sqrt(x[2]*x[3]);  h = x[2]/(w+1e-6)
    x1 = x[0]-w/2.;  y1 = x[1]-h/2.
    x2 = x[0]+w/2.;  y2 = x[1]+h/2.
    if score is None:
        return np.array([x1,y1,x2,y2], dtype=np.float32).reshape(1,4)
    return np.array([x1,y1,x2,y2,score], dtype=np.float32).reshape(1,5)

class SimpleKalmanFilter:
    def __init__(self):
        self.x = np.zeros((4,1), np.float32)
        self.F = np.eye(4, dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)*10.
        self.Q = np.eye(4, dtype=np.float32)
        self.R = np.eye(4, dtype=np.float32)*10.
        self.H = np.eye(4, dtype=np.float32)

    def initiate(self, z):
        self.x = z.copy()

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

class KalmanBoxTracker:
    _count = 0
    def __init__(self, bbox):
        self.kf = SimpleKalmanFilter()
        self.kf.initiate(convert_bbox_to_z(bbox))
        self.id = KalmanBoxTracker._count;  KalmanBoxTracker._count += 1
        self.time_since_update = 0
        self.hit_streak = 0

    def update(self, bbox):
        self.time_since_update = 0
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return convert_x_to_bbox(self.kf.x)[0]

    def get_state(self):
        return convert_x_to_bbox(self.kf.x)[0]

class Sort:
    def __init__(self, max_age=20, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0,5), np.float32)):
        self.frame_count += 1
        # 1) 예측
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pred = trk.predict()
            trks.append([*pred, 0.])
            if np.any(np.isnan(pred)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
        trks = np.array(trks, np.float32) if trks else np.empty((0,5),np.float32)

        # 2) 매칭
        if dets.shape[0] == 0:
            matches, u_dets, u_trks = np.empty((0,2),int), list(), list(range(trks.shape[0]))
        else:
            iou_mat = iou_batch(dets[:,:4], trks[:,:4])
            m_idx = linear_sum_assignment(-iou_mat)
            matches = np.array(list(zip(*m_idx))) if len(m_idx[0])>0 else np.empty((0,2),int)
            u_dets = [d for d in range(dets.shape[0]) if d not in matches[:,0]]
            u_trks = [t for t in range(trks.shape[0]) if t not in matches[:,1]]

        # 3) 업데이트 & 생성
        for m in matches:
            self.trackers[m[1]].update(dets[m[0],:4])
        for d in u_dets:
            self.trackers.append(KalmanBoxTracker(dets[d,:4]))

        # 4) 결과 수집 & 소멸
        ret = []
        for i in reversed(range(len(self.trackers))):
            trk = self.trackers[i]
            if trk.time_since_update < 1 and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                d = trk.get_state()
                ret.append([*d, trk.id])
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        return np.array(ret, np.float32) if ret else np.empty((0,5),np.float32)


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

    # 1. 필터링된 결과 추출 (box, score, class)
    filtered = [
        (boxes[i], float(scores[i]), int(classes[i]))
        for i in range(len(boxes))
        if scores[i] >= th and get_labels()[int(classes[i])] in {"person", "car", "truck"}
    ]

    # 2. dets_np 구성: [x1, y1, x2, y2, conf]
    dets_np = []
    for b, conf, _ in filtered:
        if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 4:
            try:
                x, y, w, h = map(float, b)
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                dets_np.append([x1, y1, x2, y2, conf])
            except Exception as e:
                print(f"[!] Invalid bbox: {b}, error: {e}")
        else:
            print(f"[!] Skipped malformed box: {b}")

    # 3. numpy 변환 및 빈 배열 대응
    if dets_np:
        print("[DEBUG] dets_np shape check:", [len(e) for e in dets_np])
        dets_np = np.array(dets_np, dtype=np.float32).reshape(-1, 5)
    else:
        print("[DEBUG] dets_np is empty")
        dets_np = np.empty((0, 5), dtype=np.float32)

    # 4. Kalman SORT 업데이트
    tracks = tracker.update(dets_np)

    # 5. 트래킹 결과 → Detection 객체 생성
    dets = []
    for tr in tracks:
        x1, y1, x2, y2, tid = tr.astype(int)
        w, h = x2 - x1, y2 - y1
        box = (x1, y1, w, h)

        # 원본 검출에서 가장 높은 IoU로 conf/클래스 가져오기
        best_idx, best_iou = -1, 0.0
        for i, (b, conf, cat) in enumerate(filtered):
            raw_box = [b[0], b[1], b[0] + b[2], b[1] + b[3]]
            iou_score = iou(raw_box, (x1, y1, x2, y2))
            if iou_score > best_iou:
                best_iou = iou_score
                best_idx = i

        if best_idx >= 0:
            conf = filtered[best_idx][1]
            cat  = filtered[best_idx][2]
        else:
            conf, cat = 0.0, -1

        dets.append(Detection(box, cat, conf, metadata, tid))

    return dets


def read_overlay_config():
    try:
        try:
            # 먼저 attach 시도
            shm = shared_memory.SharedMemory(name="overlay_config")
        except FileNotFoundError:
            # 없으면 새로 생성
            shm = shared_memory.SharedMemory(name="overlay_config", create=True, size=1024)
            default = {"show_bbox": False, "show_timestamp": False}
            encoded = json.dumps(default).encode('utf-8')
            shm.buf[:len(encoded)] = encoded
            shm.buf[len(encoded)] = 0
            print("[INFO] overlay_config created with default false,false")

        config_json = bytes(shm.buf[:]).split(b'\0', 1)[0].decode().strip()
        shm.close()

        if not config_json:
            return False, False

        config = json.loads(config_json)
        return config.get("show_bbox", False), config.get("show_timestamp", False)

    except Exception as e:
        print(f"[!] read_overlay_config error: {e}")
        return False, False

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

        show_bbox, show_timestamp = read_overlay_config()

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
        for trk in tracker.trackers:
            oid = trk.id
            # KalmanBoxTracker.get_state()는 [x1,y1,x2,y2]
            x1, y1, x2, y2 = map(int, trk.get_state())
            box = (x1, y1, x2 - x1, y2 - y1)

            if oid in det_map:
                det  = det_map[oid]
                conf = det.conf
                box  = det.box
            else:
                conf = 0.0

            history = list(position_history.get(oid, []))
            active  = 1 if oid in current_ids else 0
            lbl     = get_labels()[det_map[oid].category] if oid in det_map else "unknown"

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
    
    # —— 콘솔에 FPS 출력 ——  
    global prev_time, fps
    now = time.time()
    dt = now - prev_time
    if dt > 0:
        fps = 1.0 / dt
    prev_time = now

    # '\r'로 같은 줄에 덮어쓰기, flush=True로 즉시 표시
    print(f"\rFPS: {fps:.1f}", end="", flush=True)

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

    tracker = Sort(max_age=50, min_hits=1, iou_threshold=args.iou)

    picam2 = Picamera2(imx500.camera_num)
    cfg = picam2.create_preview_configuration(
        main={"size": (1280, 720)},
        controls={"FrameRate": intrinsics.inference_rate}
    )
    picam2.start(cfg, show_preview=True)
    picam2.pre_callback = draw_and_publish

    # 메타-프레임 링버퍼 유지 위해 빈 루프
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

