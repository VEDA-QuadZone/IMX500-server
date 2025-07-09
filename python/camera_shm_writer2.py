#!/usr/bin/env python3
import argparse
import sys
import json
from functools import lru_cache
from multiprocessing import shared_memory, resource_tracker
from collections import defaultdict, deque
import time
#박유나ㅎㅇ마로며ㅑ
import cv2
import numpy as np
from picamera2 import Picamera2, MappedArray
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics, postprocess_nanodet_detection

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


def write_metadata(name: str, metadata: dict, max_size: int = 8192):
    data = json.dumps(metadata).encode('utf-8')
    shm = create_or_attach(name, max_size)
    shm.buf[:len(data)] = data
    shm.buf[len(data)] = 0
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

def draw_and_publish(request, stream="main"):
    global frame_count, best_shots

    # 1) detection & tracking
    dets = parse_detections(request.get_metadata())

    frame_count += 1
    fid = frame_count
    slot = fid % BUFFER_SLOTS

    # 2) 원본 프레임 → SHM ring buffer
    with MappedArray(request, stream) as m:
        raw = m.array.copy()
        write_frame(f"{FRAME_SHM_BASE}_{slot}", raw)

        # 3) 히스토리 업데이트
        for det in dets:
            if det.id is not None:
                position_history[det.id].append(list(det.box))

        # 4) 메타 구성(보이는 객체만)
        objs = defaultdict(list)
        for det in dets:
            lbl = get_labels()[det.category]
            objs[lbl].append({
                "id":      det.id if det.id is not None else -1,
                "conf":    det.conf,
                "box":     list(det.box),
                "history": list(position_history.get(det.id, []))
            })
        meta = {"frame_id": fid, **objs}
        write_metadata(f"{META_SHM_BASE}_{slot}", meta)

        # 5) best-shot → SHM per-ID
        for det in dets:
            if det.id is None:
                continue
            prev_conf, _ = best_shots.get(det.id, (0.0, None))
            if det.conf > prev_conf:
                x, y, w, h = det.box
                snap = raw[y:y+h, x:x+w]
                if snap.shape[2] == 4:
                    snap_bgr = cv2.cvtColor(snap, cv2.COLOR_RGBA2BGRA)
                else:
                    snap_bgr = snap
                best_shots[det.id] = (det.conf, snap_bgr.copy())
                write_snapshot(f"{SNAP_SHM_BASE}_{det.id}", snap_bgr)

        # 6) 화면 오버레이
        for det in dets:
            x, y, w, h = det.box
            label = f"{get_labels()[det.category]} ID:{det.id} ({det.conf:.2f})"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(m.array, (x, y-th-4), (x+tw+4, y), (255,255,255), cv2.FILLED)
            cv2.putText(m.array, label, (x+2, y-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
            cv2.rectangle(m.array, (x, y), (x+w, y+h), (0,255,0), 2)

    # 7) 최신 슬롯 인덱스 → SHM
    write_index(INDEX_SHM, slot)

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
    return p.parse_args()

def main():
    global args, imx500, intrinsics, picam2, tracker
    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    intrinsics.task = "object detection"
    intrinsics.update_with_defaults()

    tracker = IoUTracker(max_disappeared=20, iou_threshold=args.iou)

    picam2 = Picamera2(imx500.camera_num)
    cfg = picam2.create_preview_configuration(
        controls={"FrameRate": intrinsics.inference_rate}
    )
    picam2.start(cfg, show_preview=True)
    picam2.pre_callback = draw_and_publish

    # 메타-프레임 링버퍼 유지 위해 빈 루프
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()


