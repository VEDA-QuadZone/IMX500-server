import os, numpy as np, cv2, re

in_dir  = '/dev/shm'
out_dir = os.path.expanduser('~/snapshot_png')
os.makedirs(out_dir, exist_ok=True)

# id, timestamp, width, height 네 그룹을 모두 잡도록 수정
pattern = re.compile(r'shm_snapshot_(\d+)_(\d+)_(\d+)x(\d+)')

for fname in os.listdir(in_dir):
    m = pattern.match(fname)
    if not m:
        # 매치 안 된 파일 이름 확인용 디버그
        # print("skip:", fname)
        continue

    id_str, ts_str, w_str, h_str = m.groups()
    w, h = int(w_str), int(h_str)
    expected_size = w * h * 4  # BGRA

    fpath = os.path.join(in_dir, fname)
    data = open(fpath, 'rb').read()

    img = None
    if len(data) == expected_size:
        try:
            img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 4))
        except:
            pass

    # 🎯 fallback: 해상도 다를 때 예시(필요 없으면 지우셔도 됩니다)
    if img is None:
        fb_w, fb_h = 1280, 720
        if len(data) == fb_w * fb_h * 4:
            try:
                img = np.frombuffer(data, dtype=np.uint8).reshape((fb_h, fb_w, 4))
                print(f"[!] {fname} → fallback 적용됨 (1280x720)")
            except:
                pass

    if img is None:
        print(f"[!] 완전 실패: {fname} ({w}x{h}), 실제 {len(data)}B")
        continue

    try:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        out_path = os.path.join(out_dir, fname + '.png')
        cv2.imwrite(out_path, img_bgr)
        # print(f"saved: {out_path}")
    except Exception as e:
        print(f"[!] 이미지 저장 실패: {fname}, 오류: {e}")
