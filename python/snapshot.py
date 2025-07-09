import os, numpy as np, cv2, re

in_dir = '/dev/shm'
out_dir = os.path.expanduser('~/snapshot_png')
os.makedirs(out_dir, exist_ok=True)

pattern = re.compile(r'shm_snapshot_(\d+)_(\d+)x(\d+)')

for fname in os.listdir(in_dir):
    match = pattern.match(fname)
    if not match:
        continue

    id_str, w_str, h_str = match.groups()
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

    # 🎯 fallback: try 640x480
    if img is None:
        fallback_w, fallback_h = 640, 480
        fallback_size = fallback_w * fallback_h * 4
        if len(data) == fallback_size:
            try:
                img = np.frombuffer(data, dtype=np.uint8).reshape((fallback_h, fallback_w, 4))
                print(f"[!] {fname} → fallback 적용됨 (640x480)")
            except:
                pass

    if img is None:
        print(f"[!] 완전 실패: {fname} ({w}x{h}), 실제 {len(data)}B")
        continue

    try:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(os.path.join(out_dir, fname + '.png'), img_bgr)
    except Exception as e:
        print(f"[!] 이미지 저장 실패: {fname}, 오류: {e}")
