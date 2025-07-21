import numpy as np
import cv2
from multiprocessing import shared_memory

# [1] 이미지 정보 수동 설정 (기본 1280x720 BGRA = 4채널, dtype=uint8)
WIDTH = 1280
HEIGHT = 720
CHANNELS = 4  # BGRA
SHM_NAME = "shm_frame_1"

# [2] 공유 메모리 열기
shm = shared_memory.SharedMemory(name=SHM_NAME)
size = WIDTH * HEIGHT * CHANNELS

# [3] numpy 배열로 변환 (BGRA)
frame = np.ndarray((HEIGHT, WIDTH, CHANNELS), dtype=np.uint8, buffer=shm.buf[:size])

# [4] BGRA → BGR 변환 (JPEG은 3채널)
bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

# [5] JPEG 인코딩 및 저장
success, jpeg_buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
if success:
    with open("output.jpg", "wb") as f:
        f.write(jpeg_buf.tobytes())
    print("✅ Saved: output.jpg")
else:
    print("❌ JPEG 인코딩 실패")

shm.close()
