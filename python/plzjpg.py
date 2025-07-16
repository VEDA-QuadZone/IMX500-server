import cv2
import numpy as np

# 1) 이진 파일 읽기
with open('shm_snapshot_0_0_20250716_164514', 'rb') as f:
    data = f.read()

# 2) numpy 배열로 변환
arr = np.frombuffer(data, dtype=np.uint8)

# 3) 디코딩 (JPEG → BGR 이미지)
img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
if img is None:
    raise ValueError("디코딩에 실패했습니다. 입력 파일이 JPEG 바이너리가 맞는지 확인하세요.")

# 4) JPG로 저장
cv2.imwrite('output.jpg', img)
print("output.jpg로 저장 완료")
