from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (1280, 720)}))
picam2.set_controls({
    "ExposureTime": 15000,      # 밝기 부족 시 늘리기 (us 단위)
    "AnalogueGain": 4.0         # 낮은 밝기면 이걸 8.0까지도 올려보기
})
picam2.start()

time.sleep(2)  # 워밍업
frame = picam2.capture_array()
print("Frame sum:", frame.sum())  # 0이면 완전 검정

cv2.imshow("Preview", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
