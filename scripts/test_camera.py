import cv2
import sys

found = False
for i in range(5):
    cap = cv2.VideoCapture(i)
    ok, frame = False, None
    if cap is not None:
        ok, frame = cap.read()
    if ok and frame is not None:
        out = f"camera_ok_{i}.jpg"
        cv2.imwrite(out, frame)
        print(f"INDEX_OK {i} -> saved {out}")
        found = True
        cap.release()
        break
    else:
        print(f"INDEX_FAIL {i}")
    if cap:
        cap.release()

if not found:
    print('NO_CAMERA_AVAILABLE')
    sys.exit(2)
