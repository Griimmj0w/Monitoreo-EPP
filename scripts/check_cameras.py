import cv2
import os

os.makedirs('scripts/cam_checks', exist_ok=True)

for i in range(6):
    print(f'--- Testing camera index {i} ---')
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    opened = cap.isOpened()
    print('isOpened:', opened)
    ok = False
    if opened:
        ret, frame = cap.read()
        print('read ok:', ret)
        if ret and frame is not None:
            out = f'scripts/cam_checks/frame_{i}.jpg'
            cv2.imwrite(out, frame)
            print('Saved frame to', out)
            ok = True
    cap.release()
    if not opened or not ok:
        print(f'Camera {i} not available or no frame read')
    print('')
print('Done')
