import os
import sys
import numpy as np
import cv2

# ensure repo root on sys.path
sys.path.insert(0, os.getcwd())
from core.detector import YoloTracker

# locate model
model_path = os.path.join(os.getcwd(), 'best.pt')
if not os.path.exists(model_path):
    raise SystemExit(f"Model not found at {model_path}")

# find sample image
sample = None
for root, dirs, files in os.walk('archive'):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample = os.path.join(root, f)
            break
    if sample:
        break

if sample is None:
    # fallback: look for any image in repo
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                sample = os.path.join(root, f)
                break
        if sample:
            break

if sample is None:
    # create a synthetic image
    print('No sample image found; creating synthetic image.')
    img = 255 * np.ones((640, 960, 3), dtype='uint8')
    cv2.rectangle(img, (200, 150), (400, 500), (0, 0, 255), 3)
    sample = 'synthetic.jpg'
    cv2.imwrite(sample, img)

print('Using image:', sample)

# read image
frame = cv2.imread(sample)
if frame is None:
    raise SystemExit('Failed to read sample image')

# load model
print('Loading model...')
tracker = YoloTracker(model_path)
print('Model loaded.')

# run inference
print('Running inference...')
dets = tracker.infer(frame, conf=0.25, iou=0.45, tracker='botsort.yaml', persist=True)
print('Detections:')
for d in dets:
    print(d)
    x1, y1, x2, y2 = d['bbox']
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{d['cls']} {d['conf']:.2f}"
    cv2.putText(frame, label, (x1, max(15, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

out_path = 'out_inference.jpg'
cv2.imwrite(out_path, frame)
print('Annotated image saved to', out_path)
