import cv2
import numpy as np
from ultralytics import YOLO
model_path = "-"
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame,
        conf=0.4,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Vision Model INT8", annotated_frame)

    # Press q to quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
