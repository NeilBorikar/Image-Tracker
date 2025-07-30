from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the 'nano' model (fast and light)

# Create DeepSORT tracker
tracker = DeepSort(max_age=30)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLO
    results = model(frame)[0]

    # List to store detections
    detections = []

    # Extract each detection
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box corners
        conf = float(result.conf[0])               # Confidence
        cls = int(result.cls[0])                   # Class ID
        label = model.names[cls]                   # Class name

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    # Update the tracker with current frame detections
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
