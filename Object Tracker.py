from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np


CLASS_COLORS = {
    "person": (0, 255, 0),         # Green
    "car": (255, 0, 0),            # Blue
    "cell phone": (255, 255, 0),   # Cyan
    "handbag": (255, 0, 255),      # Pink
    "backpack": (0, 255, 255),     # Yellow
    "dog": (0, 128, 255),          # Orange
    
}

model = YOLO("yolov8s.pt") 

tracker = DeepSort(max_age=30)

cap = cv2.VideoCapture(0)
unidentified_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)[0]

    
    detections = []
    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  
        conf = float(result.conf[0])               
        cls = int(result.cls[0])                   
        label = model.names[cls]                   

        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

    
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        original_label = track.get_det_class()
        if original_label in CLASS_COLORS:
            label = original_label
            color = CLASS_COLORS[original_label]
        else:
            label = "unidentified"
            color = (255, 255, 255)
            unidentified_count += 1
        
        display_text = f"{label} {track_id}"  
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        
        
        cv2.putText(frame, display_text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Total unidentified objects during session: {unidentified_count}")
