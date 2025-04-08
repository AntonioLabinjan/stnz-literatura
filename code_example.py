from ultralytics import YOLO
import cv2
import time

# Load YOLOv8 model (use 'yolov8s.pt', 'm', 'l', or 'x' for stronger models)
model = YOLO("yolov8n.pt")  # 'n' = nano (fastest, lightweight)

# Initialize webcam
cap = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Run YOLOv8 detection
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])      # Bounding box coordinates
        conf = float(box.conf[0])                  # Confidence score
        cls = int(box.cls[0])                      # Class ID
        label = f"{model.names[cls]}: {conf:.2f}"  # Class name + confidence

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("YOLOv8 Live Detection", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
