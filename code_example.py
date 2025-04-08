from ultralytics import YOLO
import cv2
import time

# Load a stronger YOLOv8 model for better detection
model = YOLO("yolov8m.pt")  # Can use 'l' or 'x' if your GPU can handle it

# Initialize webcam
cap = cv2.VideoCapture(0)

# For FPS calculation
prev_time = 0

# Define relevant classes for autonomous driving
important_classes = ['person', 'car', 'bus', 'truck', 'motorbike', 'bicycle', 'traffic light', 'stop sign']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Run YOLOv8 detection
    results = model(frame)[0]

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls]

        # Only process relevant classes
        if class_name in important_classes:
            label = f"{class_name}: {conf:.2f}"
            color = (0, 255, 255) if class_name == "person" else (0, 255, 0)

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Simulate warning logic
            if class_name == "person" and (x2 - x1) * (y2 - y1) > 50000:
                cv2.putText(frame, "Alert: Pedestrian Nearby!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            elif class_name == "traffic light":
                cv2.putText(frame, "Alert: Traffic Light Detected", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            elif class_name == "stop sign":
                cv2.putText(frame, "Alert: Stop Sign Detected", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Autonomous Driving Simulation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
