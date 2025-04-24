import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time
import numpy as np

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

important_classes = ['person', 'car', 'bus', 'truck', 'motorbike', 'bicycle', 'traffic light', 'stop sign']

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Driving Dashboard")
        self.root.geometry("1000x700")
        self.root.configure(bg="#222")

        self.canvas = tk.Label(root)
        self.canvas.pack()

        self.status_label = tk.Label(root, text="Decision: GO", font=("Helvetica", 24), fg="lime", bg="#222")
        self.status_label.pack(pady=10)

        self.fps_label = tk.Label(root, text="FPS: 0", font=("Helvetica", 16), fg="white", bg="#222")
        self.fps_label.pack()

        self.prev_time = time.time()
        self.update()

    def detect_lanes(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        height, width = frame.shape[:2]
        mask = np.zeros_like(edges)
        roi = np.array([[(0, height), (width, height), (width, height//2), (0, height//2)]])
        cv2.fillPoly(mask, roi, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50, maxLineGap=50)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    def update(self):
        ret, frame = cap.read()
        if not ret:
            return

        results = model(frame)[0]
        self.detect_lanes(frame)

        decision = "GO"
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name in important_classes:
                color = (0, 255, 255) if class_name == "person" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

                # Decision logic
                if class_name == "person" and (y2 - y1) > 200:
                    decision = "STOP - PEDESTRIAN"
                elif class_name == "stop sign":
                    decision = "STOP - SIGN"
                elif class_name == "traffic light":
                    decision = "TRAFFIC LIGHT AHEAD"
                elif class_name in ['car', 'bus', 'truck', 'motorbike', 'bicycle']:
                    decision = "OBSTACLE AHEAD"

        # Update FPS
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time

        # Update UI
        self.status_label.config(text=f"Alert: {decision}", fg="red" if "STOP" in decision else "lime")
        self.fps_label.config(text=f"FPS: {int(fps)}")

        # Convert image and show
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

        self.root.after(1, self.update)

root = tk.Tk()
app = App(root)
root.mainloop()
cap.release()
