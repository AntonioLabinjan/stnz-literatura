import tkinter as tk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time
import numpy as np
from datetime import datetime
import pyttsx3
import threading
import asyncio
import queue

engine = pyttsx3.init()

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

        self.visibility_label = tk.Label(root, text="", font=("Helvetica", 14), fg="orange", bg="#222")
        self.visibility_label.pack()

        self.log_label = tk.Label(root, text="Detections log:", font=("Helvetica", 12), fg="white", bg="#222", justify="left")
        self.log_label.pack(pady=10)

        self.detection_log = []
        self.last_alert_time = time.time()

        self.prev_time = time.time()

        self.queue = queue.Queue()

        self.processing_thread = threading.Thread(target=self.process_video_feed)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.update()

    def log_detection(self, class_name, confidence):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp} - {class_name} ({confidence:.2f})"
        self.detection_log.append(entry)

        if len(self.detection_log) > 10:
            self.detection_log.pop(0)

        self.log_label.config(text="Detections log:\n" + "\n".join(self.detection_log))

    def estimate_distance(self, height):
        return round(10000 / (height + 1), 2)


    def play_alert(self, message):
        def speak():
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()

        if time.time() - self.last_alert_time > 1:
            self.last_alert_time = time.time()
            threading.Thread(target=speak, daemon=True).start()


    def process_video_feed(self):
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            results = list(model.predict(source=frame, stream=True))[0]


            decision = "GO"
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name in important_classes:
                    color = (0, 255, 255) if class_name == "person" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    height = y2 - y1
                    distance = self.estimate_distance(height)
                    cv2.putText(frame, f"{class_name}: {conf:.2f} | {distance}cm", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                    self.log_detection(class_name, conf)

                    if class_name == "person" and (y2 - y1) > 200:
                        decision = "STOP - PEDESTRIAN"
                        self.play_alert(f"Stop, pedestrian detected at {distance} cm")
                    elif class_name == "stop sign":
                        decision = "STOP - SIGN"
                        self.play_alert(f"Stop sign detected at {distance} cm")
                    elif class_name == "traffic light":
                        decision = "TRAFFIC LIGHT AHEAD"
                        self.play_alert(f"Traffic light detected at {distance} cm")
                    elif class_name in ['car', 'bus', 'truck', 'motorbike', 'bicycle']:
                        decision = "OBSTACLE AHEAD"
                        self.play_alert(f"Obstacle detected at {distance} cm")

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            self.queue.put((frame, decision, fps))  

            time.sleep(0.03)  # Pauza da se ne optereti procesor

    def update(self):
        if not self.queue.empty():
            frame, decision, fps = self.queue.get()

            self.status_label.config(text=f"Alert: {decision}", fg="red" if "STOP" in decision else "lime")
            self.fps_label.config(text=f"FPS: {int(fps)}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)

            if brightness < 50:
                self.visibility_label.config(text="Low visibility detected – please turn on headlights!", fg="orange")
            else:
                self.visibility_label.config(text="")

            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)

        self.root.after(1, self.update)  # Pozovi se ponovo nakon 1ms

root = tk.Tk()
app = App(root)
root.mainloop()
cap.release()
