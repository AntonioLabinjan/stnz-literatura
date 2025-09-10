import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import time
import numpy as np
from datetime import datetime
import pyttsx3
import threading
import queue

# Init yolo & camera capture
model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

# sve klase koje bi imalo smisla za prepoznat u vožnji
important_classes = [
    'person','car','bus','truck','motorbike','bicycle',
    'traffic light','stop sign','parking meter','fire hydrant','bench',
    'stroller','wheelchair','traffic cone','backpack','umbrella',
    'skateboard','suitcase',"bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe"
]

# ikone za log
class_icons = {
    "person":"🚶",
    "car":"🚗","bus":"🚌","truck":"🚛","motorbike":"🏍️","bicycle":"🚲",
    "traffic light":"🚦","stop sign":"🛑","parking meter":"🅿️","traffic cone":"⚠️",
    "dog":"🐕","cat":"🐈","horse":"🐎","cow":"🐄","sheep":"🐑","elephant":"🐘",
    "bear":"🐻","zebra":"🦓","giraffe":"🦒","bird":"🐦"
}

# omogućava govor
engine = pyttsx3.init()

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Driving Dashboard")
        self.root.geometry("1280x800")
        self.root.configure(bg="#1e1e1e")

        main_frame = tk.Frame(root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True)

        # Video feed - lijevo
        self.video_frame = tk.Label(main_frame, bg="#1e1e1e")
        self.video_frame.grid(row=0, column=0, padx=20, pady=20)

        # Info panel - desno
        info_frame = tk.Frame(main_frame, bg="#2c2c2c", width=400)
        info_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        info_frame.grid_propagate(False)

        title = tk.Label(info_frame, text="Live Dashboard", font=("Arial", 20, "bold"), fg="#00FFB3", bg="#2c2c2c")
        title.pack(pady=10)

        # Status
        self.status_label = tk.Label(info_frame, text="Status: GO", font=("Helvetica", 18, "bold"), fg="lime", bg="#2c2c2c")
        self.status_label.pack(pady=10)

        # FPS meter (kružni)
        self.fps_canvas = tk.Canvas(info_frame, width=120, height=120, bg="#2c2c2c", highlightthickness=0)
        self.fps_canvas.pack(pady=10)

        # Vidljivost
        self.visibility_label = tk.Label(info_frame, text="", font=("Helvetica", 12), fg="orange", bg="#2c2c2c", wraplength=380)
        self.visibility_label.pack(pady=10)
        self.visibility_bar = ttk.Progressbar(info_frame, orient="horizontal", length=300, mode="determinate")
        self.visibility_bar.pack(pady=(5, 20))

        # Logs
        self.log_title = tk.Label(info_frame, text="Detections:", font=("Helvetica", 14, "bold"), fg="#00FFB3", bg="#2c2c2c")
        self.log_title.pack(pady=(20, 0))

        self.log_text = tk.Text(info_frame, height=12, width=45, bg="#1e1e1e", fg="white", font=("Courier", 10))
        self.log_text.pack(pady=(5, 20))
        self.log_text.config(state=tk.DISABLED)

        # Queue za komunikaciju
        self.queue = queue.Queue()
        self.prev_time = time.time()
        self.last_alert_time = time.time()

        # Thread za video obradu
        self.processing_thread = threading.Thread(target=self.process_video_feed)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.update_ui()

    def estimate_distance(self, height):
        return round(10000 / (height + 1), 2)

    def play_alert(self, message):
        def speak():
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
        if time.time() - self.last_alert_time > 1.5:
            self.last_alert_time = time.time()
            threading.Thread(target=speak, daemon=True).start()

    def log_detection(self, entry):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{entry}\n")
        self.log_text.see(tk.END)
        lines = self.log_text.get("1.0", tk.END).splitlines()
        if len(lines) > 12:
            self.log_text.delete("1.0", "2.0")
        self.log_text.config(state=tk.DISABLED)

    def process_video_feed(self):
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            results = list(model.predict(source=frame, stream=True))[0]
            decision = "GO"
            priority = 0  # veći broj = veća opasnost

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                if class_name in important_classes:
                    height = y2 - y1
                    distance = self.estimate_distance(height)

                    color = (0, 255, 255) if class_name == "person" else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name}: {conf:.2f} | {distance}cm", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    timestamp = datetime.now().strftime("%H:%M:%S")
                    icon = class_icons.get(class_name, "❓")
                    self.log_detection(f"[{timestamp}] {icon} {class_name} ({conf:.2f})")

                    # prioritizacija
                    if class_name == "person" and height > 200:
                        if priority < 3:
                            decision = "STOP - PEDESTRIAN"
                            priority = 3
                            self.play_alert(f"Stop, pedestrian detected at {distance} cm")
                    elif class_name == "stop sign":
                        if priority < 3:
                            decision = "STOP - SIGN"
                            priority = 3
                            self.play_alert(f"Stop, stop sign detected at {distance} cm")
                    elif class_name == "traffic light":
                        if priority < 2:
                            decision = "WATCH OUT - TRAFFIC LIGHT AHEAD"
                            priority = 2
                            self.play_alert(f"Watch out, traffic light detected at {distance} cm")
                    elif class_name in ['car', 'bus', 'truck', 'motorbike', 'bicycle']:
                        if priority < 2:
                            decision = "WATCH OUT - VEHICLE AHEAD"
                            priority = 2
                            self.play_alert(f"Watch out, vehicle detected at {distance} cm")
                    elif class_name in ['traffic cone']:
                        if priority < 1:
                            decision = "WATCH OUT - ROADWORKS"
                            priority = 1
                            self.play_alert(f"Watch out, roadworks ahead at {distance} cm")
                    elif class_name in ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]:
                        if priority < 1:
                            decision = "WATCH OUT - ANIMAL"
                            priority = 1
                            self.play_alert(f"Watch out, animal detected at {distance} cm")

            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            self.queue.put((frame, decision, fps))
            time.sleep(0.03)

    def draw_fps_meter(self, fps):
        self.fps_canvas.delete("all")
        angle = min(int(fps), 60) * 6
        self.fps_canvas.create_oval(10, 10, 110, 110, outline="#444", width=6)
        self.fps_canvas.create_arc(10, 10, 110, 110, start=90, extent=-angle, outline="#00FFB3", width=6, style="arc")
        self.fps_canvas.create_text(60, 60, text=f"{int(fps)} FPS", fill="white", font=("Helvetica", 12, "bold"))

    def update_ui(self):
        if not self.queue.empty():
            frame, decision, fps = self.queue.get()
            color = "red" if "STOP" in decision else "yellow" if "WATCH OUT" in decision else "lime"
            self.status_label.config(text=f"Status: {decision}", fg=color)

            self.draw_fps_meter(fps)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            self.visibility_bar["value"] = min(max(int(brightness * 2), 0), 100)
            if brightness < 50:
                self.visibility_label.config(text="⚠️ Low visibility - turn your lights on!", fg="orange")
                self.play_alert("Low visibility - turn your lights on!")
            else:
                self.visibility_label.config(text="")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.root.after(10, self.update_ui)

# Pokretanje aplikacije
root = tk.Tk()
app = App(root)
root.mainloop()
cap.release()
