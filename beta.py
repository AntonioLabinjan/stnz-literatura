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
    'person',             
    'car',                
    'bus',                
    'truck',              
    'motorbike',          
    'bicycle',            
    'traffic light',      
    'stop sign',         
    'parking meter',      
    'fire hydrant',       
    'bench',              
    'stroller',           
    'wheelchair',         
    'traffic cone',       
    'backpack',           
    'umbrella',           
    'skateboard',         
    'suitcase',           
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
]





# omogućava govor
engine = pyttsx3.init()

class App:
    # inicijalizacija objekta app klase (dela se "frontend")
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Driving Dashboard")
        self.root.geometry("1280x800")
        self.root.configure(bg="#1e1e1e")

        # Glavni frame
        main_frame = tk.Frame(root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True)

        # Video feed - livo
        self.video_frame = tk.Label(main_frame, bg="#1e1e1e")
        self.video_frame.grid(row=0, column=0, padx=20, pady=20)

        # Info panel - desno
        info_frame = tk.Frame(main_frame, bg="#2c2c2c", width=400)
        info_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 20), pady=20)
        info_frame.grid_propagate(False)

        # Naslov
        title = tk.Label(info_frame, text="Live Dashboard", font=("Arial", 20, "bold"), fg="#00FFB3", bg="#2c2c2c")
        title.pack(pady=10)

        # Status
        self.status_label = tk.Label(info_frame, text="Status: GO", font=("Helvetica", 18), fg="lime", bg="#2c2c2c")
        self.status_label.pack(pady=10)

        # FPS
        self.fps_label = tk.Label(info_frame, text="FPS: 0", font=("Helvetica", 14), fg="white", bg="#2c2c2c")
        self.fps_label.pack()

        # Vidljivost
        self.visibility_label = tk.Label(info_frame, text="", font=("Helvetica", 12), fg="orange", bg="#2c2c2c", wraplength=380)
        self.visibility_label.pack(pady=10)
        # Visibility Progress Bar
        self.visibility_bar = ttk.Progressbar(info_frame, orient="horizontal", length=300, mode="determinate")
        self.visibility_bar.pack(pady=(5, 20))


        # Logs
        self.log_title = tk.Label(info_frame, text="Detections:", font=("Helvetica", 14, "bold"), fg="#00FFB3", bg="#2c2c2c")
        self.log_title.pack(pady=(20, 0))

        self.log_text = tk.Text(info_frame, height=12, width=45, bg="#1e1e1e", fg="white", font=("Courier", 10))
        self.log_text.pack(pady=(5, 20))
        self.log_text.config(state=tk.DISABLED)

        # Queue za komunikaciju između threadova
        self.queue = queue.Queue()
        self.prev_time = time.time()
        self.last_alert_time = time.time()

        # Thread za video obradu
        self.processing_thread = threading.Thread(target=self.process_video_feed)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.update_ui()


    # provjena udaljenosti za detektirane objekte
    def estimate_distance(self, height):
        return round(10000 / (height + 1), 2)

    # izgovaranje poruka svakih 1.5 sekundu
    # stavljeno u svoj thread da se glavna aplikacija ne sprži
    def play_alert(self, message):
        def speak():
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()

        if time.time() - self.last_alert_time > 1.5:
            self.last_alert_time = time.time()
            threading.Thread(target=speak, daemon=True).start()

    # logiranje svih detektiranih stvari. UI se dinamički refresha s novim detekcijama (12 linija mu je max veličina)
    def log_detection(self, entry):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{entry}\n")
        self.log_text.see(tk.END)
        lines = self.log_text.get("1.0", tk.END).splitlines() # odvajanje linija; da nemamo samo 1 jako dugu
        if len(lines) > 12:
            self.log_text.delete("1.0", "2.0")
        self.log_text.config(state=tk.DISABLED)
    
    # procesiranje videa
    def process_video_feed(self):
        # sve dok je aplikacija upaljena, lovimo frameove i čitamo ih
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # u results se sprema lista svega ča yolo izdetektira/predvidi...analizira frameove kao kontinuirani stream
            results = list(model.predict(source=frame, stream=True))[0]

            # defaultna odluka je GO (ako ne vidiš niš opasno, samo kreni)
            decision = "GO"

            # za svaki yolo bounding box definiraj koja je predviđena klasa(koji su uočeni bitni featuresi), koji je confidence (koliko % je model siguran) i koji je točan naziv klase
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # ako je predviđena klasa neka od onih klasa koje su definirane kao važne
                if class_name in important_classes:
                    # izračunaj visinu i procijeni udaljenost prema formuli gore
                    height = y2 - y1
                    distance = self.estimate_distance(height)

                    color = (0, 255, 255) if class_name == "person" else (0, 255, 0) # žuto ako je osoba, zeleno ako je bilo ča drugo
                    # nacrtaj bbox i stavi labelu + udaljensst
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_name}: {conf:.2f} | {distance}cm", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    # vidi vrijeme detekcije i logiraj ga u formatu time class name
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    self.log_detection(f"[{timestamp}] {class_name} ({conf:.2f})")
                    
                    # ako je osoba i ako je jako blizu (height > 200)
                    # za ostale klase ne gleda distance nego samo ako postoje
                    if class_name == "person" and height > 200:
                        decision = "STOP - PEDESTRIAN"
                        self.play_alert(f"Stop, pedestrian detected at {distance} cm")
                    elif class_name == "stop sign":
                        decision = "STOP - SIGN"
                        self.play_alert(f"Stop, stop sign detected at {distance} cm")
                    elif class_name == "traffic light":
                        decision = "WATCH OUT - TRAFFIC LIGHT AHEAD"
                        self.play_alert(f"Watch out, traffic light detected at {distance} cm")
                    elif class_name in ['car', 'bus', 'truck', 'motorbike', 'bicycle']:
                        decision = "WATCH OUT - VEHICLE AHEAD"
                        self.play_alert(f"Watch out, vehicle detected at {distance} cm")
                    elif class_name in ['parking meter']:
                        decision = "WATCH OUT - PARKING NEAR"
                        self.play_alert(f"Parking is near. You can park at {distance} cm")
                    elif class_name in ['traffic cone', 'helmet']:
                        decision = "WATCH OUT - ROADWORKS"
                        self.play_alert(f"Watch out, roadworks ahead at {distance} cm")
                    elif class_name in ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]:
                        decision = "WATCH OUT - ANIMAL"
                        self.play_alert(f"Watch out, animal detected at {distance} cm")

            # praćenje trenutnog vremena, izračun fps-a
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) # računa koliko frameova se odvrti u sekundi
            self.prev_time = curr_time

            # cooldown od 0.03 sekundi između donošenja odluka da se moru pročitat (bilo bi epileptično i nečitko da se ispisuju with each detection)
            self.queue.put((frame, decision, fps)) # queue za odluke
            time.sleep(0.03)

    def update_ui(self):
        # ako queue ni prazan (ako je donesena neka odluka)
        if not self.queue.empty():
            # dohvat odluke iz queuea
            frame, decision, fps = self.queue.get()
            # ako poruka sadrži stop, farbaj crveno, ako sadrži watch out, farbaj žuto, ako ne, farbaj zeleno
            color = "red" if "STOP" in decision else "yellow" if "WATCH OUT" in decision else "lime"
            # ispis odluke i FPS-a
            self.status_label.config(text=f"Status: {decision}", fg=color)

            self.fps_label.config(text=f"FPS: {int(fps)}")

            # pretvaranje obojane slike u grayscale za lakšu obradu
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # računamo prosječnu svjetlost svih frameova
            brightness = np.mean(gray)
            self.visibility_bar["value"] = min(max(int(brightness * 2), 0), 100)
            # ako je manja od 50, svjetlost je niska -> alert
            if brightness < 50:
                self.visibility_label.config(text="⚠️ Low visibility - turn your lights on!", fg="orange")
                self.play_alert("Low visibility - turn your lights on!")
            # ako je veća, onda je OK, nema alerta
            else:
                self.visibility_label.config(text="")

            # pretvaranje OpenCV framea u Tkinter sliku i svakih 10 ms osvježava prikaz videa
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
