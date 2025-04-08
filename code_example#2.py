!pip install yt-dlp opencv-python ultralytics

import os
import cv2
import time
import yt_dlp
from ultralytics import YOLO

# Step 1: Download YouTube video
def download_youtube_video(url, output_path="input_video.mp4"):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': output_path,
        'merge_output_format': 'mp4'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

# Step 2: Process each frame
def process_video(input_path, output_path, model_name="yolov8m.pt"):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps_input, (width, height))

    model = YOLO(model_name)
    important_classes = ['person', 'car', 'bus', 'truck', 'motorbike', 'bicycle', 'traffic light', 'stop sign']

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model(frame)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name in important_classes:
                label = f"{class_name}: {conf:.2f}"
                color = (0, 255, 255) if class_name == "person" else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

                # Warnings
                if class_name == "person" and (x2 - x1) * (y2 - y1) > 50000:
                    cv2.putText(frame, "⚠️ Pedestrian Nearby!", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                elif class_name == "traffic light":
                    cv2.putText(frame, "🚦 Traffic Light Detected", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                elif class_name == "stop sign":
                    cv2.putText(frame, "🛑 Stop Sign Detected", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        out.write(frame)
        print(f"Processed frame: {frame_count}", end='\r')

    cap.release()
    out.release()
    print(f"\n✅ Processing complete. Saved to: {output_path}")

# Step 3: Use the pipeline
if __name__ == "__main__":
    youtube_url = input("Enter YouTube video URL: ")
    downloaded_video = "input_video.mp4"
    output_video = "labeled_output.mp4"

    print("📥 Downloading video...")
    download_youtube_video(youtube_url, downloaded_video)

    print("🔍 Processing video with YOLOv8...")
    process_video(downloaded_video, output_video)

    print(f"🎬 Done! Output saved as {output_video}")
