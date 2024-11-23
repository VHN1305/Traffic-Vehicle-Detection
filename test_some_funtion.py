from ultralytics import YOLO

model = YOLO("model/best.pt")

results = model.track("data/video_data/output_video1.avi", show=True)