import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

class VehicleDetection:
    def __init__(self, model_path, max_track_length=30):
        self.model = YOLO(model_path)
        self.vehicle_classes = [0, 1, 2, 5, 6, 7, 8, 10]
        self.track_history = defaultdict(list)  # Store history of track IDs and their positions
        self.max_track_length = max_track_length  # Maximum number of points to store per
        self.zone_config = {
            "zone1": (0, 0, 250, 250),
            "zone2": (0, 0, 100, 100),
            "zone3": (0, 0, 100, 100),
            "zone4": (0, 0, 100, 100),
        }

    def track(self, frame, is_check_in_zone=False, is_check_parking=False):

        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()
        if is_check_in_zone:
            self.plot_zone(annotated_frame, self.zone_config["zone1"])
        # # clache
        # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
        # cla = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))
        # annotated_frame = cla.apply(annotated_frame)
        # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_GRAY2BGR)

        results = self.model.track(frame, tracker="track.yaml", persist=True, verbose=False, conf=0.05)

        # Get the boxes and track IDs (check if track IDs exist)
        boxes = results[0].boxes.xywh.cpu()  # Get bounding boxes in xywh format (center x, center y, width, height)
        track_ids = results[0].boxes.id  # Get the track IDs for each detected object

        # If track_ids is None, skip further processing
        if track_ids is None:
            return annotated_frame

        track_ids = track_ids.int().cpu().tolist()  # Convert track IDs to list

        active_track_ids = set()  # Keep track of the current frame's active IDs

        # Iterate through detections
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box  # Unpack the bounding box (center x, center y, width, height)

            # Only consider vehicle classes
            if int(results[0].boxes.cls[track_ids.index(track_id)]) in self.vehicle_classes:
                active_track_ids.add(track_id)  # Add track ID to the active list

                # Store the current center point of the vehicle in the tracking history
                self.track_history[track_id].append((float(x), float(y)))

                # Limit the length of the tracking history to max_track_length frames
                if len(self.track_history[track_id]) > self.max_track_length:
                    self.track_history[track_id].pop(0)

                vehicle_color = (0, 255, 0)

                if is_check_in_zone:
                    if self.check_box_in_zone((x, y, w, h), self.zone_config["zone1"]):
                        print("Vehicle in zone1")
                        vehicle_color = (0, 0, 255)
                # Draw bounding box and label (as 'car')
                annotated_frame = cv2.rectangle(
                    annotated_frame,
                    (int(x - w / 2), int(y - h / 2)),  # top-left corner
                    (int(x + w / 2), int(y + h / 2)),  # bottom-right corner
                    vehicle_color,  # color (green)
                    2  # thickness
                )
                label = 'vehicle'
                annotated_frame = cv2.putText(
                    annotated_frame,
                    label,
                    (int(x - w / 2), int(y - h / 2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # font scale
                    vehicle_color,  # color (green)
                    2  # thickness
                )

        # Remove history of objects that are no longer in the frame
        self.clean_up_tracks(active_track_ids)

        # Plot tracking history lines for each vehicle
        annotated_frame = self.plot_tracks(annotated_frame)

        return annotated_frame

    def clean_up_tracks(self, active_track_ids):
        # Remove track history for any track ID that is no longer active
        track_ids_to_remove = [track_id for track_id in self.track_history if track_id not in active_track_ids]
        for track_id in track_ids_to_remove:
            del self.track_history[track_id]

    def plot_tracks(self, frame):
        # Iterate through each tracked object and plot its history
        for track_id, track in self.track_history.items():
            if len(track) > 1:  # If the track has more than one point
                # Convert the track points to the format required for cv2.polylines
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                # Draw the tracking lines (white with thickness 2)
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)
        return frame

    def xywh_to_xyxy(self, box):
        x, y, w, h = box
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2

    def check_box_in_zone(self, box, zone):
        x1_box, y1_box, x2_box, y2_box = self.xywh_to_xyxy(box)
        x1_zone, y1_zone, x2_zone, y2_zone = zone
        # return (x1_zone < x1_box < x2_zone and y1_zone < y1_box < y2_zone) or \
        #        (x1_zone < x2_box < x2_zone and y1_zone < y2_box < y2_zone) or \
        #        (x1_zone < x1_box < x2_zone and y1_zone < y2_box < y2_zone) or \
        #        (x1_zone < x2_box < x2_zone and y1_zone < y1_box < y2_zone)
        return (x1_zone <= x1_box <= x2_zone and y1_zone <= y1_box <= y2_zone) or \
            (x1_zone <= x2_box <= x2_zone and y1_zone <= y2_box <= y2_zone)

    def plot_zone(self, frame, zone):
        x1, y1, x2, y2 = zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
        return frame



if __name__ == "__main__":
    model_path = "model/yolov8n.pt"
    vehicle_detection = VehicleDetection(model_path)

    # URL của luồng RTSP
    rtsp_url = "rtsp://admin:Admin123@qmh1.cameraddns.net:8102/ISAPI/Streaming/Channels/202"
    cap = cv2.VideoCapture(rtsp_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            annotated_frame = vehicle_detection.track(frame, is_check_in_zone=True)
            cv2.imshow("Vehicle Detection", cv2.resize(annotated_frame, (900, 900)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()