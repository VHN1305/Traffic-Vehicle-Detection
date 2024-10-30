import cv2
import numpy as np
from collections import defaultdict
import time
from ultils import is_point_in_polygon
from ultralytics import YOLO

class VehicleDetection:
    def __init__(self, model_path, max_track_length=30, parked_threshold=10):
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 5, 6, 7, 8, 10]
        self.track_history = defaultdict(list)  # Store history of track IDs and their positions
        self.max_track_length = max_track_length  # Maximum number of points to store per vehicle
        self.zone_config = {
            "zone1": [(15, 120), (38, 450), (450, 450), (100, 120)],
            "zone2": [(150, 120), (500, 300), (620, 260), (280, 120)]
        }
        self.lane_vector_direction = {
            "zone1": (1, 1),
            "zone2": (1, 1)
        }
        self.max_template_width = 0
        self.max_template_height = 0
        self.parked_threshold = parked_threshold  # Threshold for number of consecutive frames to check for parking
        self.track_templates = defaultdict(list)
        self.box_templates = defaultdict(list)
        self.pack_start_time_templates = defaultdict(float)  # Ensure this is a dictionary of floats
        self.pack_current_time_templates = defaultdict(float)
        self.opposite_vehicle_direction = []
        self.start_time_at_opposite_direction = defaultdict(float)
        self.update_pack_time = defaultdict(float)

    def still_packed(self, frame, box, template, start_time, track_id, current_time, match_threshold=0.8):
        img = frame[max(0, int(box[1]-10)):min(frame.shape[0], int(box[3]+10)),
                  max(0, int(box[0]-10)):min(frame.shape[1], int(box[2]+10))
                  ]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(results >= match_threshold)
        if len(locations[0]):
            if time.time() - self.update_pack_time[track_id] > 20:
                self.update_pack_time[track_id] = time.time()
                new_template = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                new_template = cv2.cvtColor(new_template, cv2.COLOR_BGR2GRAY)
                cv2.imshow("Before Update",self.track_templates[track_id][0])
                self.track_templates[track_id] = [new_template]
                cv2.imshow("After Update", self.track_templates[track_id][0])
            self.pack_current_time_templates[track_id] = time.time()
            return True
        else:
            if time.time() - current_time > 10:
                for key, value in self.track_templates.items():
                    if any(np.array_equal(template, t) for t in value):
                        track_id = key
                        # pop the template if it has been packed for more than 10 seconds
                        del self.pack_current_time_templates[track_id]
                        del self.track_templates[track_id]
                        del self.box_templates[track_id]
                        del self.pack_start_time_templates[track_id]
                        break
            return False

    # Other methods remain unchanged

    def find_max_template_size(self, templates):
        for template_id, template in templates.items():
            for track_id in range(len(template)):
                self.max_template_width = max(self.max_template_width, template[track_id].shape[1])
                self.max_template_height = max(self.max_template_height, template[track_id].shape[0])

    def is_match_template(self, image, templates, match_threshold=0.6):
        """
        :param image: image size must be larger than templates size
        :param templates: hastable of templates
        :param match_threshold: threshold to consider a match
        :return: template_id where the image is matched with the template
        """
        if len(templates):
            for template_id, template in templates.items():
                for track_id in range(len(template)):
                    result = cv2.matchTemplate(image, template[track_id], cv2.TM_CCOEFF_NORMED)
                    locations = np.where(result >= match_threshold)
                    if len(locations[0]):
                        return True
        # return None
        return False

    def track(self, frame, is_check_in_zone=False, is_check_parking=False, is_check_template=False):
        annotated_frame = frame.copy()
        if is_check_in_zone:
            for key, value in self.zone_config.items():
                self.plot_zone(annotated_frame, value)

        results = self.model.track(frame, tracker="track.yaml", persist=True, verbose=False, conf=0.05)
        boxes = results[0].boxes.xywh.cpu()  # Get bounding boxes in xywh format (center x, center y, width, height)
        track_ids = results[0].boxes.id  # Get the track IDs for each detected object

        if track_ids is None:
            return annotated_frame

        track_ids = track_ids.int().cpu().tolist()

        active_track_ids = set()  # Keep track of the current frame's active IDs

        track_templates_items = list(self.track_templates.items())
        box_templates_items = list(self.box_templates.items())
        pack_start_time_templates_items = list(self.pack_start_time_templates.items())
        pack_current_time_templates_items = list(self.pack_current_time_templates.items())

        for (template_id, template), (_, box), (track_id, start_time), (_, current_time) in zip(track_templates_items, box_templates_items,
                                                                             pack_start_time_templates_items, pack_current_time_templates_items):
            for idx in range(len(template)):
                if self.still_packed(frame, box[idx], template[idx], start_time, track_id, current_time):
                    vehicle_color = (255, 0, 0)
                    x1_template, y1_template, x2_template, y2_template = box[idx]
                    annotated_frame = cv2.rectangle(
                        annotated_frame,
                        (int(x1_template), int(y1_template)),  # top-left corner
                        (int(x2_template), int(y2_template)),  # bottom-right corner
                        vehicle_color,
                        2
                    )
                    pack_time = time.time() - start_time
                    annotated_frame = cv2.putText(
                        annotated_frame,
                        f"{pack_time:.2f}",
                        (int(x1_template), int(y1_template - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,  # font scale
                        vehicle_color,
                        2
                    )

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box

            if track_id in self.opposite_vehicle_direction:
                x1, y1, x2, y2 = self.xywh_to_xyxy(box)
                vehicle_in_opposite_direction = frame[int(y1):int(y2), int(x1):int(x2)]
                time_vehicle_in_opposite_direction = self.start_time_at_opposite_direction[track_id]
                cv2.imshow("Vehicle in opposite direction", vehicle_in_opposite_direction)

            # Only consider vehicle classes
            if int(results[0].boxes.cls[track_ids.index(track_id)]) in self.vehicle_classes:

                vehicle_color = (0, 255, 0)

                # Check if the vehicle is in the zone
                if is_check_in_zone:
                    is_vehicle_in_zone = False
                    for _, zone in self.zone_config.items():
                        if self.check_box_in_zone((x, y, w, h), zone):
                            is_vehicle_in_zone = True
                            break
                    if is_vehicle_in_zone:
                        vehicle_color = (0, 0, 255)
                        # Check for parking status
                        x1_template, y1_template, x2_template, y2_template = self.xywh_to_xyxy(box)

                        box_width = x2_template - x1_template
                        box_height = y2_template - y1_template

                        # if track_id not in self.track_templates.keys():
                        new_template = frame[max(0, int(y1_template)):min(frame.shape[0], int(y2_template)),
                                       max(0, int(x1_template)):min(frame.shape[1], int(x2_template))]
                        self.find_max_template_size(self.track_templates)

                        x1_temp, y1_temp, x2_temp, y2_temp = x1_template, y1_template, x2_template, y2_template
                        if box_width < self.max_template_width:
                            x1_temp = int(x1_template - (self.max_template_width - box_width) / 2 - 10)
                            x2_temp = int(x2_template + (self.max_template_width - box_width) / 2 + 10)
                        if box_height < self.max_template_height:
                            y1_temp = int(y1_template - (self.max_template_height - box_height) / 2 - 10)
                            y2_temp = int(y2_template + (self.max_template_height - box_height) / 2 + 10)
                        if x1_temp < 0:
                            x2_temp = x2_temp + abs(x1_temp) + 10
                            x1_temp = 0
                        if y1_temp < 0:
                            y2_temp += y2_temp + abs(y1_temp) + 10
                            y1_temp = 0
                        if x2_temp > frame.shape[1]:
                            x1_temp -= x1_temp - (x2_temp - frame.shape[1]) - 10
                            x2_temp = frame.shape[1]
                        if y2_temp > frame.shape[0]:
                            y1_temp -= y1_temp - (y2_temp - frame.shape[0]) - 10
                            y2_temp = frame.shape[0]
                        track_template = frame[int(y1_temp):int(y2_temp), int(x1_temp):int(x2_temp)]

                        track_template = cv2.cvtColor(track_template, cv2.COLOR_BGR2GRAY)
                        new_template = cv2.cvtColor(new_template, cv2.COLOR_BGR2GRAY)

                        matched_track_id = self.is_match_template(track_template, self.track_templates)
                        if matched_track_id:
                            continue

                        if is_check_parking:
                            if not matched_track_id and self.is_parked(track_id):
                            # if not matched_track_id:
                                self.track_templates[track_id].append(new_template)
                                self.box_templates[track_id].append(
                                    (x1_template, y1_template, x2_template, y2_template))
                                self.pack_start_time_templates[track_id] = time.time()
                                self.pack_current_time_templates[track_id] = time.time()
                                self.update_pack_time[track_id] = time.time()

                active_track_ids.add(track_id)
                # Store the current center point of the vehicle in the tracking history
                self.track_history[track_id].append((float(x), float(y)))

                # Limit the length of the tracking history to max_track_length frames
                if len(self.track_history[track_id]) > self.max_track_length:
                    self.track_history[track_id].pop(0)
                # Draw bounding box and label
                annotated_frame = cv2.rectangle(
                    annotated_frame,
                    (int(x - w / 2), int(y - h / 2)),  # top-left corner
                    (int(x + w / 2), int(y + h / 2)),  # bottom-right corner
                    vehicle_color,
                    2
                )

        # Remove history of objects that are no longer in the frame
        self.clean_up_tracks(active_track_ids)

        # Plot tracking history lines for each vehicle
        for zone, direction in zip(self.zone_config.values(), self.lane_vector_direction.values()):
            annotated_frame = self.plot_tracks(annotated_frame, direction, zone)

        return annotated_frame

    def is_parked(self, track_id, distance_threshold=1):
        # Check if a vehicle has remained in the same position for the past parked_threshold frames
        track = self.track_history.get(track_id, [])
        if len(track) < self.parked_threshold:
            return False
        # Compare the distance between the first and last points in the thresholded track
        start_point = np.array(track[-self.parked_threshold])
        end_point = np.array(track[-1])
        distance = np.linalg.norm(end_point - start_point)
        return distance < distance_threshold  # Distance threshold to consider as parked

    def clean_up_tracks(self, active_track_ids):
        track_ids_to_remove = [track_id for track_id in self.track_history if track_id not in active_track_ids]
        for track_id in track_ids_to_remove:
            del self.track_history[track_id]

    def plot_tracks(self, frame, lane_vector_direction, zone):
        for track_id, track in self.track_history.items():
            if len(track) > 1:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 255, 255), thickness=2)

            if len(track) > 10:
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                count_point_in_zone = 0
                for point in points:
                    if is_point_in_polygon((point[0][0], point[0][1]), zone):
                        count_point_in_zone += 1

                if count_point_in_zone > 10:
                    start_point = np.array(track[0])
                    end_point = np.array(track[-1])
                    # calculate distance of two points
                    distance = np.linalg.norm(end_point - start_point)
                    if distance > 10:
                        direction = end_point - start_point
                        cosin = np.dot(direction, lane_vector_direction) / (
                                np.linalg.norm(direction) * np.linalg.norm(lane_vector_direction))
                        if cosin < 0:
                            if track_id not in self.opposite_vehicle_direction:
                                self.opposite_vehicle_direction.append(track_id)
                                self.start_time_at_opposite_direction[track_id] = time.time()
                            cv2.polylines(frame, [points], isClosed=False, color=(0, 255, 255), thickness=2)
        return frame

    def xywh_to_xyxy(self, box):
        x, y, w, h = box
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2

    def check_box_in_zone(self, box, zone):
        x_center, y_center, w, h = box
        x1_box, y1_box, x2_box, y2_box = self.xywh_to_xyxy(box)
        return is_point_in_polygon((x1_box, y1_box), zone) or is_point_in_polygon((x2_box, y2_box), zone) or \
               is_point_in_polygon((x1_box, y2_box), zone) or is_point_in_polygon((x2_box, y1_box), zone) or \
               is_point_in_polygon((x_center, y_center), zone)

    def plot_zone(self, frame, zone):
        # Convert polygon to integer coordinates
        polygon_pts = np.array(zone, dtype=np.int32)
        # Draw the polygon
        cv2.polylines(frame, [polygon_pts], isClosed=True, color=(0, 0, 250), thickness=2)

        cv2.arrowedLine(frame, (40, 120), (80, 400), (255, 255, 255), 2)
        cv2.arrowedLine(frame, (610, 270), (270, 130), (255, 255, 255), 2)
        return frame



if __name__ == "__main__":
    model_path = "model/yolov8n.pt"
    vehicle_detection = VehicleDetection(model_path)

    # URL của luồng RTSP
    # rtsp_url = "rtsp://admin:Admin123@qmh1.cameraddns.net:8102/ISAPI/Streaming/Channels/302"
    rtsp_url = "data/video_data/output_video8.avi"
    cap = cv2.VideoCapture(rtsp_url)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            annotated_frame = vehicle_detection.track(frame, is_check_in_zone=True, is_check_parking=True, is_check_template=True)
            cv2.imshow("Vehicle Detection", cv2.resize(annotated_frame, (900, 900)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()