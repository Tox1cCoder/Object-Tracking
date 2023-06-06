import math
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

object_id = 0

class EuclideanDistTracker:
    def __init__(self, svm_model):
        self.tracks = []
        self.svm_model = svm_model
        self.nms_threshold = 0.3

    def detect_vehicle(self, img):
        img = cv2.resize(img, (64, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray)
        features = features.flatten().reshape(1, -1)
        pred = self.svm_model.predict(features)
        return pred

    def update(self, detections, frame):

        global object_id  # Access the global object ID variable

        if len(self.tracks) == 0:
            for det in detections:
                x, y, w, h = det
                cx = x + (w // 2)
                cy = y + (h // 2)

                roi = frame[y:y + h, x:x + w]
                is_vehicle = self.detect_vehicle(roi)

                if is_vehicle:
                    self.tracks.append({
                        'id': object_id,  # Assign the current object ID
                        'bbox': (x, y, w, h),
                        'centroid': (cx, cy),
                        'age': 1,
                    })
                    object_id += 1  # Increment the object ID

        else:
            # Create detection matrix
            detection_matrix = np.zeros((len(self.tracks), len(detections)))
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detections):
                    x, y, w, h = det
                    cx = x + (w // 2)
                    cy = y + (h // 2)
                    detection_matrix[i, j] = math.hypot(cx - track['centroid'][0], cy - track['centroid'][1])

            # Perform data association using Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(detection_matrix)

            # Update existing tracks
            for i, track in enumerate(self.tracks):
                if i not in row_ind:
                    track['age'] += 1
                    if track['age'] > 5:  # Remove tracks that are not detected for a certain number of frames
                        self.tracks.remove(track)

            # Create new tracks for unmatched detections
            for j, det in enumerate(detections):
                if j not in col_ind:
                    x, y, w, h = det
                    cx = x + (w // 2)
                    cy = y + (h // 2)

                    roi = frame[y:y + h, x:x + w]
                    is_vehicle = self.detect_vehicle(roi)

                    if is_vehicle:
                        self.tracks.append({
                            'id': object_id,  # Assign the current object ID
                            'bbox': (x, y, w, h),
                            'centroid': (cx, cy),
                            'age': 1,
                        })
                        object_id += 1  # Increment the object ID

            # Update existing tracks with matched detections
            for i, j in zip(row_ind, col_ind):
                if i < len(self.tracks):
                    x, y, w, h = detections[j]
                    cx = x + (w // 2)
                    cy = y + (h // 2)
                    self.tracks[i]['bbox'] = (x, y, w, h)
                    self.tracks[i]['centroid'] = (cx, cy)
                    self.tracks[i]['age'] = 1

        # Apply NMS to each class
        class_ids = [track['id'] for track in self.tracks]
        unique_class_ids = np.unique(class_ids)

        final_objects_bbs_ids = []

        for class_id in unique_class_ids:
            class_tracks = [track for track in self.tracks if track['id'] == class_id]
            class_bboxes = [track['bbox'] for track in class_tracks]

            # Convert bounding boxes to NMS input format (x1, y1, x2, y2)
            bboxes = np.array([[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]] for bbox in class_bboxes])

            # Apply NMS
            indices = cv2.dnn.NMSBoxes(bboxes.tolist(), np.ones(len(bboxes)), score_threshold=0.5, nms_threshold=self.nms_threshold)

            # Append the final bounding boxes with class IDs
            final_objects_bbs_ids.extend([[int(bboxes[idx][0]), int(bboxes[idx][1]), int(bboxes[idx][2] - bboxes[idx][0]),
                                           int(bboxes[idx][3] - bboxes[idx][1]), class_id] for idx in indices])

        return final_objects_bbs_ids
