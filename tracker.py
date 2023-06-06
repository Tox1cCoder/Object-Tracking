import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

object_id = 0

class EuclideanDistTracker:
    def __init__(self, svm_model):
        self.tracks = []
        self.svm_model = svm_model

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
                    if track['age'] > 10:  # Remove tracks that are not detected for a certain number of frames
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

        final_objects_bbs_ids = [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track['id']]
                                 for track in self.tracks
                                 for bbox in [track['bbox']]]

        return final_objects_bbs_ids
