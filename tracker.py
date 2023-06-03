import math

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


def non_max_suppression(boxes, overlap_threshold):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    picked_indices = []

    for i in range(len(boxes)):
        indices = list(range(i + 1, len(boxes)))
        curr_x1 = x1[i]
        curr_y1 = y1[i]
        curr_x2 = x2[i]
        curr_y2 = y2[i]
        curr_area = areas[i]
        for j in indices:
            comp_x1 = x1[j]
            comp_y1 = y1[j]
            comp_x2 = x2[j]
            comp_y2 = y2[j]
            comp_area = areas[j]

            int_x1 = max(curr_x1, comp_x1)
            int_y1 = max(curr_y1, comp_y1)
            int_x2 = min(curr_x2, comp_x2)
            int_y2 = min(curr_y2, comp_y2)

            int_area = max(0, int_x2 - int_x1 + 1) * max(0, int_y2 - int_y1 + 1)
            iou = int_area / (curr_area + comp_area - int_area)

            if iou < overlap_threshold:
                picked_indices.append(j)

    picked_boxes = [box for i, box in enumerate(boxes) if i not in picked_indices]
    return picked_boxes


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

        if len(self.tracks) == 0:
            for det in detections:
                x, y, w, h = det
                cx = x + (w // 2)
                cy = y + (h // 2)

                roi = frame[y:y + h, x:x + w]
                is_vehicle = self.detect_vehicle(roi)

                if is_vehicle:
                    self.tracks.append({
                        'id': len(self.tracks),
                        'bbox': (x, y, w, h),
                        'centroid': (cx, cy),
                        'age': 1,
                    })
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
                            'id': len(self.tracks),
                            'bbox': (x, y, w, h),
                            'centroid': (cx, cy),
                            'age': 1,
                        })

            # Update existing tracks with matched detections
            for i, j in zip(row_ind, col_ind):
                if i < len(self.tracks):
                    x, y, w, h = detections[j]
                    cx = x + (w // 2)
                    cy = y + (h // 2)
                    self.tracks[i]['bbox'] = (x, y, w, h)
                    self.tracks[i]['centroid'] = (cx, cy)
                    self.tracks[i]['age'] = 1

            # Create new tracks for unmatched detections
            unmatched_indices = set(range(len(detections))).difference(col_ind)
            for j in unmatched_indices:
                x, y, w, h = detections[j]
                cx = x + (w // 2)
                cy = y + (h // 2)

                roi = frame[y:y + h, x:x + w]
                is_vehicle = self.detect_vehicle(roi)

                if is_vehicle:
                    self.tracks.append({
                        'id': len(self.tracks),
                        'bbox': (x, y, w, h),
                        'centroid': (cx, cy),
                        'age': 1,
                    })

        final_objects_bbs_ids = [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]), track['id']]
                                 for track in self.tracks
                                 for bbox in [track['bbox']]]

        return final_objects_bbs_ids