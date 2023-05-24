import math

import cv2
import numpy as np


def non_max_suppression(boxes, overlap_threshold):
    if len(boxes) == 0:
        return []

    # Convert the bounding boxes to NumPy array for easier manipulation
    boxes = np.array(boxes)

    # Extract the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Calculate the areas of the bounding boxes
    areas = boxes[:, 2] * boxes[:, 3]

    # Initialize a list to store the picked indices
    picked_indices = []

    # Calculate the IOU (Intersection over Union) between each pair of bounding boxes
    for i in range(len(boxes)):
        # Find indices of boxes to compare with the current box
        indices = list(range(i + 1, len(boxes)))

        # Get coordinates of the current box
        curr_x1 = x1[i]
        curr_y1 = y1[i]
        curr_x2 = x2[i]
        curr_y2 = y2[i]

        # Calculate the area of the current box
        curr_area = areas[i]

        # Iterate over the remaining boxes
        for j in indices:
            # Get coordinates of the comparison box
            comp_x1 = x1[j]
            comp_y1 = y1[j]
            comp_x2 = x2[j]
            comp_y2 = y2[j]

            # Calculate the area of the comparison box
            comp_area = areas[j]

            # Calculate the intersection coordinates
            int_x1 = max(curr_x1, comp_x1)
            int_y1 = max(curr_y1, comp_y1)
            int_x2 = min(curr_x2, comp_x2)
            int_y2 = min(curr_y2, comp_y2)

            # Calculate the intersection area
            int_area = max(0, int_x2 - int_x1 + 1) * max(0, int_y2 - int_y1 + 1)

            # Calculate the IOU
            iou = int_area / (curr_area + comp_area - int_area)

            # If the IOU is below the threshold, add the comparison box index to the picked indices
            if iou < overlap_threshold:
                picked_indices.append(j)

    # Filter out the boxes that have overlapping indices
    picked_boxes = [box for i, box in enumerate(boxes) if i not in picked_indices]

    return picked_boxes

class EuclideanDistTracker:
    def __init__(self, svm_model):
        self.center_points = {}
        self.kalman_filters = {}
        self.id_count = 0
        self.svm_model = svm_model

    def detect_vehicle(self, img):
        img = cv2.resize(img, (64, 128))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        features = hog.compute(gray)
        threshold = 0.7
        self.svm_model.decision_function = lambda X: np.where(self.svm_model.decision_function(X) >= threshold, 1, -1)
        return self.svm_model.predict([features])[0]

    def update(self, objects_rect, frame):
        objects_bbs_ids = []

        # Apply Non-Maximum Suppression
        objects_rect = non_max_suppression(objects_rect, overlap_threshold=0.2)

        for rect in objects_rect:
            x, y, w, h = rect
            cx = x + (w // 2)
            cy = y + (h // 2)

            same_object_detected = False
            for object_id, center in self.center_points.items():
                prev_cx, prev_cy = center
                dist = math.hypot(cx - prev_cx, cy - prev_cy)

                if dist < 50:
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, object_id])
                    same_object_detected = True
                    break

            if not same_object_detected:
                roi = frame[y:y + h, x:x + w]
                is_vehicle = self.detect_vehicle(roi)

                if is_vehicle:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.id_count])
                    self.id_count += 1

        self.center_points = {
            object_id: center
            for object_id, center in self.center_points.items()
            if object_id in [obj_bb_id[4] for obj_bb_id in objects_bbs_ids]
        }
        return objects_bbs_ids
