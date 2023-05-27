import math

import cv2
import numpy as np


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
    picked_boxes = boxes
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
        features = features.flatten().reshape(1, -1)
        pred = self.svm_model.predict(features)
        return pred

    def update(self, objects_rect, frame):
        objects_bbs_ids = []
        min_overlap_threshold = 0.3

        for rect in objects_rect:
            x, y, w, h = rect
            cx = x + (w // 2)
            cy = y + (h // 2)

            same_object_detected = False
            for object_id, center in self.center_points.items():
                prev_cx, prev_cy = center
                dist = math.hypot(cx - prev_cx, cy - prev_cy)

                if dist < 200:
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

        objects_bbs_ids_per_object = {}
        for obj_bb_id in objects_bbs_ids:
            object_id = obj_bb_id[4]
            if object_id in objects_bbs_ids_per_object:
                objects_bbs_ids_per_object[object_id].append(obj_bb_id)
            else:
                objects_bbs_ids_per_object[object_id] = [obj_bb_id]

        final_objects_bbs_ids = []
        for object_id, bbs_ids in objects_bbs_ids_per_object.items():
            object_bbs = [bb_id[:4] for bb_id in bbs_ids]
            object_bbs = non_max_suppression(object_bbs, overlap_threshold=min_overlap_threshold)
            final_objects_bbs_ids.extend([[*bb, object_id] for bb in object_bbs])

        self.center_points = {
            object_id: center
            for object_id, center in self.center_points.items()
            if object_id in [obj_bb_id[4] for obj_bb_id in final_objects_bbs_ids]
        }

        return final_objects_bbs_ids
