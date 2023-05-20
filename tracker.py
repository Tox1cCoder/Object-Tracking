import math
import cv2
import numpy as np

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
                roi = frame[y:y+h, x:x+w]
                is_vehicle = self.detect_vehicle(roi)

                if is_vehicle:
                    self.center_points[self.id_count] = (cx, cy)

                    kalman_filter = cv2.KalmanFilter(4, 2)
                    kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                                [0, 1, 0, 0]], np.float32)
                    kalman_filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                               [0, 1, 0, 1],
                                                               [0, 0, 1, 0],
                                                               [0, 0, 0, 1]], np.float32)
                    kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                              [0, 1, 0, 0],
                                                              [0, 0, 1, 0],
                                                              [0, 0, 0, 1]], np.float32) * 0.03

                    # Initialize the Kalman filter state with the current center point
                    kalman_filter.statePre = np.array([[cx], [cy], [0], [0]], np.float32)

                    self.kalman_filters[self.id_count] = kalman_filter

                    objects_bbs_ids.append([x, y, w, h, self.id_count])
                    self.id_count += 1

        self.center_points = {
            object_id: center
            for object_id, center in self.center_points.items()
            if object_id in [obj_bb_id[4] for obj_bb_id in objects_bbs_ids]
        }
        return objects_bbs_ids
