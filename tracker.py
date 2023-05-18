import math
import cv2
import numpy as np


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Store the Kalman filters for object tracking
        self.kalman_filters = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h = rect
            cx = x + (w // 2)
            cy = y + (h // 2)

            # Find out if that object was detected already
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

        # Clean the dictionary by removing IDs that are not used anymore
        self.center_points = {
            object_id: center
            for object_id, center in self.center_points.items()
            if object_id in [obj_bb_id[4] for obj_bb_id in objects_bbs_ids]
        }

        # # Perform motion prediction using optical flow
        # prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        #
        # for obj_bb_id in objects_bbs_ids:
        #     if len(obj_bb_id) == 5:
        #         x, y, w, h, object_id = obj_bb_id
        #     elif len(obj_bb_id) == 7:
        #         x, y, w, h, object_id, predicted_cx, predicted_cy = obj_bb_id
        #     # Predict the new object position using optical flow motion vectors
        #     dx = int(np.mean(flow[y:y + h, x:x + w, 0]))
        #     dy = int(np.mean(flow[y:y + h, x:x + w, 1]))
        #     predicted_cx = cx + dx
        #     predicted_cy = cy + dy
        #
        #     # Update the Kalman filter with the predicted position
        #     kalman_filter = self.kalman_filters[object_id]
        #     kalman_filter.correct(np.array([[predicted_cx], [predicted_cy]], np.float32))
        #     predicted_state = kalman_filter.predict()
        #
        #     objects_bbs_ids.append([x, y, w, h, object_id, predicted_state[0, 0], predicted_state[1, 0]])
        # Update the Kalman filter for existing objects using optical flow


        return objects_bbs_ids
