import numpy as np
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class Sort:
    def __init__(self):
        # Store the active tracks
        self.tracks = []
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        objects_bbs_ids = []

        # If there are no tracks, create new tracks for all detections
        if len(self.tracks) == 0:
            for rect in objects_rect:
                x, y, w, h = rect
                cx = x + (w // 2)
                cy = y + (h // 2)

                track = {'id': self.id_count, 'bbox': (x, y, w, h), 'kalman_filter': self.create_kalman_filter(cx, cy)}
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.tracks.append(track)
        else:
            # Predict the next state of each track using the Kalman filter
            for track in self.tracks:
                track['kalman_filter'].predict()

            # Compute the distance between each track prediction and the detections
            track_predictions = [track['kalman_filter'].x for track in self.tracks]
            detection_bboxes = [(*rect, 0, 0) for rect in objects_rect]
            cost_matrix = self.compute_cost_matrix(track_predictions, detection_bboxes)

            # Perform data association using the Hungarian algorithm
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)

            # Update the existing tracks with the assigned detections
            for track_index, detection_index in zip(track_indices, detection_indices):
                track = self.tracks[track_index]
                bbox = objects_rect[detection_index]
                x, y, w, h = bbox
                cx = x + (w // 2)
                cy = y + (h // 2)
                track['bbox'] = bbox
                track['kalman_filter'].update(np.array([[cx], [cy]]))
                objects_bbs_ids.append([x, y, w, h, track['id']])

            # Create new tracks for the unassigned detections
            unassigned_detections = set(range(len(objects_rect))) - set(detection_indices)
            for detection_index in unassigned_detections:
                x, y, w, h = objects_rect[detection_index]
                cx = x + (w // 2)
                cy = y + (h // 2)
                track = {'id': self.id_count, 'bbox': (x, y, w, h), 'kalman_filter': self.create_kalman_filter(cx, cy)}
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.tracks.append(track)

        # Clean up the tracks by removing inactive tracks
        self.tracks = [track for track in self.tracks if track['kalman_filter'].x[0] is not None]

        return objects_bbs_ids

    def create_kalman_filter(self, cx, cy):
        kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
        kalman_filter.x = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        kalman_filter.F = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)
        kalman_filter.H = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0]], dtype=np.float32)
        kalman_filter.P *= 10  # Initial uncertainty
        kalman_filter.R = np.array([[1, 0],
                                    [0, 1]], dtype=np.float32)  # Measurement noise covariance
        kalman_filter.Q = Q_discrete_white_noise(dim=2, dt=1.0, var=0.01)  # Process noise covariance

        return kalman_filter

    def compute_cost_matrix(self, track_predictions, detection_bboxes):
        num_tracks = len(track_predictions)
        num_detections = len(detection_bboxes)
        cost_matrix = np.zeros((num_tracks, num_detections))

        for i in range(num_tracks):
            for j in range(num_detections):
                track_prediction = track_predictions[i]
                detection_bbox = detection_bboxes[j]
                cost_matrix[i, j] = self.compute_distance(track_prediction, detection_bbox)

        return cost_matrix

    @staticmethod
    def compute_distance(track_prediction, detection_bbox):
        track_x, track_y = track_prediction[0, 0], track_prediction[1, 0]
        detection_x, detection_y = detection_bbox[0] + detection_bbox[2] / 2, detection_bbox[1] + detection_bbox[3] / 2
        distance = np.sqrt((track_x - detection_x) ** 2 + (track_y - detection_y) ** 2)

        return distance
