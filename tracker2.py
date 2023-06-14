from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.kalman import KalmanFilter
from scipy.spatial import distance
import numpy as np

class Box:
    def __init__(self, positions, id=-1):
        self.positions = positions
        self.id = id
        self.time = 0
        self.missedTime = 0
        self.kf = None
    
    def increase_time(self):
        self.time += 1

    def update_id(self, new_id):
        self.id = new_id

    def __str__(self):
        return f"""Box(id={self.id}, 
                positions={self.positions}, 
                time={self.time}, 
                missedTime={self.missedTime}, 
                ukf={self.kf})"""

    

class Tracker:
    id_counter = 0

    def __init__(self):
        self.currBoxes = []
        self.preBoxes = []
        self.missBoxes = []

    def calculate_distance(self, pos1, pos2):
        center1 = self.get_center(pos1)
        center2 = self.get_center(pos2)
        return distance.euclidean(center1, center2)

    def get_center(self, positions):
        x1, y1, x2, y2 = positions
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return [center_x, center_y]

    def update_box_ids(self):
        if not self.preBoxes:
            for i in range(len(self.currBoxes)):
                self.currBoxes[i].update_id(Tracker.id_counter)
                positions = self.get_center( self.currBoxes[i].positions)
                # self.currBoxes[i].kf = self.initialize_kalman_filter(positions)
                # self.currBoxes[i].kf.update(self.get_center(self.currBoxes[i].positions))
                Tracker.id_counter += 1
            return

        for preBox in self.preBoxes:
            matchingIdx = None
            min_distance = float('inf')
            # dùng khoảng cách euclid ban đầu 
            # if preBox.time <= 6:
            for i, currBox in enumerate(self.currBoxes):
                if currBox.id == -1:
                    distance = self.calculate_distance(currBox.positions, preBox.positions)
                    if distance < min_distance and distance < 15:
                        min_distance = distance
                        matchingIdx = i 
                        
            # elif preBox.time > 6:
                #use UKF
            # print("THIS ID PREBOX================>",preBox)
            # predicted_position = preBox.kf.predict()  # Dự đoán vị trí
            # print("RESULT OF KF.PREDICT =======>: ", predicted_position)
            # for i, currBox in enumerate(self.currBoxes):
            #     curr_position = currBox.positions  # Vị trí hiện tại của currBox
            #     distance = self.calculate_distance(predicted_position, curr_position)
            #     if distance < min_distance and distance < 15:
            #         min_distance = distance
            #         matchingIdx = i
            
            if matchingIdx is not None:
                self.currBoxes[matchingIdx].update_id(preBox.id)
                self.currBoxes[matchingIdx].time = preBox.time + 1
                # self.currBoxes[matchingIdx].kf = preBox.kf
                # self.currBoxes[matchingIdx].kf.update(self.get_center(self.currBoxes[matchingIdx].positions))



        # tạo id cho 1 box được xác định là box mới
        for i in range(len(self.currBoxes)):
            if self.currBoxes[i].id == -1:
                self.currBoxes[i].update_id(Tracker.id_counter)
                Tracker.id_counter += 1
                positions = self.get_center( self.currBoxes[i].positions)
                # self.currBoxes[i].kf = self.initialize_kalman_filter(positions)
                # self.currBoxes[i].kf.update(self.get_center(self.currBoxes[i].positions))



    def track_boxes(self, positions_list):
        self.preBoxes = self.currBoxes.copy()
        self.currBoxes = []

        for positions in positions_list:
            box = Box(positions)
            self.currBoxes.append(box)

        self.update_box_ids()
        self.update_missed_boxes()
        self.remove_disappeared_boxes()

        for box in self.currBoxes:
            box.increase_time()

        return self.currBoxes
        

    def update_missed_boxes(self):
        for preBox in self.preBoxes:
            if preBox.id != -1 and preBox not in self.currBoxes:
                preBox.missedTime = 1
                self.missBoxes.append(preBox)
        

    def remove_disappeared_boxes(self):
        self.currBoxes = [box for box in self.currBoxes if box.id != -1]

        for missBox in self.missBoxes:
            if missBox.missedTime >= 6:
                self.missBoxes.remove(missBox)

    def initialize_kalman_filter(self, positions):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 0.1
        kf.F = np.array([[1, dt, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, dt],
                        [0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]])

        kf.Q = np.eye(4) * 0.01

        kf.R = np.eye(2) * 1

        kf.x = np.array([positions[0], positions[1], 0, 0])
        kf.P = np.eye(4) * 10

        return kf

    def transition_function(self, x, dt):
        # Hàm chuyển đổi trạng thái
        F = np.array([[1, dt, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, dt],
                    [0, 0, 0, 1]])
        return np.dot(F, x)

    def measurement_function(self, x):
        # Hàm chuyển đổi đầu ra
        return np.array([x[0], x[2]])



    
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    area_box1 = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_box2 = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

    xA = max(x1, x1_)
    yA = max(y1, y1_)
    xB = min(x2, x2_)
    yB = min(y2, y2_)

    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    iou = inter_area / float(area_box1 + area_box2 - inter_area)
    return iou


def non_max_suppression2(boxes, iou_threshold):
    if iou_threshold is None:
        iou_threshold = 0.5

    if len(boxes) == 0:
        return []

    sorted_boxes = sorted(boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True)
    suppressed_boxes = [False] * len(sorted_boxes)

    for i in range(len(sorted_boxes)):
        if suppressed_boxes[i]:
            continue

        current_box = sorted_boxes[i]

        for j in range(i + 1, len(sorted_boxes)):
            if suppressed_boxes[j]:
                continue

            box = sorted_boxes[j]
            iou = calculate_iou(current_box, box)

            if iou >= iou_threshold:
                suppressed_boxes[j] = True

    results = [box for i, box in enumerate(sorted_boxes) if not suppressed_boxes[i]]
    return results

    


    