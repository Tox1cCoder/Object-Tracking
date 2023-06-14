from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tracker2 import *

# Load a model
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture("videos\cars.mp4")

max_width = 800
max_height = 600
# init tracker 
tracker = Tracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame.shape[1] > max_width or frame.shape[0] > max_height:
        frame = cv2.resize(frame, (max_width, max_height))

    results = model.predict(frame, classes=[2], conf = 0.25)

    boxes = results[0].boxes.xyxy.numpy().astype(int)
    nms_results = non_max_suppression2(boxes, iou_threshold=0.5)

    # xử lý id 
    boxes_ids = tracker.track_boxes(nms_results)
    # 
    img = results[0].orig_img

    for i in range(len(boxes_ids)):
        x1, y1, x2, y2 = boxes_ids[i].positions
        id = boxes_ids[i].id
        cv2.putText(img, str(id), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cx = int((x1 + x2)/2)
        cy = int((y1 + y2)/2)
        # cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
 
    cv2.imshow("img", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()