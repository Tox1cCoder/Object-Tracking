import os
from pathlib import Path

import cv2
import joblib
import moviepy.editor as moviepy
import streamlit as st

import helper
import settings
from tracker import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

st.set_page_config(
    page_title="Object Detection And Tracking",
    layout="wide",
    initial_sidebar_state="expanded"
)


def objectTrackingVideoYOLO():
    st.title("Object Tracking in Video With YOLOv8")
    st.subheader("""
    The deep learning approach in this application uses YOLOv8, a state-of-the-art object detection algorithm. The following steps outline the process:

    1. Pre-Trained Model: The YOLOv8 pre-trained model is utilized as a starting point for object detection.
    2. Tracker: Using a Bytetrack or BOTSort tracker with high accuracy and performance
    3. Object Detection: The model is then used for vehicle detection in the input video.

    """)

    st.sidebar.header("Model Config")
    confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    helper.play_video(confidence, model)

def detect_vehicle(model, img):
    img = cv2.resize(img, (64, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog = cv2.HOGDescriptor()
    feature = hog.compute(gray)
    feature = feature.reshape(1, -1)
    pred = model.predict(feature)
    return int(pred[0])

def objectTrackingVideo():
    st.title("Object Tracking in Video (Non-Deep Learning Approach)")
    st.subheader("""
    The non-deep learning approach in this application follows the following steps:

    1. Feature Extraction: The input image is resized to a fixed size and converted to grayscale. The HOG (Histogram of Oriented Gradients) algorithm is then applied to compute the feature vector for the image.
    2. Vehicle Detection: The SVM (Support Vector Machine) model is used to predict whether the extracted features correspond to a vehicle or not.
    3. Object Tracking: The code performs object tracking by comparing the current bounding boxes with previously tracked objects. It uses the Euclidean distance metric to calculate the distance between the centers of the bounding boxes.
    4. Data Association: To associate the detections with existing tracks, the code constructs a detection matrix that represents the similarity between each track and detection pair. The Hungarian algorithm is used to solve the assignment problem and find the best associations.
    5. Non-Maximum Suppression: Before returning the final list of bounding boxes and object IDs, non-maximum suppression is applied to filter out overlapping bounding boxes and keep only the most relevant ones.
    """)
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])

    if uploaded_video != None:
        with open("video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.video(uploaded_video)

        cap = cv2.VideoCapture("video.mp4")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))

        model = joblib.load('./svm.pkl')
        tracker = EuclideanDistTracker()
        object_detector = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

        _, frame = cap.read()

        r = cv2.selectROI(frame)
        cv2.destroyAllWindows()

        with st.spinner('Please wait...'):
            while True:
                ret, frame = cap.read()
                if ret:
                    height, width, _ = frame.shape
                    roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                    mask = object_detector.apply(roi)
                    _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
                    mask = cv2.erode(mask, None, iterations=1)
                    mask = cv2.dilate(mask, None, iterations=2)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    detections = []

                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area > 600:
                            # cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                            x, y, w, h = cv2.boundingRect(cnt)
                            is_vehicle = detect_vehicle(model, roi[y:y + h, x:x + w])
                            if is_vehicle:
                                detections.append([x, y, w, h])

                    boxes_ids = tracker.update(detections, roi)

                    for box_id in boxes_ids:
                        x, y, w, h, id = box_id
                        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

                    out.write(frame)
                else:
                    break

        st.success('Done!')
        out.release()
        cap.release()
        video = open('out.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)


def main():
    new_title = '<p style="font-size: 42px;">Welcome to Object Tracking App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This Streamlit application is designed to track vehicles in a video using two approaches: non-deep learning and deep learning.
    """)
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox("Menu", (
        "About", "Object Tracking In Video (Non-Deep Learning Approach)", "Object Tracking In Video With YOLOv8"))

    if choice == "Object Tracking In Video (Non-Deep Learning Approach)":
        read_me_0.empty()
        read_me.empty()

        objectTrackingVideo()
        try:
            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4', 'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Processed Video")
        except:
            pass

    elif choice == "Object Tracking In Video With YOLOv8":
        read_me_0.empty()
        read_me.empty()
        objectTrackingVideoYOLO()
        try:
            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile("myvideo.mp4")
            st_video = open('myvideo.mp4', 'rb')
            video_bytes = st_video.read()
            st.video(video_bytes)
            st.write("Processed Video")
        except:
            pass

    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
