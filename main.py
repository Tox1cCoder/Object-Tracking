import moviepy.editor as moviepy
import os
from pathlib import Path

import moviepy.editor as moviepy
import streamlit as st

import helper
import settings
from tracker import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(
    page_title="Object Detection And Tracking",
    layout="wide",
    initial_sidebar_state="expanded"
)


def objectTrackingVideoYOLO():
    st.title("Object Tracking in Video With YOLOv8")
    st.subheader("""
        This object tracking project takes in a video and outputs the video with bounding boxes created around the objects in the video.
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


def objectTrackingVideo():
    st.title("Object Tracking in Video")
    st.subheader("""
    This object tracking project takes in a video and outputs the video with contours created around the objects in the video.
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

        tracker = EuclideanDistTracker()
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

        _, frame = cap.read()
        r = cv2.selectROI(frame)
        cv2.destroyAllWindows()

        with st.spinner('Wait for it...'):
            while True:
                ret, frame = cap.read()
                if ret:
                    height, width, _ = frame.shape
                    roi = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

                    mask = object_detector.apply(roi)
                    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)  # Slide ham findContours
                    detections = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)  # Slide ham contourArea
                        if area > 100:
                            cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)  # Slide ham drawContours
                            x, y, w, h = cv2.boundingRect(cnt)
                            detections.append([x, y, w, h])

                    boxes_ids = tracker.update(detections)
                    for box_id in boxes_ids:
                        x, y, w, h, id = box_id
                        cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
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
    This project was built using Streamlit and OpenCV.
    """)
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox("Menu", ("About", "Object Tracking In Video", "Object Tracking In Video With YOLOv8"))

    if choice == "Object Tracking In Video":
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
