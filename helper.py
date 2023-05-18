import cv2
import streamlit as st
from ultralytics import YOLO


def load_model(model_path):
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))  # Slide phan biet tracker
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)


def play_video(conf, model):
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])

    if uploaded_video != None:
        is_display_tracker, tracker = display_tracker_options()

        with open("video.mp4", "wb") as video_file:
            video_file.write(uploaded_video.getbuffer())
        st.video(uploaded_video)

        if st.sidebar.button('Detect Video Objects'):
            try:
                vid_cap = cv2.VideoCapture("video.mp4")
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        _display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                    else:
                        vid_cap.release()
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
