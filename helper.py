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
    #   0: person
    #   1: bicycle
    #   2: car
    #   3: motorcycle
    #   4: airplane
    #   5: bus
    #   6: train
    #   7: truck

    image = cv2.resize(image, (720, int(720 * (9 / 16))))

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker, stream=True, classes=[0,2,7])
    else:
        res = model.predict(image, conf=conf, stream=True, classes=[0,2,7])

    res = list(res)
    res_plotted = res[0].plot()
    st_frame.image(res_plotted, caption='Detected Video', channels="BGR", use_column_width=True)
    return res_plotted


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
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                width = 720
                height = 405
                fourcc = cv2.VideoWriter_fourcc('h', '2', '6', '4')
                out = cv2.VideoWriter('out.mp4', fourcc, fps, (width, height))
                st_frame = st.empty()
                with st.spinner('Please wait...'):
                    stop_flag = False
                    stop_button = st.button('Stop')
                    while vid_cap.isOpened() and not stop_flag:
                        success, image = vid_cap.read()
                        if success:
                            image =_display_detected_frames(conf, model, st_frame, image, is_display_tracker, tracker)
                            out.write(image)   
                        else:
                            break
                        if stop_button:
                            stop_flag = True
                st.success('Done!')
                out.release()
                vid_cap.release()
                video = open('out.mp4', 'rb')
                video_bytes = video.read()
                st_frame.video(video_bytes)
            except Exception as e:
                st.sidebar.error("Error: " + str(e))
            
