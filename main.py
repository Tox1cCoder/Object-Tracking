import cv2
import moviepy.editor as moviepy
import numpy as np
import streamlit as st
from PIL import Image
from tracker import *


# Steps:
# Preprocess the input images or video to remove noise and enhance the features you want to track.
# Extract features using a feature extraction technique such as HOG, SIFT, or SURF.
# Detect and classify pedestrians and vehicles using a classification algorithm such as SVM, decision trees, or nearest neighbor.
# Track the detected objects using an object tracking algorithm such as KLT or Mean Shift.
# Display the tracking results in real-time or save them to a file.


def object_detection_image():
    st.title('Object Detection for Images')
    st.subheader("""
    This object detection project takes in an image and outputs the image with bounding boxes created around the objects in the image.
    """)
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file != None:
        img1 = Image.open(file)
        img2 = np.array(img1)

        st.image(img1, caption="Uploaded Image")
        my_bar = st.progress(0)
        confThreshold = st.slider('Confidence', 0, 100, 50)
        nmsThreshold = st.slider('Threshold', 0, 100, 20)

        st.image(img2, caption='Proccesed Image.')

        cv2.waitKey(0)

        cv2.destroyAllWindows()
        my_bar.progress(100)


def objectTrackingVideo():
    st.title("Object Tracking in Video")
    st.subheader("""
    This object tracking project takes in a video and outputs the video with bounding boxes created around the objects in the video.
    """)
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:
        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read())

        st_video = open(vid, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        cap = cv2.VideoCapture(vid)
        _, image = cap.read()
        h, w = image.shape[:2]

        tracker = EuclideanDistTracker()
        object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

        fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))

        while True:
            ret, frame = cap.read()
            if ret == True:
                height, width, _ = frame.shape
                #roi = frame[340: 720, 500: 800]
                roi = frame[:,:]

                mask = object_detector.apply(roi)
                _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                detections = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 100:
                        #cv2.drawContours(roi, [cnt], -1, (0, 255, 0), 2)
                        x, y, w, h = cv2.boundingRect(cnt)
                        detections.append([x, y, w, h])

                boxes_ids = tracker.update(detections)
                for box_id in boxes_ids:
                    x, y, w, h, id = box_id
                    #cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

                out.write(frame)
            else:
                break

        cap.release()
        out.release()

        st.video(open("detected_video.mp4", 'rb').read())
        st.write("Processed Video")


def main():
    new_title = '<p style="font-size: 42px;">Welcome to Object Tracking App!</p>'
    read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

    read_me = st.markdown("""
    This project was built using Streamlit and OpenCV.
    """)
    st.sidebar.title("Select Activity")
    choice = st.sidebar.selectbox("Menu", ("About", "Object Tracking In Video"))

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

    elif choice == "About":
        print()


if __name__ == '__main__':
    main()
