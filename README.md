# Streamlit Object Tracking Application
This Streamlit application is designed to track vehicles in a video using two approaches: non-deep learning and deep learning.

## Non-Deep Learning Approach
The non-deep learning approach in this application follows the following steps:

1. Feature Extraction: The HOG algorithm is used to extract features from the video frames.
2. Classification: A trained SVM model is used to classify the extracted features into vehicle and non-vehicle categories.
3. Tracking: The Euclidean distance is employed to track the detected vehicles across frames.
4. Object Localization: The Kalman filter is applied to estimate and localize the tracked vehicles in the video.

## Deep Learning Approach
The deep learning approach in this application uses YOLOv8, a state-of-the-art object detection algorithm. The following steps outline the process:

1. Pre-Trained Model: The YOLOv8 pre-trained model is utilized as a starting point for object detection.
2. Custom Dataset: A custom dataset is created, consisting of annotated images or videos specifically focused on vehicle tracking.
3. Retraining: The pre-trained YOLOv8 model is fine-tuned and retrained on the custom dataset to improve its accuracy and performance.
4. Object Detection: The retrained YOLOv8 model is then used for vehicle detection in the input video.

## Usage
To run the Object Tracking Application, follow these steps:

1. Install the necessary dependencies mentioned in the requirements.txt file.
2. Run the application by executing the command streamlit run main.py in the terminal.
