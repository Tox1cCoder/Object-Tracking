# Object Tracking Application
This Streamlit application is designed to track vehicles in a video using two approaches: non-deep learning and deep learning.

## Non-Deep Learning Approach
The non-deep learning approach in this application follows the following steps:

1. Feature Extraction: The input image is resized to a fixed size and converted to grayscale. The HOG (Histogram of Oriented Gradients) algorithm is then applied to compute the feature vector for the image.
2. Vehicle Detection: The SVM (Support Vector Machine) model is used to predict whether the extracted features correspond to a vehicle or not.
3. Object Tracking: The code performs object tracking by comparing the current bounding boxes with previously tracked objects. It uses the Euclidean distance metric to calculate the distance between the centers of the bounding boxes.
4. Data Association: To associate the detections with existing tracks, the code constructs a detection matrix that represents the similarity between each track and detection pair. The Hungarian algorithm is used to solve the assignment problem and find the best associations.
5. Non-Maximum Suppression: Before returning the final list of bounding boxes and object IDs, non-maximum suppression is applied to filter out overlapping bounding boxes and keep only the most relevant ones.

## Deep Learning Approach
The deep learning approach in this application uses YOLOv8, a state-of-the-art object detection algorithm. The following steps outline the process:

1. Pre-Trained Model: The YOLOv8 pre-trained model is utilized as a starting point for object detection.
2. Tracker: Using a Bytetrack or BOTSort tracker with high accuracy and performance
3. Object Detection: The model is then used for vehicle detection in the input video.

## Usage
To run the Object Tracking Application, follow these steps:

1. Install the necessary dependencies mentioned in the requirements.txt file.
2. Run the application by executing the command `streamlit run main.py` in the terminal.
