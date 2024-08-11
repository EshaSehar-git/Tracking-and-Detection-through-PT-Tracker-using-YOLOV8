# Tracking-and-Detection-through-PT-Tracker-using-YOLOV8
This project combines the capabilities of YOLOv8 (You Only Look Once version 8) for object detection with PT (Point Tracker) tracking algorithms to achieve real-time detection and tracking of objects in video streams. The objective is to maintain the consistency of bounding boxes and ensure accurate tracking of detected objects as they move across frames.Libraries that are used in this project are OPENCV , NUMPY , Ultralytics (YOLO).

## PT Tracker
PT Tracker (Point Tracker) is a type of tracking algorithm that focuses on maintaining the position of objects across video frames using key points. Unlike traditional tracking methods that rely on bounding boxes, point trackers track specific feature points within an object, providing a more granular approach to tracking.

## OPENCV
OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library designed to provide a common infrastructure for computer vision applications. It is widely used in industry and academia for various tasks, including image processing, video analysis, object detection, and more.
1. OpenCV provides a vast array of functions for image processing tasks such as filtering, edge detection, image transformations, and color space conversions.
2. OpenCV includes tools for video capture, processing, and analysis, allowing developers to work with video streams in real-time.
3. OpenCV offers various object detection algorithms, including pre-trained models for face detection, pedestrian detection, and other specific tasks. The library supports integration with deep learning frameworks like TensorFlow, PyTorch, and Caffe for more advanced object detection and recognition.

## Numpy
The NumPy library plays a crucial role in tracking and detection tasks within computer vision by providing efficient array operations and mathematical tools that are essential for processing images, video frames, and the data associated with these tasks.
**Image and Video Frame Representation:** Images and video frames are typically represented as multi-dimensional arrays (e.g., a 2D array for grayscale images, a 3D array for color images). NumPy arrays (ndarrays) are used to store and manipulate these pixel values efficiently.
**Bounding Boxes and Keypoints:** NumPy arrays are used to represent bounding boxes (as arrays of coordinates) and keypoints (as arrays of points), which are fundamental in object detection and tracking.

## YOLOV8
YOLOv8 (You Only Look Once version 8) is a state-of-the-art model known for its high-speed and accurate object detection capabilities.
It provides real-time detection of multiple objects within images or video frames, outputting bounding boxes, class labels, and confidence scores for each detected object.
**Real-Time Object Detection**
**Improved Architecture**
**High Speed and Efficiency**
**Versatility and Flexibility**
