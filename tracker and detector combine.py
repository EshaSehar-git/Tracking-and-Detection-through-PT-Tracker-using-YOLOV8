import cv2
import numpy as np
from ultralytics import YOLO

class PointTracker:
    def __init__(self):
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)

        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.color = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        self.old_frame = None
        self.old_points = None
        self.mask = None
        self.bbox = None

    def initialize(self, frame, bbox):
        self.old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within the frame
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])

        self.bbox = (x1, y1, x2, y2)

        roi = self.old_frame[y1:y2, x1:x2]
        self.old_points = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
        if self.old_points is not None:
            self.old_points[:, 0, 0] += x1
            self.old_points[:, 0, 1] += y1
        self.mask = np.zeros_like(frame)

    def track(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.old_points is None:
            return frame, self.bbox

        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.old_frame, frame_gray, self.old_points, None, **self.lk_params)

        if new_points is None or status is None:
            self.old_points = None
            return frame, self.bbox

        good_new = new_points[status == 1]
        good_old = self.old_points[status == 1]

        if len(good_new) < 4:
            self.old_points = None
            return frame, self.bbox

        # Draw tracking points
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            self.mask = cv2.line(self.mask, (a, b), (c, d), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)

        img = cv2.add(frame, self.mask)

        self.old_frame = frame_gray.copy()
        self.old_points = good_new.reshape(-1, 1, 2)

        if len(good_new) > 0:
            new_x1 = min(good_new[:, 0])
            new_y1 = min(good_new[:, 1])
            new_x2 = max(good_new[:, 0])
            new_y2 = max(good_new[:, 1])

            # Ensure the new bounding box is within frame dimensions
            new_x1, new_y1 = max(int(new_x1), 0), max(int(new_y1), 0)
            new_x2, new_y2 = min(int(new_x2), frame.shape[1]), min(int(new_y2), frame.shape[0])

            self.bbox = (new_x1, new_y1, new_x2, new_y2)

        return img, self.bbox

# Load the trained YOLOv8 model
model = YOLO('C:\\Users\\s\\Documents\\dataset\\results\\stappler.pt')

# Initialize the video capture with the video file path
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Error: Could not open video file ")

tracker_initialized = False
tracker = PointTracker()
roi = None
class_name = None
conf = None

while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    if not tracker_initialized:
        # Perform inference
        results = model(frame)

        # Extract bounding boxes and confidence scores
        detections = results[0].boxes

        # Check if any object is detected
        if len(detections) == 0:
            cv2.putText(display_frame, 'Object not found', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Use the first detected object for tracking
            detection = detections[0]
            x1, y1, x2, y2 = detection.xyxy[0].int().tolist()
            conf = detection.conf[0]
            cls = detection.cls[0]

            if conf > 0.8:  # Only consider detections with confidence > 0.5
                roi = (x1, y1, x2, y2)

                # Get class name (Assuming you have a way to map class index to class name)
                class_name = model.names[int(cls)]

                # Initialize the tracker with the selected ROI
                tracker.initialize(frame, roi)
                tracker_initialized = True
    else:
        # Update the tracker
        display_frame, bbox = tracker.track(frame)

        # Draw bounding box and display confidence and class
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2, 1)
            cv2.putText(display_frame, f'{class_name}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(display_frame, f'X: {x1}, Y: {y1}', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display_frame, f'Width: {x2 - x1}, Height: {y2 - y1}', (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            tracker_initialized = False

    # Display result
    cv2.imshow("Tracking", display_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
