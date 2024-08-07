import cv2
import numpy as np

class PointTracker:
    def __init__(self):
        # Parameters for ShiTomasi corner detection
        self.feature_params = dict(maxCorners=100,
                                   qualityLevel=0.3,
                                   minDistance=7,
                                   blockSize=7)
        
        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # Random colors for visualizing points
        self.color = np.random.randint(0, 255, (100, 3), dtype=np.uint8)
        
        # Placeholder for old frame and points
        self.old_frame = None
        self.old_points = None
        self.mask = None

    def initialize(self, frame, bbox):
        self.old_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = bbox
        roi = self.old_frame[y:y+h, x:x+w]
        self.old_points = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
        if self.old_points is not None:
            self.old_points[:, 0, 0] += x
            self.old_points[:, 0, 1] += y
        self.mask = np.zeros_like(frame)

    def track(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.old_points is None:
            return frame

        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.old_frame, frame_gray, self.old_points, None, **self.lk_params)
        
        # Select good points
        good_new = new_points[status == 1]
        good_old = self.old_points[status == 1]
        
        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            self.mask = cv2.line(self.mask, (a, b), (c, d), self.color[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, self.color[i].tolist(), -1)
        
        img = cv2.add(frame, self.mask)
        
        # Update the previous frame and previous points
        self.old_frame = frame_gray.copy()
        self.old_points = good_new.reshape(-1, 1, 2)
        
        return img

def main():
    cap = cv2.VideoCapture(0)
    tracker = PointTracker()
    initialized = False
    bbox = None
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
    
    while True:
        ret, frame = cap.read()
        #frame=cv2.flip(frame,-1)
        if not ret:
            break
        
        if not initialized:
            bbox = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            tracker.initialize(frame, bbox)
            initialized = True
            cv2.destroyWindow("Frame")
        
        frame_with_tracks = tracker.track(frame)
        
        # Write the frame into the file 'output.avi'
        out.write(frame_with_tracks)
        
        cv2.imshow('Tracking', frame_with_tracks)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
