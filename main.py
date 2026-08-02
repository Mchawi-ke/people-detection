# ============================================================
# Railway Platform Safety Monitoring System
# Advanced YOLO + Line Geometry + Risk Analysis
# Part 1/3
# ============================================================

import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict, deque


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = (
    "/kaggle/input/cctv-video-footage/"
    "BB_cff64429-bafc-4258-bdbe-6bba7270af6c_preview.mp4"
)

OUTPUT_PATH = "/kaggle/working/railway_safety_output.mp4"


LINE_MODEL_PATH = (
    "/kaggle/input/subway-yellow-safety-line-detection/"
    "pytorch/default/1/Line_detection_weights (best).pt"
)

PERSON_MODEL_PATH = (
    "/kaggle/input/people-detection-model/"
    "pytorch/default/1/People_detection_weights(best).pt"
)


# Risk thresholds in pixels
SAFE_DISTANCE = 120
WARNING_DISTANCE = 70
DANGER_DISTANCE = 30


# Number of frames required before alarm
DANGER_CONFIRM_FRAMES = 5


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading models...")

line_model = YOLO(LINE_MODEL_PATH)

person_model = YOLO(PERSON_MODEL_PATH)


print("Models loaded successfully")


# ============================================================
# SAFETY LINE DETECTOR
# ============================================================

class SafetyLineDetector:

    def __init__(self):

        self.previous_line = None


    def extract_yellow_pixels(self, frame, bbox):

        """
        Extract yellow safety line pixels
        inside platform bounding box
        """

        if bbox is None:
            return None


        x1, y1, x2, y2 = bbox


        roi = frame[y1:y2, x1:x2]


        if roi.size == 0:
            return None


        hsv = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV
        )


        lower_yellow = np.array(
            [15, 50, 80]
        )

        upper_yellow = np.array(
            [45, 255, 255]
        )


        mask = cv2.inRange(
            hsv,
            lower_yellow,
            upper_yellow
        )


        points = cv2.findNonZero(mask)


        if points is None:
            return None


        points = points.reshape(-1,2)


        # Convert ROI coordinates to frame coordinates

        points[:,0] += x1
        points[:,1] += y1


        return points



    def fit_line(self, points):

        """
        Fit a single mathematical line
        using all yellow pixels
        """

        if points is None:
            return self.previous_line


        if len(points) < 20:
            return self.previous_line


        vx,vy,x,y = cv2.fitLine(
            points.astype(np.float32),
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )


        line = (
            float(vx),
            float(vy),
            float(x),
            float(y)
        )


        self.previous_line = line


        return line



    def draw_line(self, frame, line):

        """
        Draw fitted safety line
        """

        if line is None:
            return


        vx,vy,x,y = line


        length = 2000


        x1 = int(x - vx*length)
        y1 = int(y - vy*length)

        x2 = int(x + vx*length)
        y2 = int(y + vy*length)


        cv2.line(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            3
        )



# ============================================================
# GEOMETRY FUNCTIONS
# ============================================================

def point_line_distance(point, line):

    """
    Calculate perpendicular distance
    from point to fitted line
    """

    if line is None:
        return 9999


    vx,vy,x0,y0 = line


    px,py = point


    numerator = abs(
        vy*(px-x0)
        -
        vx*(py-y0)
    )


    denominator = np.sqrt(
        vx*vx + vy*vy
    )


    return numerator / denominator



def point_side(point,line):

    """
    Determines which side of line
    a point belongs to
    """

    if line is None:
        return 0


    vx,vy,x0,y0=line


    px,py=point


    value = (
        vx*(py-y0)
        -
        vy*(px-x0)
    )


    if value > 0:
        return 1

    else:
        return -1



# ============================================================
# FOOT POSITION
# ============================================================

def get_foot_position(box):

    x1,y1,x2,y2 = box


    return (
        int((x1+x2)/2),
        int(y2)
    )



# ============================================================
# RISK CLASSIFICATION
# ============================================================

def calculate_risk(distance):


    if distance > SAFE_DISTANCE:

        return "SAFE", (0,255,0)



    elif distance > WARNING_DISTANCE:

        return "WARNING",(0,255,255)



    elif distance > DANGER_DISTANCE:

        return "HIGH RISK",(0,165,255)



    else:

        return "DANGER",(0,0,255)



# ============================================================
# INITIALIZE
# ============================================================

safety_detector = SafetyLineDetector()


person_history = defaultdict(
    lambda: deque(maxlen=20)
)


danger_counter = defaultdict(int)


print("Part 1 loaded successfully")