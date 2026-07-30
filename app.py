import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Load the YOLO models: one for detecting the railway track, one for detecting people
line_model = YOLO('/kaggle/input/subway-yellow-safety-line-detection/pytorch/default/1/Line_detection_weights (best).pt')  # Model for detecting "stopbraille-blocks" and "Railway track"
person_model = YOLO('/kaggle/input/people-detection-model/pytorch/default/1/People_detection_weights(best).pt')  # Model for detecting people

