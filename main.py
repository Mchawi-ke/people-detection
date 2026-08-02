import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------
# Load Models
# -----------------------------
line_model = YOLO("/kaggle/input/subway-yellow-safety-line-detection/pytorch/default/1/Line_detection_weights (best).pt")
person_model = YOLO("/kaggle/input/people-detection-model/pytorch/default/1/People_detection_weights(best).pt")

# -----------------------------
# Video
# -----------------------------
video_path = "/kaggle/input/cctv-video-footage/BB_cff64429-bafc-4258-bdbe-6bba7270af6c_preview.mp4"

cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    "/kaggle/working/output.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

# -----------------------------
# Helper Functions
# -----------------------------
def detect_yellow_line(frame, bbox):
    x1, y1, x2, y2 = bbox

    roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower = np.array([15, 60, 100])
    upper = np.array([40, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    lines = cv2.HoughLinesP(
        mask,
        1,
        np.pi / 180,
        50,
        minLineLength=100,
        maxLineGap=10
    )

    if lines is None:
        return None

    x_positions = []

    for line in lines:
        xa, ya, xb, yb = line[0]

        cv2.line(
            frame,
            (xa + x1, ya + y1),
            (xb + x1, yb + y1),
            (0, 255, 0),
            2
        )

        x_positions.append((xa + xb) / 2 + x1)

    return int(np.mean(x_positions))


def foot_point(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


# -----------------------------
# Main Loop
# -----------------------------
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    railway_x = None
    yellow_x = None

    # -------------------------
    # Detect railway & platform
    # -------------------------
    results = line_model(frame)[0]

    platform_box = None

    for box in results.boxes.data.tolist():

        x1, y1, x2, y2, conf, cls = box

        name = line_model.names[int(cls)]

        if name == "Railway track":
            railway_x = int((x1 + x2) / 2)

        elif name == "stopbraille-blocks":
            platform_box = list(map(int, [x1, y1, x2, y2]))

    if platform_box is not None:
        yellow_x = detect_yellow_line(frame, platform_box)

    railway_left = None

    if railway_x is not None and yellow_x is not None:
        railway_left = railway_x < yellow_x

    # -------------------------
    # Detect People
    # -------------------------
    people = person_model(frame)[0]

    danger_count = 0

    for box in people.boxes.data.tolist():

        x1, y1, x2, y2, conf, cls = box

        if person_model.names[int(cls)] != "Person":
            continue

        bbox = list(map(int, [x1, y1, x2, y2]))

        fx, fy = foot_point(bbox)

        color = (0, 255, 0)

        if railway_left is not None:

            if railway_left and fx < yellow_x:
                color = (0, 0, 255)
                danger_count += 1

            elif not railway_left and fx > yellow_x:
                color = (0, 0, 255)
                danger_count += 1

        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[2], bbox[3]),
            color,
            2
        )

        cv2.circle(frame, (fx, fy), 5, color, -1)

    # -------------------------
    # Danger Counter
    # -------------------------
    overlay = frame.copy()

    status_color = (0, 255, 0)

    if danger_count > 0:
        status_color = (0, 0, 255)

    cv2.rectangle(
        overlay,
        (width - 220, 20),
        (width - 20, 90),
        status_color,
        -1
    )

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(
        frame,
        f"Danger: {danger_count}",
        (width - 205, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    out.write(frame)

cap.release()
out.release()

print("Processing completed.")