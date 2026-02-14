import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe hand landmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)


def is_fist(landmarks):
    tips = [8, 12, 16, 20]
    folded = 0
    for tip in tips:
        if landmarks[tip].y > landmarks[tip - 2].y:
            folded += 1
    return folded >= 3


def draw_puppet(frame, landmarks, sleeping=False):
    h, w, _ = frame.shape

    x = int(landmarks[0].x * w)
    y = int(landmarks[0].y * h)

    color = (200, 200, 200) if sleeping else (0, 255, 255)
    cv2.circle(frame, (x, y - 80), 40, color, -1)

    if sleeping:
        cv2.line(frame, (x - 15, y - 90), (x - 5, y - 90), (0, 0, 0), 2)
        cv2.line(frame, (x + 5, y - 90), (x + 15, y - 90), (0, 0, 0), 2)
    else:
        cv2.circle(frame, (x - 10, y - 90), 5, (0, 0, 0), -1)
        cv2.circle(frame, (x + 10, y - 90), 5, (0, 0, 0), -1)

    cv2.line(frame, (x, y - 40), (x, y + 40), (255, 0, 0), 4)

    ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
    px, py = int(landmarks[20].x * w), int(landmarks[20].y * h)

    cv2.line(frame, (x, y - 20), (ix, iy), (0, 255, 0), 3)
    cv2.line(frame, (x, y - 20), (px, py), (0, 255, 0), 3)

    cv2.line(frame, (x, y + 40), (x - 20, y + 80), (255, 0, 255), 3)
    cv2.line(frame, (x, y + 40), (x + 20, y + 80), (255, 0, 255), 3)

    if sleeping:
        cv2.putText(frame, "Zzz...", (x - 30, y - 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, frame_id)
    frame_id += 1

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        sleeping = is_fist(landmarks)
        draw_puppet(frame, landmarks, sleeping)

    cv2.imshow("AR Hand Puppet", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
