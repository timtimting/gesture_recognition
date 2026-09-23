import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
os.environ.setdefault("QT_LOGGING_RULES", "*.warning=false")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


GESTURE_NAMES = {
    "fist": "fist",
    "open_palm": "open palm",
    "pointing": "pointing",
    "victory": "victory",
    "thumbs_up": "thumbs up",
    "thumbs_down": "thumbs down",
    "unknown": "unknown",
}

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WINDOW_NAME = "Gesture Recognition"
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
)


def parse_args():
    parser = argparse.ArgumentParser(description="使用电脑摄像头识别常见手势")
    parser.add_argument("--camera-index", type=int, default=0, help="摄像头编号，默认 0")
    parser.add_argument("--width", type=int, default=640, help="摄像头画面宽度")
    parser.add_argument("--height", type=int, default=480, help="摄像头画面高度")
    parser.add_argument(
        "--model-path",
        default=".models/hand_landmarker.task",
        help="Hand Landmarker 模型路径，默认自动下载到 .models/",
    )
    parser.add_argument(
        "--no-blur",
        action="store_true",
        help="关闭背景虚化",
    )
    return parser.parse_args()


def ensure_model(model_path):
    model_file = Path(model_path)
    if model_file.is_file():
        return model_file

    model_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"首次运行，正在下载手部模型到 {model_file} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_file)
    except Exception as error:
        if model_file.exists():
            model_file.unlink()
        raise RuntimeError(
            "手部模型下载失败，请检查网络，或使用 --model-path 指定本地 .task 模型文件"
        ) from error
    return model_file


def finger_states(landmarks, handedness):
    fingers = {}

    fingers["index"] = landmarks[8].y < landmarks[6].y
    fingers["middle"] = landmarks[12].y < landmarks[10].y
    fingers["ring"] = landmarks[16].y < landmarks[14].y
    fingers["pinky"] = landmarks[20].y < landmarks[18].y

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    if handedness == "Right":
        fingers["thumb"] = thumb_tip.x < thumb_ip.x
    else:
        fingers["thumb"] = thumb_tip.x > thumb_ip.x

    return fingers


def classify_gesture(landmarks, handedness):
    fingers = finger_states(landmarks, handedness)
    thumb = fingers["thumb"]
    index = fingers["index"]
    middle = fingers["middle"]
    ring = fingers["ring"]
    pinky = fingers["pinky"]

    if thumb and not index and not middle and not ring and not pinky:
        if landmarks[4].y < landmarks[2].y:
            return "thumbs_up"
        if landmarks[4].y > landmarks[2].y:
            return "thumbs_down"
    if not thumb and not index and not middle and not ring and not pinky:
        return "fist"
    if thumb and index and middle and ring and pinky:
        return "open_palm"
    if not thumb and index and middle and not ring and not pinky:
        return "victory"
    if not thumb and index and not middle and not ring and not pinky:
        return "pointing"
    return "unknown"


def draw_result(frame, landmarks, gesture_name, handedness):
    height, width = frame.shape[:2]
    points = []
    for landmark in landmarks:
        point = (int(landmark.x * width), int(landmark.y * height))
        points.append(point)
        cv2.circle(frame, point, 4, (0, 140, 255), -1)
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 220, 0), 2)

    x = max(10, points[0][0] - 20)
    y = max(35, points[0][1] - 20)
    label = f"{handedness}: {GESTURE_NAMES[gesture_name]}"
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def pixel_point(landmark, width, height):
    x = max(0, min(width - 1, int(landmark.x * width)))
    y = max(0, min(height - 1, int(landmark.y * height)))
    return (x, y)


def two_hand_polygon(frame, hand_landmarks):
    if len(hand_landmarks) < 2:
        return None

    height, width = frame.shape[:2]
    ordered_hands = sorted(hand_landmarks, key=lambda landmarks: landmarks[0].x)
    left_hand, right_hand = ordered_hands[:2]
    return np.array(
        [
            pixel_point(left_hand[4], width, height),
            pixel_point(right_hand[4], width, height),
            pixel_point(right_hand[8], width, height),
            pixel_point(left_hand[8], width, height),
        ],
        dtype=np.int32,
    )


def draw_two_hand_connections(frame, hand_landmarks):
    polygon = two_hand_polygon(frame, hand_landmarks)
    if polygon is None:
        return

    thumb_left, thumb_right, index_right, index_left = polygon
    cv2.line(frame, tuple(thumb_left), tuple(thumb_right), (255, 0, 255), 3)
    cv2.line(frame, tuple(index_left), tuple(index_right), (255, 255, 0), 3)
    cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)
    for point in polygon:
        cv2.circle(frame, tuple(point), 7, (0, 255, 255), -1)


def blur_background(frame, hand_landmarks):
    height, width = frame.shape[:2]
    small_width = max(1, width // 3)
    small_height = max(1, height // 3)
    small_frame = cv2.resize(frame, (small_width, small_height))
    small_blurred = cv2.GaussianBlur(small_frame, (0, 0), sigmaX=6, sigmaY=6)
    blurred = cv2.resize(small_blurred, (width, height), interpolation=cv2.INTER_LINEAR)
    if not hand_landmarks:
        return blurred

    hand_mask = np.zeros((height, width), dtype=np.uint8)
    for landmarks in hand_landmarks:
        points = np.array(
            [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmarks],
            dtype=np.int32,
        )
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(hand_mask, hull, 255)

    clear_polygon = two_hand_polygon(frame, hand_landmarks)
    if clear_polygon is not None:
        cv2.fillPoly(hand_mask, [clear_polygon], 255)

    dilation_size = max(15, int(min(height, width) * 0.04))
    dilation_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
    )
    hand_mask = cv2.dilate(hand_mask, dilation_kernel, iterations=1)
    hand_mask = cv2.GaussianBlur(hand_mask, (0, 0), sigmaX=5, sigmaY=5)
    alpha = hand_mask.astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]
    return (frame * alpha + blurred * (1 - alpha)).astype(np.uint8)


def open_camera(camera_index, width, height):
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    capture = cv2.VideoCapture(camera_index, backend)
    if not capture.isOpened() and backend != cv2.CAP_ANY:
        capture.release()
        capture = cv2.VideoCapture(camera_index)

    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def main():
    args = parse_args()
    model_path = ensure_model(args.model_path)
    capture = open_camera(args.camera_index, args.width, args.height)

    if not capture.isOpened():
        raise RuntimeError(
            f"无法打开摄像头 {args.camera_index}，请检查摄像头权限，或尝试 --camera-index 1"
        )

    last_time = time.perf_counter()
    fps = 0.0
    timestamp_ms = 0
    black_frame_reported = False
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.6,
    )

    try:
        with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
            while True:
                success, frame = capture.read()
                if not success:
                    print(
                        f"读取摄像头画面失败，请尝试 --camera-index 1（当前设备：{args.camera_index}）"
                    )
                    break
                if not black_frame_reported and frame.max() == 0:
                    print(
                        "摄像头返回了全黑画面，请检查摄像头隐私开关、系统权限，或尝试 --camera-index 1"
                    )
                    black_frame_reported = True

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                results = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
                timestamp_ms += 1

                if not args.no_blur:
                    frame = blur_background(frame, results.hand_landmarks)

                if results.hand_landmarks:
                    for index, landmarks in enumerate(results.hand_landmarks):
                        handedness = "手"
                        if index < len(results.handedness):
                            handedness = results.handedness[index][0].category_name or "手"
                        gesture = classify_gesture(landmarks, handedness)
                        draw_result(frame, landmarks, gesture, handedness)
                    draw_two_hand_connections(frame, results.hand_landmarks)

                current_time = time.perf_counter()
                elapsed = current_time - last_time
                if elapsed > 0:
                    fps = 1 / elapsed
                last_time = current_time
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}   Press Q or ESC to quit",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow(WINDOW_NAME, frame)

                key = cv2.waitKey(1) & 0xFF
                try:
                    window_visible = cv2.getWindowProperty(
                        WINDOW_NAME, cv2.WND_PROP_VISIBLE
                    )
                except cv2.error:
                    window_visible = 0
                if key in (ord("q"), 27) or window_visible < 1:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
