import warnings
import cv2
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

from config import DEVICE, CONFIDENCE, IOU, MODEL_WEIGHTS, FRAME_SKIP, TRACK_BUFFER, TRACK_THRESHOLD, MIN_TRACK_FRAMES


def track_players(video_path: str) -> dict[int, list[tuple[float, float]]]:
    model = YOLO(MODEL_WEIGHTS)
    tracker = sv.ByteTrack(
        track_activation_threshold=TRACK_THRESHOLD,
        lost_track_buffer=TRACK_BUFFER,
        minimum_consecutive_frames=3,
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Stores normalized (x, y) foot positions per tracking ID
    positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % FRAME_SKIP != 0:
            frame_num += 1
            continue

        results = model(frame, conf=CONFIDENCE, iou=IOU, device=DEVICE, classes=[0], verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is not None:
            for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                x1, y1, x2, y2 = xyxy
                foot_x = (x1 + x2) / 2  # horizontal center of bounding box
                foot_y = y2              # bottom of bounding box = foot level
                positions[int(tracker_id)].append((foot_x / frame_width, foot_y / frame_height))

        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}/{total_frames}")

    cap.release()

    # Drop ghost tracks — players seen in too few frames are noise
    filtered = {pid: pos for pid, pos in positions.items() if len(pos) >= MIN_TRACK_FRAMES}
    print(f"  {len(positions)} raw IDs → {len(filtered)} kept (min {MIN_TRACK_FRAMES} frames).")
    return filtered
