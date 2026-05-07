import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, Counter

warnings.filterwarnings("ignore", category=FutureWarning)

from config import (
    DEVICE, CONFIDENCE, IOU, MODEL_WEIGHTS, FRAME_SKIP,
    TRACK_BUFFER, TRACK_THRESHOLD, MIN_TRACK_FRAMES,
    JERSEY_TOP_FRAC, JERSEY_BOT_FRAC, JERSEY_SIDE_FRAC,
    FIELD_POLYGON, OCR_EVERY_N_FRAMES,
)
from utils import point_in_polygon
from jersey import read_jersey_number


def _jersey_mean_color(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float] | None:
    h, w_box = y2 - y1, x2 - x1
    jy1 = int(y1 + JERSEY_TOP_FRAC * h)
    jy2 = int(y1 + JERSEY_BOT_FRAC * h)
    jx1 = int(x1 + JERSEY_SIDE_FRAC * w_box)
    jx2 = int(x2 - JERSEY_SIDE_FRAC * w_box)
    crop = frame[jy1:jy2, jx1:jx2]
    if crop.size == 0:
        return None
    m = cv2.mean(crop)
    return (m[0], m[1], m[2])  # BGR


def track_players(
    video_path: str,
) -> tuple[
    dict[int, list[tuple[float, float]]],
    dict[int, list[tuple[float, float, float]]],
    dict[int, Counter],
    dict[int, tuple[int, int]],
]:
    """
    Returns (positions, color_samples, number_votes, frame_ranges).
      positions     — {player_id: [(norm_x, norm_y), ...]}
      color_samples — {player_id: [(B, G, R), ...]}
      number_votes  — {player_id: Counter({number: count, ...})}
      frame_ranges  — {player_id: (first_processed_frame, last_processed_frame)}
    """
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

    positions: dict[int, list[tuple[float, float]]] = defaultdict(list)
    color_samples: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    number_votes: dict[int, Counter] = defaultdict(Counter)
    first_frame: dict[int, int] = {}
    last_frame: dict[int, int] = {}
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

        if FIELD_POLYGON is not None and len(detections) > 0:
            mask = np.array([
                point_in_polygon(
                    (x1 + x2) / 2 / frame_width,
                    y2 / frame_height,
                    FIELD_POLYGON,
                )
                for x1, _y1, x2, y2 in detections.xyxy
            ], dtype=bool)
            detections = detections[mask]

        detections = tracker.update_with_detections(detections)

        if detections.tracker_id is not None:
            for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
                x1, y1, x2, y2 = xyxy
                foot_x = (x1 + x2) / 2
                foot_y = y2
                pid = int(tracker_id)
                positions[pid].append((foot_x / frame_width, foot_y / frame_height))
                if pid not in first_frame:
                    first_frame[pid] = frame_num
                last_frame[pid] = frame_num

                color = _jersey_mean_color(frame, x1, y1, x2, y2)
                if color is not None:
                    color_samples[pid].append(color)

                if frame_num % OCR_EVERY_N_FRAMES == 0:
                    num = read_jersey_number(frame, x1, y1, x2, y2)
                    if num is not None:
                        number_votes[pid][num] += 1

        frame_num += 1
        if frame_num % 100 == 0:
            print(f"  Frame {frame_num}/{total_frames}")

    cap.release()

    filtered_pos = {pid: pos for pid, pos in positions.items() if len(pos) >= MIN_TRACK_FRAMES}
    filtered_colors = {pid: color_samples[pid] for pid in filtered_pos}
    filtered_votes = {pid: number_votes[pid] for pid in filtered_pos}
    filtered_ranges = {pid: (first_frame[pid], last_frame[pid]) for pid in filtered_pos}
    print(f"  {len(positions)} raw IDs → {len(filtered_pos)} kept (min {MIN_TRACK_FRAMES} frames).")
    return filtered_pos, filtered_colors, filtered_votes, filtered_ranges
