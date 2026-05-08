import warnings
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

warnings.filterwarnings("ignore", category=FutureWarning)

from config import (
    DEVICE, CONFIDENCE, IOU, MODEL_WEIGHTS, FRAME_SKIP, BATCH_SIZE,
    TRACK_BUFFER, TRACK_THRESHOLD, PRE_MERGE_MIN_FRAMES,
    JERSEY_TOP_FRAC, JERSEY_BOT_FRAC, JERSEY_SIDE_FRAC,
    FIELD_POLYGON, SAMPLE_CROPS_PER_PLAYER,
)
from utils import point_in_polygon


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
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
    jersey_mask = cv2.bitwise_not(green_mask)
    if cv2.countNonZero(jersey_mask) < 30:
        return None
    m = cv2.mean(crop, mask=jersey_mask)
    return (m[0], m[1], m[2])


def _sample_crop(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> np.ndarray | None:
    h, w = int(y2 - y1), int(x2 - x1)
    if h < 15 or w < 8:
        return None
    crop = frame[int(y1):int(y2), int(x1):int(x2)]
    if crop.size == 0:
        return None
    # Upscale with high-quality interpolation for better preview display
    target_h = max(120, h * 2)
    target_w = max(60, w * 2)
    return cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def _process_detections(
    detections: sv.Detections,
    frame: np.ndarray,
    frame_num: int,
    frame_width: int,
    frame_height: int,
    positions: dict,
    color_samples: dict,
    sample_crops: dict,
    first_frame: dict,
    last_frame: dict,
) -> None:
    if detections.tracker_id is None:
        return
    for tracker_id, xyxy in zip(detections.tracker_id, detections.xyxy):
        x1, y1, x2, y2 = xyxy
        pid = int(tracker_id)
        positions[pid].append(((x1 + x2) / 2 / frame_width, y2 / frame_height))
        if pid not in first_frame:
            first_frame[pid] = frame_num
        last_frame[pid] = frame_num
        color = _jersey_mean_color(frame, x1, y1, x2, y2)
        if color is not None:
            color_samples[pid].append(color)
        if len(sample_crops[pid]) < SAMPLE_CROPS_PER_PLAYER:
            crop = _sample_crop(frame, x1, y1, x2, y2)
            if crop is not None:
                sample_crops[pid].append(crop)


def track_players(
    video_path: str,
) -> tuple[
    dict[int, list[tuple[float, float]]],
    dict[int, list[tuple[float, float, float]]],
    dict[int, list[np.ndarray]],
    dict[int, tuple[int, int]],
]:
    """
    Returns (positions, color_samples, sample_crops, frame_ranges).
      frame_ranges — {player_id: (first_frame, last_frame)}
    """
    model = YOLO(MODEL_WEIGHTS)
    model.to(DEVICE)

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
    sample_crops: dict[int, list[np.ndarray]] = defaultdict(list)
    first_frame: dict[int, int] = {}
    last_frame: dict[int, int] = {}

    frame_num = 0
    batch_frames: list[np.ndarray] = []
    batch_frame_nums: list[int] = []
    half = DEVICE == "cuda"

    def flush_batch() -> None:
        if not batch_frames:
            return
        batch_results = model(
            batch_frames, conf=CONFIDENCE, iou=IOU,
            classes=[0], verbose=False, half=half,
        )
        for frame_i, fn, results in zip(batch_frames, batch_frame_nums, batch_results):
            dets = sv.Detections.from_ultralytics(results)
            if FIELD_POLYGON is not None and len(dets) > 0:
                mask = np.array([
                    point_in_polygon(
                        (x1 + x2) / 2 / frame_width,
                        y2 / frame_height,
                        FIELD_POLYGON,
                    )
                    for x1, _y1, x2, y2 in dets.xyxy
                ], dtype=bool)
                dets = dets[mask]
            dets = tracker.update_with_detections(dets)
            _process_detections(
                dets, frame_i, fn, frame_width, frame_height,
                positions, color_samples, sample_crops, first_frame, last_frame,
            )
        batch_frames.clear()
        batch_frame_nums.clear()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % FRAME_SKIP == 0:
            batch_frames.append(frame)
            batch_frame_nums.append(frame_num)
            if len(batch_frames) >= BATCH_SIZE:
                flush_batch()

        frame_num += 1
        if frame_num % 3000 == 0:
            pct = frame_num / total_frames * 100
            print(f"  Frame {frame_num}/{total_frames}  ({pct:.0f}%)")

    flush_batch()
    cap.release()

    filtered_pos = {pid: pos for pid, pos in positions.items() if len(pos) >= PRE_MERGE_MIN_FRAMES}
    filtered_colors = {pid: color_samples[pid] for pid in filtered_pos}
    filtered_crops = {pid: sample_crops[pid] for pid in filtered_pos}
    filtered_ranges = {pid: (first_frame[pid], last_frame[pid]) for pid in filtered_pos}
    print(f"  {len(positions)} raw IDs → {len(filtered_pos)} kept before merge (min {PRE_MERGE_MIN_FRAMES} positions).")
    return filtered_pos, filtered_colors, filtered_crops, filtered_ranges
