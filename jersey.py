import math
import re
from collections import Counter, defaultdict

import cv2
import numpy as np

from config import (
    JERSEY_NUM_TOP_FRAC, JERSEY_NUM_BOT_FRAC, JERSEY_NUM_SIDE_FRAC,
    OCR_MIN_VOTES, OCR_MIN_CONFIDENCE,
    MERGE_MAX_GAP_FRAMES, MERGE_MAX_DIST,
)

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        except ImportError:
            _reader = False
    return _reader


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    min_h = 64
    if crop.shape[0] < min_h:
        scale = min_h / crop.shape[0]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(gray)


def read_jersey_number(
    frame: np.ndarray, x1: float, y1: float, x2: float, y2: float
) -> int | None:
    reader = _get_reader()
    if not reader:
        return None
    h, w_box = y2 - y1, x2 - x1
    ny1 = int(y1 + JERSEY_NUM_TOP_FRAC * h)
    ny2 = int(y1 + JERSEY_NUM_BOT_FRAC * h)
    nx1 = int(x1 + JERSEY_NUM_SIDE_FRAC * w_box)
    nx2 = int(x2 - JERSEY_NUM_SIDE_FRAC * w_box)
    crop = frame[ny1:ny2, nx1:nx2]
    processed = _preprocess_crop(crop)
    if processed.size == 0:
        return None
    results = reader.readtext(processed, allowlist="0123456789", detail=0)
    for text in results:
        text = text.strip()
        if re.fullmatch(r"\d{1,2}", text):
            n = int(text)
            if 1 <= n <= 99:
                return n
    return None


def best_number(votes: Counter) -> int | None:
    if not votes:
        return None
    num, count = votes.most_common(1)[0]
    total = sum(votes.values())
    if count >= OCR_MIN_VOTES and count / total >= OCR_MIN_CONFIDENCE:
        return num
    return None


def merge_by_number(
    positions: dict[int, list[tuple[float, float]]],
    color_samples: dict[int, list[tuple[float, float, float]]],
    number_votes: dict[int, Counter],
    frame_ranges: dict[int, tuple[int, int]],
    team_labels: dict[int, int],
) -> tuple[
    dict[int, list[tuple[float, float]]],
    dict[int, list[tuple[float, float, float]]],
    dict[int, int | None],
    dict[int, tuple[int, int]],
]:
    """
    Stage 1: merge tracks that share the same (team_id, jersey_number).

    team_id is included in the key so that player #9 from Team 0 and player #9
    from Team 1 are never merged together.

    Returns (positions, color_samples, player_numbers, frame_ranges).
    """
    track_numbers = {pid: best_number(number_votes.get(pid, Counter())) for pid in positions}

    # Key: (team_id, jersey_number) — both must match to be the same player
    identity_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    no_number: list[int] = []

    for pid, num in track_numbers.items():
        if num is not None:
            team = team_labels.get(pid, -1)
            identity_groups[(team, num)].append(pid)
        else:
            no_number.append(pid)

    merged_pos: dict[int, list[tuple[float, float]]] = {}
    merged_colors: dict[int, list[tuple[float, float, float]]] = {}
    merged_ranges: dict[int, tuple[int, int]] = {}
    player_numbers: dict[int, int | None] = {}

    for (team, num), pids in identity_groups.items():
        canonical = min(pids)
        merged_pos[canonical] = [p for pid in pids for p in positions[pid]]
        merged_colors[canonical] = [c for pid in pids for c in color_samples.get(pid, [])]
        merged_ranges[canonical] = (
            min(frame_ranges[pid][0] for pid in pids),
            max(frame_ranges[pid][1] for pid in pids),
        )
        player_numbers[canonical] = num
        if len(pids) > 1:
            print(f"  [#merge] Team {team} jersey #{num}: tracks {sorted(pids)} → ID {canonical}")

    for pid in no_number:
        merged_pos[pid] = positions[pid]
        merged_colors[pid] = color_samples.get(pid, [])
        merged_ranges[pid] = frame_ranges[pid]
        player_numbers[pid] = None

    return merged_pos, merged_colors, player_numbers, merged_ranges


def merge_spatiotemporal(
    positions: dict[int, list[tuple[float, float]]],
    color_samples: dict[int, list[tuple[float, float, float]]],
    frame_ranges: dict[int, tuple[int, int]],
    team_labels: dict[int, int],
) -> tuple[
    dict[int, list[tuple[float, float]]],
    dict[int, list[tuple[float, float, float]]],
    dict[int, tuple[int, int]],
]:
    """
    Stage 2: merge tracks of the same team that are temporally non-overlapping
    and whose boundary positions are spatially close.

    Logic: if player A disappeared and player B appeared shortly after at nearly
    the same location (same team jersey), they are the same player re-entering frame.
    """
    pids = list(positions.keys())

    # Union-Find
    parent = {pid: pid for pid in pids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i, pid_a in enumerate(pids):
        for pid_b in pids[i + 1:]:
            if team_labels.get(pid_a) != team_labels.get(pid_b):
                continue

            a_start, a_end = frame_ranges[pid_a]
            b_start, b_end = frame_ranges[pid_b]

            # Determine which track comes first in time
            if a_end < b_start:
                earlier, later = pid_a, pid_b
                gap = b_start - a_end
            elif b_end < a_start:
                earlier, later = pid_b, pid_a
                gap = a_start - b_end
            else:
                continue  # tracks overlap in time → different players on pitch simultaneously

            if gap > MERGE_MAX_GAP_FRAMES:
                continue

            end_pos = positions[earlier][-1]
            start_pos = positions[later][0]
            dist = math.sqrt((end_pos[0] - start_pos[0]) ** 2 + (end_pos[1] - start_pos[1]) ** 2)

            if dist <= MERGE_MAX_DIST:
                union(pid_a, pid_b)

    # Collect Union-Find groups
    groups: dict[int, list[int]] = defaultdict(list)
    for pid in pids:
        groups[find(pid)].append(pid)

    merged_pos: dict[int, list[tuple[float, float]]] = {}
    merged_colors: dict[int, list[tuple[float, float, float]]] = {}
    merged_ranges: dict[int, tuple[int, int]] = {}

    for group_pids in groups.values():
        canonical = min(group_pids)
        merged_pos[canonical] = [p for pid in group_pids for p in positions[pid]]
        merged_colors[canonical] = [c for pid in group_pids for c in color_samples.get(pid, [])]
        merged_ranges[canonical] = (
            min(frame_ranges[pid][0] for pid in group_pids),
            max(frame_ranges[pid][1] for pid in group_pids),
        )
        if len(group_pids) > 1:
            print(f"  [spatial] same team, same location: tracks {sorted(group_pids)} → ID {canonical}")

    return merged_pos, merged_colors, merged_ranges
