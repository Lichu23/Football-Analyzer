import cv2


def video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "fps": fps,
        "frame_count": frame_count,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "duration_sec": frame_count / max(fps, 1),
    }
    cap.release()
    return info
