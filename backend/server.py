import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "headers;Referer: https://www.earthcam.com/"
)

import logging
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ultralytics import YOLO


FEED_1_SOURCE = os.environ.get("CROWDWATCH_FEED_1_SOURCE", "")
FEED_2_SOURCE = os.environ.get("CROWDWATCH_FEED_2_SOURCE", "")
FEED_1_FALLBACK = "backup_feed1.mp4"
FEED_3_SOURCE = os.environ.get("CROWDWATCH_FEED_3_SOURCE", "backup_feed3.mp4")
FEED_4_SOURCE = os.environ.get("CROWDWATCH_FEED_4_SOURCE", "backup_feed4.mp4")
FEED_5_SOURCE = os.environ.get("CROWDWATCH_FEED_5_SOURCE", "backup_feed5.mp4")
FEED_6_SOURCE = os.environ.get("CROWDWATCH_FEED_6_SOURCE", "backup_feed6.mp4")
FEED_IDS = tuple(range(1, 7))
FEED_SOURCES = {
    1: FEED_1_SOURCE,
    2: FEED_2_SOURCE,
    3: FEED_3_SOURCE,
    4: FEED_4_SOURCE,
    5: FEED_5_SOURCE,
    6: FEED_6_SOURCE,
}
DEFAULT_ZONE_AREAS = {1: 260.0, 2: 180.0, 3: 220.0, 4: 220.0, 5: 220.0, 6: 220.0}
FRAME_SIZE = (640, 360)
FRAME_INTERVAL = 1 / 30
INFERENCE_INTERVAL = 1 / 5
TILE_WIDTH = 352
TILE_HEIGHT = 200
PERSON_CONFIDENCE = 0.10
PERSON_IOU = 0.75
PERSON_NMS_IOU = 0.72
BOUNDARY = b"--frame\r\n"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crowdwatch")

app = FastAPI(title="CrowdWatch YOLOv8 backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = {
    **{f"feed{feed}_count": 0 for feed in FEED_IDS},
    **{f"feed{feed}_live": False for feed in FEED_IDS},
    **{f"feed{feed}_detection_fps": 0.0 for feed in FEED_IDS},
    **{f"feed{feed}_shared_source": False for feed in FEED_IDS if feed > 1},
    "fps": 0.0,
    "detection_fps": 0.0,
    "status": "FALLBACK",
}
_state_lock = threading.Lock()
_model_lock = threading.Lock()
_source_lock = threading.Lock()
_jpeg_lock = threading.Lock()
_fps_lock = threading.Lock()
_detection_fps_lock = threading.Lock()
_frame_lock = threading.Lock()
_tracking_lock = threading.Lock()
_model = None
_inference_device = "cpu"
_stop_event = threading.Event()
_fps_window_start = time.monotonic()
_detection_window_start = time.monotonic()
_fps_frames = {feed: 0 for feed in FEED_IDS}
_stream_fps = {feed: 0.0 for feed in FEED_IDS}
_detection_frames = {feed: 0 for feed in FEED_IDS}
_detection_rates = {feed: 0.0 for feed in FEED_IDS}
_sources = dict(FEED_SOURCES)
_source_versions = {feed: 0 for feed in FEED_IDS}
_feed_live = {feed: False for feed in FEED_IDS}
_latest_jpeg = {feed: None for feed in FEED_IDS}
_latest_frames = {feed: None for feed in FEED_IDS}
_latest_inference_frames = {feed: None for feed in FEED_IDS}
_feed_has_video_source = {feed: False for feed in FEED_IDS}
_frame_versions = {feed: 0 for feed in FEED_IDS}
_detection_versions = {feed: 0 for feed in FEED_IDS}
_frame_events = {feed: threading.Event() for feed in FEED_IDS}
_latest_detections = {feed: [] for feed in FEED_IDS}
_latest_raw_detections = {feed: [] for feed in FEED_IDS}
_person_tracks = {feed: {} for feed in FEED_IDS}
_next_person_id = {feed: 1 for feed in FEED_IDS}
_feed_shares_source = {feed: False for feed in FEED_IDS}
_zone_polygons = {feed: None for feed in FEED_IDS}
_zone_areas = dict(DEFAULT_ZONE_AREAS)


class StreamUpdate(BaseModel):
    feed: int = Field(ge=1, le=6)
    url: str = Field(min_length=1)


class ZoneCalibration(BaseModel):
    feed: int = Field(ge=1, le=6)
    area_m2: float = Field(gt=0)
    points: list[tuple[float, float]]


def _resolve_source(source: str) -> str:
    if source.lower().startswith(("http://", "https://", "rtsp://")):
        return source
    candidates = (Path(source), Path(__file__).parent / source)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return source


def _open_capture(source: str):
    resolved = _resolve_source(source)
    try:
        capture = cv2.VideoCapture(
            resolved,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5000,
            ],
        )
    except (TypeError, cv2.error):
        capture = cv2.VideoCapture(resolved)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def _fallback_sources(feed: int, primary: str, feed1_source: str) -> list[str]:
    candidates = []
    if feed == 1:
        candidates.append(FEED_1_FALLBACK)
    else:
        candidates.extend((feed1_source, FEED_1_FALLBACK))
    return [candidate for candidate in candidates if candidate != primary]


def _same_camera_source(left: str, right: str) -> bool:
    def identity(source: str) -> tuple[str, ...]:
        parsed = urlsplit(source)
        if parsed.scheme in ("http", "https"):
            camera_segment = next(
                (part for part in parsed.path.split("/") if part.endswith(".flv")),
                None,
            )
            if camera_segment:
                return ("camera", parsed.netloc.lower(), camera_segment.lower())
        return ("source", source)

    return identity(left) == identity(right)


def _synthetic_frame(feed: int) -> np.ndarray:
    frame = np.full((FRAME_SIZE[1], FRAME_SIZE[0], 3), (30, 35, 48), dtype=np.uint8)
    for x in range(0, FRAME_SIZE[0], 40):
        cv2.line(frame, (x, 0), (x, FRAME_SIZE[1]), (42, 48, 62), 1)
    for y in range(0, FRAME_SIZE[1], 40):
        cv2.line(frame, (0, y), (FRAME_SIZE[0], y), (42, 48, 62), 1)
    cv2.putText(
        frame,
        f"FEED {feed} - CAMERA UNAVAILABLE",
        (24, 170),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (185, 193, 208),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Synthetic fallback",
        (24, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (115, 189, 248),
        1,
        cv2.LINE_AA,
    )
    return frame


def _draw_detections(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int, float, int]],
    feed: int,
) -> np.ndarray:
    color = (11, 158, 245) if feed == 1 else (248, 189, 56)
    for x1, y1, x2, y2, confidence, person_id in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        label = f"ID {person_id} {confidence:.2f}"
        text_y = max(13, y1 - 4)
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def _detect_people(frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    frame_height, frame_width = frame.shape[:2]
    if frame_width <= FRAME_SIZE[0] and frame_height <= FRAME_SIZE[1]:
        tile_width = min(TILE_WIDTH, frame_width)
        tile_height = min(TILE_HEIGHT, frame_height)
        x_starts = (0, frame_width - tile_width)
        y_starts = (0, frame_height - tile_height)
    else:
        columns = max(1, int(np.ceil(frame_width / FRAME_SIZE[0])))
        rows = max(1, int(np.ceil(frame_height / FRAME_SIZE[1])))
        tile_width = min(frame_width, int(np.ceil(frame_width / columns * 1.15)))
        tile_height = min(frame_height, int(np.ceil(frame_height / rows * 1.15)))
        x_starts = np.rint(
            np.linspace(0, frame_width - tile_width, columns)
        ).astype(int)
        y_starts = np.rint(
            np.linspace(0, frame_height - tile_height, rows)
        ).astype(int)
    origins = [(int(x), int(y)) for y in y_starts for x in x_starts]
    tiles = [frame[y : y + tile_height, x : x + tile_width] for x, y in origins]
    results = _model(
        tiles,
        classes=[0],
        conf=PERSON_CONFIDENCE,
        iou=PERSON_IOU,
        imgsz=640,
        max_det=1000,
        verbose=False,
        device=_inference_device,
    )

    detections = []
    for result, (offset_x, offset_y) in zip(results, origins):
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = max(0, min(frame_width - 1, int(x1) + offset_x))
            y1 = max(0, min(frame_height - 1, int(y1) + offset_y))
            x2 = max(x1 + 1, min(frame_width, int(x2) + offset_x))
            y2 = max(y1 + 1, min(frame_height, int(y2) + offset_y))
            detections.append((x1, y1, x2, y2, float(box.conf[0])))

    if not detections:
        return []
    nms_boxes = [
        [x1, y1, x2 - x1, y2 - y1]
        for x1, y1, x2, y2, _ in detections
    ]
    confidences = [detection[4] for detection in detections]
    kept = cv2.dnn.NMSBoxes(
        nms_boxes, confidences, PERSON_CONFIDENCE, PERSON_NMS_IOU
    )
    indices = np.asarray(kept).reshape(-1).tolist()
    scale_x = FRAME_SIZE[0] / frame_width
    scale_y = FRAME_SIZE[1] / frame_height
    return [
        (
            int(detections[index][0] * scale_x),
            int(detections[index][1] * scale_y),
            max(1, int(detections[index][2] * scale_x)),
            max(1, int(detections[index][3] * scale_y)),
            detections[index][4],
        )
        for index in indices
    ]


def _box_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(1, left[2] - left[0]) * max(1, left[3] - left[1])
    right_area = max(1, right[2] - right[0]) * max(1, right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _assign_track_ids(
    detections: list[tuple[int, int, int, int, float]], feed: int
) -> list[tuple[int, int, int, int, float, int]]:
    now = time.monotonic()
    tracks = _person_tracks[feed]
    expired = [
        track_id
        for track_id, track in tracks.items()
        if now - track["last_seen"] > 1.5
    ]
    for track_id in expired:
        del tracks[track_id]

    candidates = []
    for detection_index, detection in enumerate(detections):
        box = detection[:4]
        for track_id, track in tracks.items():
            overlap = _box_iou(box, track["box"])
            if overlap >= 0.2:
                candidates.append((overlap, detection_index, track_id))
    candidates.sort(reverse=True)

    matched_detections = set()
    matched_tracks = set()
    track_ids = {}
    for _, detection_index, track_id in candidates:
        if detection_index in matched_detections or track_id in matched_tracks:
            continue
        matched_detections.add(detection_index)
        matched_tracks.add(track_id)
        track_ids[detection_index] = track_id

    for detection_index, detection in enumerate(detections):
        if detection_index not in track_ids:
            track_ids[detection_index] = _next_person_id[feed]
            _next_person_id[feed] += 1
        track_id = track_ids[detection_index]
        tracks[track_id] = {"box": detection[:4], "last_seen": now}

    return [
        (*detection, track_ids[index])
        for index, detection in enumerate(detections)
    ]


def _track_detections(
    detections: list[tuple[int, int, int, int, float]], feed: int
) -> list[tuple[int, int, int, int, float, int]]:
    with _tracking_lock:
        return _assign_track_ids(detections, feed)


def _filter_zone(
    detections: list[tuple[int, int, int, int, float]], feed: int
) -> list[tuple[int, int, int, int, float]]:
    with _frame_lock:
        polygon = _zone_polygons[feed]
    if polygon is None:
        return detections
    return [
        detection
        for detection in detections
        if cv2.pointPolygonTest(
            polygon,
            ((detection[0] + detection[2]) // 2, detection[3]),
            False,
        )
        >= 0
    ]


def _refresh_status() -> None:
    for feed in FEED_IDS:
        if feed > 1 and _feed_shares_source[feed]:
            _feed_live[feed] = _feed_live[1]
    with _state_lock:
        for feed in FEED_IDS:
            state[f"feed{feed}_live"] = _feed_live[feed]
        state["status"] = "LIVE" if all(_feed_live.values()) else "FALLBACK"


def _feed_worker(feed: int) -> None:
    capture = None
    active_source = None
    active_version = -1
    shared_frame_version = -1
    consecutive_failures = 0
    force_fallback = False
    share_feed1 = False
    next_open_attempt = 0.0
    capture_interval = FRAME_INTERVAL
    while not _stop_event.is_set():
        with _source_lock:
            primary = _sources[feed]
            feed1_source = _sources[1]
            version = _source_versions[feed]

        if version != active_version or (
            capture is None
            and not share_feed1
            and time.monotonic() >= next_open_attempt
        ):
            if capture is not None:
                capture.release()
            capture = None
            share_feed1 = False
            if version != active_version:
                force_fallback = False
                next_open_attempt = 0.0
            active_version = version
            active_source = None
            with _frame_lock:
                _feed_has_video_source[feed] = False

            candidates = _fallback_sources(feed, primary, feed1_source)
            if not force_fallback:
                candidates.insert(0, primary)
            for candidate in candidates:
                if feed != 1 and _same_camera_source(candidate, feed1_source):
                    active_source = candidate
                    share_feed1 = True
                    break
                capture = _open_capture(candidate)
                if capture is not None:
                    active_source = candidate
                    break
            if (
                feed != 1
                and capture is None
                and not share_feed1
                and feed1_source == primary
            ):
                active_source = feed1_source
                share_feed1 = True

            with _state_lock:
                _feed_live[feed] = active_source == primary or (
                    share_feed1 and _feed_live[1]
                )
                if feed != 1:
                    state[f"feed{feed}_shared_source"] = share_feed1
            with _frame_lock:
                _feed_has_video_source[feed] = capture is not None or (
                    share_feed1 and _feed_has_video_source[1]
                )
            _refresh_status()
            if capture is None:
                if share_feed1:
                    logger.info("Feed %s is sharing Feed 1's live capture", feed)
                else:
                    logger.warning("Feed %s unavailable; serving a synthetic frame", feed)
                    next_open_attempt = time.monotonic() + 2.0
            elif active_source != primary:
                logger.warning("Feed %s using fallback source %s", feed, active_source)
            if capture is not None:
                source_fps = capture.get(cv2.CAP_PROP_FPS)
                capture_interval = (
                    1 / min(30.0, source_fps) if source_fps > 0 else FRAME_INTERVAL
                )
            else:
                capture_interval = FRAME_INTERVAL
            with _frame_lock:
                _feed_shares_source[feed] = share_feed1

        loop_started = time.monotonic()
        if share_feed1:
            with _frame_lock:
                shared_frame = _latest_frames[1]
                shared_inference_frame = _latest_inference_frames[1]
                source_frame_version = _frame_versions[1]
            if shared_frame is None:
                frame = _synthetic_frame(feed)
                inference_frame = frame
            elif source_frame_version != shared_frame_version:
                frame = shared_frame.copy()
                inference_frame = shared_inference_frame
                shared_frame_version = source_frame_version
            else:
                delay = FRAME_INTERVAL - (time.monotonic() - loop_started)
                if delay > 0:
                    _stop_event.wait(delay)
                continue
        elif capture is None:
            frame = _synthetic_frame(feed)
            inference_frame = frame
        else:
            ok, inference_frame = capture.read()
            if not ok or inference_frame is None:
                consecutive_failures += 1
                is_local_file = not active_source.lower().startswith(
                    ("http://", "https://", "rtsp://")
                )
                if is_local_file and consecutive_failures <= 2:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue
                logger.warning("Feed %s stopped producing frames; reopening source", feed)
                force_fallback = True
                next_open_attempt = 0.0
                with _frame_lock:
                    _feed_has_video_source[feed] = False
                capture.release()
                capture = None
                consecutive_failures = 0
                continue
            consecutive_failures = 0

        frame = cv2.resize(inference_frame, FRAME_SIZE, interpolation=cv2.INTER_AREA)
        with _frame_lock:
            _latest_frames[feed] = frame.copy()
            _latest_inference_frames[feed] = inference_frame.copy()
            _frame_versions[feed] += 1
            _frame_events[feed].set()

        delay = capture_interval - (time.monotonic() - loop_started)
        if delay > 0:
            _stop_event.wait(delay)

    if capture is not None:
        capture.release()


def _record_stream_frame(feed: int) -> None:
    global _fps_window_start
    with _fps_lock:
        _fps_frames[feed] += 1
        now = time.monotonic()
        elapsed = now - _fps_window_start
        if elapsed >= 1.0:
            for stream_feed in FEED_IDS:
                _stream_fps[stream_feed] = _fps_frames[stream_feed] / elapsed
                _fps_frames[stream_feed] = 0
            _fps_window_start = now
            with _state_lock:
                state["fps"] = round(sum(_stream_fps.values()) / len(FEED_IDS), 1)


def _record_detection(feed: int) -> None:
    global _detection_window_start
    with _detection_fps_lock:
        _detection_frames[feed] += 1
        now = time.monotonic()
        elapsed = now - _detection_window_start
        if elapsed >= 1.0:
            for detection_feed in FEED_IDS:
                _detection_rates[detection_feed] = (
                    _detection_frames[detection_feed] / elapsed
                )
                _detection_frames[detection_feed] = 0
            _detection_window_start = now
            with _state_lock:
                state["detection_fps"] = round(
                    sum(
                        _detection_rates[feed]
                        for feed in FEED_IDS
                        if _feed_has_video_source[feed]
                        and not _feed_shares_source[feed]
                    )
                    / max(
                        1,
                        sum(
                            _feed_has_video_source[feed]
                            and not _feed_shares_source[feed]
                            for feed in FEED_IDS
                        ),
                    ),
                    1,
                )
                for detection_feed in FEED_IDS:
                    state[f"feed{detection_feed}_detection_fps"] = round(
                        _detection_rates[detection_feed], 1
                    )


def _render_worker(feed: int) -> None:
    previous_frame = None
    current_frame = None
    current_frame_time = 0.0
    source_interval = FRAME_INTERVAL
    last_frame_version = 0
    was_sharing_source = False

    while not _stop_event.is_set():
        loop_started = time.monotonic()
        with _frame_lock:
            shares_source = _feed_shares_source[feed]
        if feed != 1 and shares_source:
            was_sharing_source = True
            with _jpeg_lock:
                shared_jpeg = _latest_jpeg[1]
                if shared_jpeg is not None:
                    _latest_jpeg[feed] = shared_jpeg
            if shared_jpeg is not None:
                _record_stream_frame(feed)
            _stop_event.wait(FRAME_INTERVAL)
            continue
        if was_sharing_source:
            previous_frame = None
            current_frame = None
            last_frame_version = 0
            was_sharing_source = False

        with _frame_lock:
            latest_frame = _latest_frames[feed]
            frame_version = _frame_versions[feed]
            detections = list(_latest_detections[feed])

        if latest_frame is not None and frame_version != last_frame_version:
            if current_frame is not None:
                previous_frame = current_frame
                source_interval = max(
                    FRAME_INTERVAL, loop_started - current_frame_time
                )
            else:
                previous_frame = latest_frame
            current_frame = latest_frame
            current_frame_time = loop_started
            last_frame_version = frame_version

        if current_frame is None:
            _stop_event.wait(FRAME_INTERVAL)
            continue

        if previous_frame is None:
            frame = current_frame.copy()
        else:
            progress = min(
                1.0,
                max(0.0, (loop_started - current_frame_time) / source_interval),
            )
            frame = cv2.addWeighted(
                previous_frame, 1.0 - progress, current_frame, progress, 0
            )
        frame = _draw_detections(frame, detections, feed)

        encoded, jpeg = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
        )
        if encoded:
            with _jpeg_lock:
                _latest_jpeg[feed] = jpeg.tobytes()
            _record_stream_frame(feed)

        delay = FRAME_INTERVAL - (time.monotonic() - loop_started)
        if delay > 0:
            _stop_event.wait(delay)


def _inference_worker(feed: int) -> None:
    last_frame_version = 0
    last_shared_detection_version = 0
    last_inference = 0.0
    while not _stop_event.is_set():
        _frame_events[feed].wait(timeout=1.0)
        if _stop_event.is_set():
            return
        remaining = INFERENCE_INTERVAL - (time.monotonic() - last_inference)
        if remaining > 0 and _stop_event.wait(remaining):
            return
        with _frame_lock:
            _frame_events[feed].clear()
            frame = _latest_frames[feed]
            inference_frame = _latest_inference_frames[feed]
            frame_version = _frame_versions[feed]
            shares_source = _feed_shares_source[feed]
            has_video_source = _feed_has_video_source[feed]
            shared_detection_version = _detection_versions[1] if feed != 1 else 0
            shared_detections = (
                list(_latest_raw_detections[1]) if feed != 1 else []
            )
        if frame is None or inference_frame is None or frame_version == last_frame_version:
            continue
        last_frame_version = frame_version

        if feed != 1 and shares_source:
            if shared_detection_version == last_shared_detection_version:
                continue
            last_shared_detection_version = shared_detection_version
            filtered_detections = _track_detections(
                _filter_zone(shared_detections, feed), feed
            )
            with _frame_lock:
                _latest_raw_detections[feed] = shared_detections
                _latest_detections[feed] = filtered_detections
            with _state_lock:
                state[f"feed{feed}_count"] = len(filtered_detections)
            continue

        if not has_video_source:
            with _frame_lock:
                if _latest_detections[feed] or _latest_raw_detections[feed]:
                    _latest_raw_detections[feed] = []
                    _latest_detections[feed] = []
                    _detection_versions[feed] += 1
            with _state_lock:
                state[f"feed{feed}_count"] = 0
            continue

        with _model_lock:
            raw_detections = _detect_people(inference_frame)
        detections = _track_detections(_filter_zone(raw_detections, feed), feed)
        with _frame_lock:
            _latest_raw_detections[feed] = raw_detections
            _latest_detections[feed] = detections
            _detection_versions[feed] += 1
        with _state_lock:
            state[f"feed{feed}_count"] = len(detections)
        _record_detection(feed)
        last_inference = time.monotonic()


@app.on_event("startup")
def startup() -> None:
    global _inference_device, _model
    cv2.setNumThreads(1)
    if torch.cuda.is_available():
        _inference_device = 0
    else:
        _inference_device = "cpu"
        torch.set_num_threads(2)
    _model = YOLO(Path(__file__).with_name("yolov8n.pt"))
    _stop_event.clear()
    for feed in FEED_IDS:
        threading.Thread(
            target=_feed_worker,
            args=(feed,),
            name=f"crowdwatch-feed-{feed}",
            daemon=True,
        ).start()
        threading.Thread(
            target=_render_worker,
            args=(feed,),
            name=f"crowdwatch-render-{feed}",
            daemon=True,
        ).start()
        threading.Thread(
            target=_inference_worker,
            args=(feed,),
            name=f"crowdwatch-inference-{feed}",
            daemon=True,
        ).start()


@app.on_event("shutdown")
def shutdown() -> None:
    _stop_event.set()


@app.get("/api/stats")
def get_stats() -> dict:
    with _state_lock:
        return dict(state)


@app.post("/api/update-stream")
def update_stream(update: StreamUpdate) -> dict:
    url = update.url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="url must not be blank")
    with _source_lock:
        _sources[update.feed] = url
        _source_versions[update.feed] += 1
    return {"feed": update.feed, "url": url, "status": "updated"}


@app.post("/api/calibrate-zone")
def calibrate_zone(calibration: ZoneCalibration) -> dict:
    if len(calibration.points) != 4:
        raise HTTPException(status_code=422, detail="Exactly four ROI corner points are required")
    if any(
        not (0 <= x < FRAME_SIZE[0] and 0 <= y < FRAME_SIZE[1])
        for x, y in calibration.points
    ):
        raise HTTPException(status_code=422, detail="ROI points must be inside the 640x360 frame")
    polygon = np.asarray(calibration.points, dtype=np.int32).reshape((-1, 1, 2))
    if abs(cv2.contourArea(polygon)) < 1000:
        raise HTTPException(status_code=422, detail="ROI is too small to calibrate")
    with _frame_lock:
        _zone_polygons[calibration.feed] = polygon
        _zone_areas[calibration.feed] = calibration.area_m2
    return {
        "feed": calibration.feed,
        "area_m2": calibration.area_m2,
        "points": calibration.points,
        "status": "calibrated",
    }


@app.get("/api/calibrations")
def get_calibrations() -> dict:
    with _frame_lock:
        feeds = {
            str(feed): {
                "area_m2": _zone_areas[feed],
                "points": (
                    _zone_polygons[feed].reshape((-1, 2)).tolist()
                    if _zone_polygons[feed] is not None
                    else None
                ),
            }
            for feed in FEED_IDS
        }
    return {"feeds": feeds}


@app.delete("/api/calibrate-zone/{feed}")
def clear_zone_calibration(feed: int) -> dict:
    if feed not in FEED_IDS:
        raise HTTPException(status_code=404, detail="Feed must be between 1 and 6")
    with _frame_lock:
        _zone_polygons[feed] = None
    with _tracking_lock:
        _person_tracks[feed].clear()
    return {"feed": feed, "status": "cleared"}


def _mjpeg_stream(feed: int):
    while not _stop_event.is_set():
        with _jpeg_lock:
            frame = _latest_jpeg[feed]
        if frame is None:
            time.sleep(0.1)
            continue
        yield BOUNDARY + b"Content-Type: image/jpeg\r\nContent-Length: "
        yield str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n"
        time.sleep(FRAME_INTERVAL)


@app.get("/stream/feed{feed}")
def stream_feed(feed: int) -> StreamingResponse:
    if feed not in FEED_IDS:
        raise HTTPException(status_code=404, detail="Feed must be between 1 and 6")
    return StreamingResponse(
        _mjpeg_stream(feed),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
