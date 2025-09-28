#!/usr/bin/env python3
"""
main.py

Face Detection + Recognition + Tracking + Logging + DB
Uses: ultralytics YOLO (v8), insightface FaceAnalysis, OpenCV, sqlite3, numpy.

Features:
- Processes video file or RTSP stream.
- YOLOv8 for face detection.
- InsightFace embeddings for recognition.
- Registers new faces with unique ID into SQLite DB.
- Tracks faces across frames using CSRT trackers.
- Emits exactly one entry & exit event per presence.
- Saves cropped face images for both entry and exit events.
- Configurable via config.json.
"""

import os
import sys
import argparse
import json
import sqlite3
import logging
from datetime import datetime
import uuid
import cv2
import numpy as np

# YOLO and InsightFace imports
try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("Install ultralytics: pip install ultralytics")

try:
    from insightface.app import FaceAnalysis
except ImportError:
    raise ImportError("Install insightface: pip install insightface")

# ---------------------------
# Config
# ---------------------------
DEFAULT_CONFIG = {
    "detection_skip_frames": 2,
    "detection_conf_threshold": 0.45,
    "embedding_similarity_threshold": 0.40,
    "exit_frame_threshold": 30,
    "save_cropped": True,
    "logs_folder": "logs",
    "db_path": "faces.db",
    "model_yolo": "yolov8n-face.pt",
    "det_size": 640,
    "visualize": True,
    "camera_source": 0
}

CONFIG_PATH = "config.json"

def load_or_create_config(path=CONFIG_PATH):
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = json.load(f)
        for k, v in DEFAULT_CONFIG.items():
            if k not in cfg:
                cfg[k] = v
        return cfg
    else:
        with open(path, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"[INFO] Created default config.json at {path}")
        return DEFAULT_CONFIG.copy()

config = load_or_create_config()

# ---------------------------
# Logging
# ---------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="events.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("face_pipeline")
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(console_handler)

# ---------------------------
# Database (SQLite)
# ---------------------------
DB_PATH = config.get("db_path", "faces.db")

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            last_seen_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            face_id TEXT,
            embedding BLOB,
            created_at TEXT,
            FOREIGN KEY(face_id) REFERENCES faces(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            face_id TEXT,
            event_type TEXT,
            timestamp TEXT,
            img_path TEXT,
            FOREIGN KEY(face_id) REFERENCES faces(id)
        )
    """)
    conn.commit()
    return conn

db_conn = init_db()
db_cur = db_conn.cursor()

def register_face_db(face_id, embedding_vector, timestamp_iso):
    db_cur.execute(
        "INSERT OR REPLACE INTO faces (id, created_at, last_seen_at) VALUES (?, ?, ?)",
        (face_id, timestamp_iso, timestamp_iso)
    )
    emb_blob = embedding_vector.astype(np.float32).tobytes()
    db_cur.execute(
        "INSERT INTO embeddings (face_id, embedding, created_at) VALUES (?, ?, ?)",
        (face_id, emb_blob, timestamp_iso)
    )
    db_conn.commit()

def update_last_seen(face_id, timestamp_iso):
    db_cur.execute("UPDATE faces SET last_seen_at = ? WHERE id = ?", (timestamp_iso, face_id))
    db_conn.commit()

def save_event(event_id, face_id, event_type, timestamp_iso, img_path):
    db_cur.execute(
        "INSERT INTO events (event_id, face_id, event_type, timestamp, img_path) VALUES (?, ?, ?, ?, ?)",
        (event_id, face_id, event_type, timestamp_iso, img_path)
    )
    db_conn.commit()

def get_all_embeddings():
    db_cur.execute("SELECT face_id, embedding FROM embeddings")
    rows = db_cur.fetchall()
    result = []
    for face_id, emb_blob in rows:
        vec = np.frombuffer(emb_blob, dtype=np.float32)
        result.append((face_id, vec))
    return result

def get_unique_visitor_count():
    db_cur.execute("SELECT COUNT(DISTINCT id) FROM faces")
    return int(db_cur.fetchone()[0] or 0)

# ---------------------------
# Utilities
# ---------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def timestamp_iso():
    return datetime.utcnow().isoformat()

def save_cropped_face(img, prefix, logs_folder=config["logs_folder"]):
    if not config.get("save_cropped", True):
        return None
    date_str = datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(logs_folder, prefix, date_str)
    ensure_dir(folder)
    fname = f"{datetime.now().strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(folder, fname)
    cv2.imwrite(path, img)
    return path

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ---------------------------
# Model Initialization
# ---------------------------
logger.info("Initializing models...")

# YOLO
yolo_model_name = config.get("model_yolo", "yolov8n-face.pt")
yolo = YOLO(yolo_model_name)

# InsightFace
face_app = FaceAnalysis(name='buffalo_l')
try:
    face_app.prepare(ctx_id=0, det_size=(config.get("det_size", 640), config.get("det_size", 640)))
except Exception:
    logger.warning("GPU not available for InsightFace, switching to CPU.")
    face_app.prepare(ctx_id=-1, det_size=(config.get("det_size", 640), config.get("det_size", 640)))

logger.info("Models initialized.")

# ---------------------------
# Tracking structures
# ---------------------------
tracked_people = {}  # face_id -> {tracker, last_seen_frame, bbox, last_crop, conf, last_seen_time}

# ---------------------------
# Visualization
# ---------------------------
def _visualize(frame, tracked_people, frame_num, visualize):
    if not visualize:
        return
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 3
    text_color = (0, 0, 255)  # Red text
    text_color2 = (255, 255, 255)  # Blue text
    bbox_color = (0, 255, 0)      # Green bounding box

    # Get frame dimensions
    frame_h, frame_w = vis.shape[:2]

    # ------------------- Draw Corner Annotations -------------------
    # Top-left: Unique visitor count
    visitors_text = f"Visitors: {get_unique_visitor_count()}"
    cv2.putText(vis, visitors_text, (10, 30), font, font_scale, text_color, font_thickness)

    # Top-right: Frame number
    frame_text = f"Frame: {frame_num}"
    (text_w, _), _ = cv2.getTextSize(frame_text, font, font_scale, font_thickness)
    cv2.putText(vis, frame_text, (frame_w - text_w - 10, 30), font, font_scale, text_color, font_thickness)

    # Bottom-left: Timestamp
    timestamp_text = f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    cv2.putText(vis, timestamp_text, (10, frame_h - 10), font, font_scale, text_color, font_thickness)

    # ------------------- Draw Per-Face Annotations -------------------
    for idx, (fid, pdata) in enumerate(tracked_people.items()):
        bb = pdata.get("bbox")
        if not bb:
            continue
        x, y, w, h = bb

        # Draw bounding box
        cv2.rectangle(vis, (x, y), (x+w, y+h), bbox_color, 3)

        # Draw face ID above bounding box (e.g., "id: 0")
        label = f"id: {fid[:5]}.."  # Shorten UUID for display
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = x + (w - text_w) // 2  # Center horizontally
        text_y = max(0, y - 10)         # Place above bounding box
        cv2.putText(vis, label, (text_x, text_y), font, font_scale, text_color, font_thickness)

    # ------------------- Draw Summary Annotations -------------------
    # In-frame count

    inframe_people = len([1 for p in tracked_people.values() if p.get("bbox") is not None])
    inframe_text = f"In-frame: {inframe_people}"
    cv2.putText(vis, inframe_text, (10, 60), font, font_scale, text_color, font_thickness)


    entry_count = db_cur.execute("SELECT COUNT(*) FROM events WHERE event_type='entry'").fetchone()[0] or 0
    exit_count = db_cur.execute("SELECT COUNT(*) FROM events WHERE event_type='exit'").fetchone()[0] or 0

    cv2.putText(vis, f"Entries: {entry_count}", (10, 90), font, font_scale, text_color2, font_thickness)
    cv2.putText(vis, f"Exits: {exit_count}", (10, 120), font, font_scale, text_color2, font_thickness)


    # ------------------- Display Frame -------------------
    cv2.namedWindow("Face Pipeline", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Face Pipeline", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Face Pipeline", vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Quitting by user request.")
        exit(0)

# ---------------------------
# Main Processing Loop
# ---------------------------
def process_video(source, max_frames=None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error(f"Cannot open video source: {source}")
        return

    frame_num = 0
    detect_skip = max(1, int(config.get("detection_skip_frames", 2)))
    conf_thresh = float(config.get("detection_conf_threshold", 0.45))
    sim_thresh = float(config.get("embedding_similarity_threshold", 0.40))
    exit_thresh = int(config.get("exit_frame_threshold", 30))
    visualize = bool(config.get("visualize", True))

    known_embeddings = get_all_embeddings()
    logger.info(f"Loaded {len(known_embeddings)} embeddings from DB.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of stream or cannot fetch frame.")
            break
        frame_num += 1

        # ------------------- Update trackers -------------------
        if frame_num % detect_skip != 0 and tracked_people:
            for fid, pdata in list(tracked_people.items()):
                tracker = pdata.get("tracker")
                if tracker:
                    ok, box = tracker.update(frame)
                    if ok:
                        x, y, w, h = [int(v) for v in box]
                        pdata["bbox"] = (x, y, w, h)
                        pdata["last_seen_frame"] = frame_num
                        update_last_seen(fid, timestamp_iso())
                    else:
                        pdata["bbox"] = None
            _handle_exits(frame, frame_num, exit_thresh)
            if visualize:
                _visualize(frame, tracked_people, frame_num, visualize)
            continue

        # ------------------- YOLO Detection -------------------
        results = yolo.predict(frame, imgsz=config.get("det_size", 640),
                               conf=conf_thresh, verbose=False)
        detections = []
        if results:
            r = results[0]
            if hasattr(r, "boxes"):
                for b in r.boxes:
                    if b.cls != 0:  # Assuming class 0 is face
                        continue
                    xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                    conf = float(b.conf.cpu().numpy()) if hasattr(b, "conf") else float(b.conf)
                    if conf < conf_thresh:
                        continue
                    x1, y1, x2, y2 = xyxy[:4]
                    w, h = x2 - x1, y2 - y1
                    x1, y1 = max(0, x1), max(0, y1)
                    detections.append((x1, y1, w, h, conf))

        # ------------------- Process Detections -------------------
        for x, y, w, h, conf in detections:
            pad = int(0.1 * max(w, h))
            xa, ya = max(0, x - pad), max(0, y - pad)
            xb, yb = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            crop = frame[ya:yb, xa:xb]
            if crop.size == 0:
                continue

            # Get embedding
            faces = face_app.get(crop) or []
            emb = None
            if faces:
                f0 = sorted(faces, key=lambda f: f.det_score, reverse=True)[0]
                emb = getattr(f0, "embedding", None)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32)

            # Compare to known embeddings
            best_sim = -1
            best_id = None
            for fid, known_emb in known_embeddings:
                sim = cosine_similarity(emb, known_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = fid

            timestamp_now = timestamp_iso()
            is_new_face = False
            if best_sim >= sim_thresh and best_id:
                assigned_id = best_id
                logger.info(f"Recognized face {assigned_id} (sim={best_sim:.3f})")
            else:
                # New face
                assigned_id = uuid.uuid4().hex
                is_new_face = True
                register_face_db(assigned_id, emb, timestamp_now)
                known_embeddings.append((assigned_id, emb))
                logger.info(f"Registered new face {assigned_id}")

            # Log entry if this is a new presence
            log_entry = assigned_id not in tracked_people
            if log_entry:
                cropped_path = save_cropped_face(crop, "entries")
                event_uuid = uuid.uuid4().hex
                save_event(event_uuid, assigned_id, "entry", timestamp_now, cropped_path)
                logger.info(f"Face {assigned_id} entry event saved at {cropped_path}")

            if not is_new_face:
                db_cur.execute(
                    "INSERT INTO embeddings (face_id, embedding, created_at) VALUES (?, ?, ?)",
                    (assigned_id, emb.tobytes(), timestamp_now)
                )
                db_conn.commit()
                known_embeddings.append((assigned_id, emb))

            update_last_seen(assigned_id, timestamp_now)

            # Create or update tracker
            try:
                tracker = create_tracker(frame, (x, y, w, h))
            except Exception:
                tracker = None

            tracked_people[assigned_id] = {
                "tracker": tracker,
                "last_seen_frame": frame_num,
                "bbox": (x, y, w, h),
                "last_crop": crop,
                "conf": conf,
                "last_seen_time": timestamp_now,
            }

        _handle_exits(frame, frame_num, exit_thresh)

        # ------------------- Visualization -------------------
        if visualize:
            _visualize(frame, tracked_people, frame_num, visualize)

        if max_frames and frame_num >= max_frames:
            logger.info(f"Reached max_frames={max_frames}. Stopping.")
            break

    _handle_exits(frame, frame_num, 0)

    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    logger.info("Processing finished.")

# ---------------------------
# Exit handler
# ---------------------------
def _handle_exits(frame, current_frame_num, exit_thresh):
    to_remove = []
    for fid, pdata in tracked_people.items():
        last_seen = pdata.get("last_seen_frame", 0)
        if current_frame_num - last_seen > exit_thresh:
            logger.info(f"Face {fid} EXIT at frame {current_frame_num}")
            crop = pdata.get("last_crop")
            img_path = save_cropped_face(crop, "exits") if crop is not None else None
            timestamp_now = timestamp_iso()
            event_uuid = uuid.uuid4().hex
            save_event(event_uuid, fid, "exit", timestamp_now, img_path)
            logger.info(f"Exit event for {fid} saved (img: {img_path})")
            to_remove.append(fid)
    for fid in to_remove:
        tracked_people.pop(fid, None)

def create_tracker(frame, bbox):
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, tuple(bbox))
    return tracker

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-file Face Detection/Recognition Pipeline")
    parser.add_argument("--source", type=str, default=str(config.get("camera_source", 0)),
                        help="Video source (file path or RTSP URL) or integer camera index")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for testing)")
    args = parser.parse_args()

    source = args.source
    if source.isdigit():
        source = int(source)
    logger.info(f"Starting pipeline on source: {source}")
    process_video(source, max_frames=args.max_frames)

if __name__ == "__main__":
    main()
