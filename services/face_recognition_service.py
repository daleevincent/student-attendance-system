"""
services/face_recognition_service.py

Two-stage face recognition pipeline:

Stage 1 — DETECTION  (OpenCV Haar Cascade + DNN backup)
  Finds WHERE faces are in the frame → returns bounding boxes.

Stage 2 — RECOGNITION (20180402-114759.pb — FaceNet InceptionResnetV1)
  Takes EACH cropped face detected by Stage 1.
  Converts it to a 512-dim embedding.
  Compares against stored student embeddings using cosine similarity.
  Returns the closest match above the threshold.
"""

import os
import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACENET_PATH    = os.path.join(BASE_DIR, 'models', '20180402-114759.pb')
SSD_PROTO_PATH  = os.path.join(BASE_DIR, 'models', 'deploy.prototxt')
SSD_MODEL_PATH  = os.path.join(BASE_DIR, 'models', 'res10_300x300_ssd_iter_140000.caffemodel')

# ── Thresholds ─────────────────────────────────────────────────────────────────
SSD_CONFIDENCE   = 0.3    # low threshold — catch more faces
COSINE_THRESHOLD = 0.55   # minimum cosine similarity to accept a match
FACE_INPUT_SIZE  = 160    # FaceNet input size

# ── Module-level caches ────────────────────────────────────────────────────────
_facenet_sess  = None
_facenet_input = None
_facenet_phase = None
_facenet_embed = None
_ssd_net       = None
_haar_detector = None
_detector_type = None   # 'ssd' | 'haar'


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — FACE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _load_haar_cascade():
    """
    Load Haar cascade trying multiple paths — handles Windows + Linux differences.
    Returns a loaded CascadeClassifier, or raises RuntimeError if none found.
    """
    import sys

    candidates = []

    # 1. cv2.data.haarcascades (most reliable, works on all platforms)
    try:
        p = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        candidates.append(p)
    except Exception:
        pass

    # 2. OpenCV package data directory (Windows pip install path)
    try:
        import cv2 as _cv2
        pkg_dir = os.path.dirname(_cv2.__file__)
        candidates.append(os.path.join(pkg_dir, 'data', 'haarcascade_frontalface_default.xml'))
    except Exception:
        pass

    # 3. Common system paths
    candidates += [
        r'C:\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml',
        r'C:\Users\{}\AppData\Local\Programs\Python\Python311\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml'.format(os.environ.get('USERNAME', '')),
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
    ]

    for path in candidates:
        if path and os.path.exists(path):
            clf = cv2.CascadeClassifier(path)
            if not clf.empty():
                print(f"[FaceService] Haar cascade loaded from: {path}")
                return clf

    # Last resort: search site-packages
    try:
        import site
        for sp in site.getsitepackages():
            path = os.path.join(sp, 'cv2', 'data', 'haarcascade_frontalface_default.xml')
            if os.path.exists(path):
                clf = cv2.CascadeClassifier(path)
                if not clf.empty():
                    print(f"[FaceService] Haar cascade found at: {path}")
                    return clf
    except Exception:
        pass

    raise RuntimeError(
        "Could not find haarcascade_frontalface_default.xml. "
        "Try: pip install opencv-python --upgrade"
    )


def _load_detector():
    global _ssd_net, _haar_detector, _detector_type
    if _detector_type is not None:
        return

    # Always load Haar cascade as the primary detector.
    # The SSD caffemodel requires a specific prototxt that matches its weights exactly —
    # without that file the SSD produces garbage results (detects noise, misses real faces).
    # Haar cascade is reliable, ships with every OpenCV install, and works well for webcams.
    _haar_detector = _load_haar_cascade()
    _detector_type = 'haar'
    print(f"[FaceService] Haar cascade detector ready.")


def detect_faces(image_bytes: bytes) -> list:
    """
    Detect faces in image bytes.
    Returns list of (x1, y1, x2, y2) bounding boxes.
    """
    _load_detector()

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return []

    return _detect_haar(frame)


def _detect_ssd(frame, w, h) -> list:
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    _ssd_net.setInput(blob)
    detections = _ssd_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < SSD_CONFIDENCE:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            faces.append((x1, y1, x2, y2))
    return faces


def _detect_haar(frame) -> list:
    """
    Multi-scale Haar cascade detection tuned for webcam use.
    Uses multiple scale factors and picks the best detections.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Equalize histogram for better detection under varying lighting
    gray = cv2.equalizeHist(gray)

    faces = _haar_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,     # smaller step = more thorough scan
        minNeighbors=4,       # lower = more detections, slightly more false positives
        minSize=(80, 80),     # minimum face size in pixels
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    results = []
    if len(faces) > 0:
        for (x, y, fw, fh) in faces:
            results.append((x, y, x + fw, y + fh))
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — FACE RECOGNITION (FaceNet)
# ══════════════════════════════════════════════════════════════════════════════

def _load_facenet() -> bool:
    global _facenet_sess, _facenet_input, _facenet_phase, _facenet_embed

    if _facenet_sess is not None:
        return True
    if not os.path.exists(FACENET_PATH):
        print(f"[FaceService] FaceNet model not found: {FACENET_PATH}")
        return False

    try:
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()
        graph_def = tf.compat.v1.GraphDef()
        with open(FACENET_PATH, 'rb') as f:
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as g:
            tf.import_graph_def(graph_def, name='')
            _facenet_sess  = tf.compat.v1.Session(graph=g)
            _facenet_input = g.get_tensor_by_name('input:0')
            _facenet_phase = g.get_tensor_by_name('phase_train:0')
            _facenet_embed = g.get_tensor_by_name('embeddings:0')
        print("[FaceService] FaceNet model loaded successfully.")
        return True
    except Exception as e:
        print(f"[FaceService] FaceNet load failed: {e}")
        return False


def _preprocess_face(face_img: np.ndarray) -> np.ndarray:
    """Resize + normalize BGR face crop for FaceNet input."""
    rgb     = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (FACE_INPUT_SIZE, FACE_INPUT_SIZE),
                         interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32)
    arr = (arr - 127.5) / 128.0   # normalize to [-1, 1]
    return np.expand_dims(arr, axis=0)


def _embed_face(face_img: np.ndarray):
    """Run FaceNet on a face crop. Returns 512-dim embedding or None."""
    if not _load_facenet():
        return None
    preprocessed = _preprocess_face(face_img)
    emb = _facenet_sess.run(
        _facenet_embed,
        feed_dict={_facenet_input: preprocessed, _facenet_phase: False}
    )
    return emb[0]   # shape (512,)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN API
# ══════════════════════════════════════════════════════════════════════════════

def predict_face(image_bytes: bytes, db_conn) -> dict:
    """
    Full pipeline: detect faces → embed → match against DB.
    Returns recognition result dict.
    """
    face_boxes = detect_faces(image_bytes)

    if not face_boxes:
        return {
            'recognized':  False,
            'student_id':  None,
            'confidence':  0.0,
            'faces_found': 0,
            'message':     'No face detected in frame'
        }

    # Decode frame for cropping
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Fetch all enrolled students
    students = db_conn.execute(
        "SELECT id, face_embedding FROM students WHERE face_embedding IS NOT NULL"
    ).fetchall()

    if not students:
        return {
            'recognized':  False,
            'student_id':  None,
            'confidence':  0.0,
            'faces_found': len(face_boxes),
            'message':     'No enrolled faces in database — upload photos for students first.'
        }

    best_id    = None
    best_score = -1.0
    h, w       = frame.shape[:2]

    for (x1, y1, x2, y2) in face_boxes:
        # Generous padding
        pad  = 30
        x1p  = max(0, x1 - pad)
        y1p  = max(0, y1 - pad)
        x2p  = min(w, x2 + pad)
        y2p  = min(h, y2 + pad)

        face_crop = frame[y1p:y2p, x1p:x2p]
        if face_crop.size == 0:
            continue

        live_emb = _embed_face(face_crop)
        if live_emb is None:
            return _mock_predict(db_conn)

        for row in students:
            stored_emb = np.frombuffer(row['face_embedding'], dtype=np.float32)
            score = _cosine_similarity(live_emb, stored_emb)
            if score > best_score:
                best_score = score
                best_id    = row['id']

    if best_score >= COSINE_THRESHOLD:
        return {
            'recognized':  True,
            'student_id':  best_id,
            'confidence':  best_score,
            'faces_found': len(face_boxes),
            'message':     f'Recognized via {_detector_type.upper()} + FaceNet'
        }
    else:
        return {
            'recognized':  False,
            'student_id':  None,
            'confidence':  best_score,
            'faces_found': len(face_boxes),
            'message':     f'Face detected but no match (score: {best_score:.2f})'
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ENROLLMENT
# ══════════════════════════════════════════════════════════════════════════════

def enroll_student_face(image_path: str, db_conn, student_id: int) -> bool:
    """
    Compute and store face embedding for a student from their uploaded photo.
    Auto-detects and crops the face before embedding.
    """
    full_path = os.path.join(BASE_DIR, 'static', 'uploads', image_path)
    if not os.path.exists(full_path):
        print(f"[FaceService] Image not found: {full_path}")
        return False

    frame = cv2.imread(full_path)
    if frame is None:
        print(f"[FaceService] Could not read image: {full_path}")
        return False

    _load_detector()
    with open(full_path, 'rb') as f:
        image_bytes = f.read()

    face_boxes = detect_faces(image_bytes)

    if face_boxes:
        # Use the largest detected face
        face_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        x1, y1, x2, y2 = face_boxes[0]
        fh, fw = frame.shape[:2]
        pad = 30
        face_crop = frame[max(0, y1-pad):min(fh, y2+pad),
                          max(0, x1-pad):min(fw, x2+pad)]
        print(f"[FaceService] Face detected in enrollment photo for student {student_id}.")
    else:
        # No face box found — use center crop of image
        h, w = frame.shape[:2]
        margin = min(h, w) // 8
        face_crop = frame[margin:h-margin, margin:w-margin]
        print(f"[FaceService] No face box found — using center crop for student {student_id}.")

    emb = _embed_face(face_crop)
    if emb is None:
        print(f"[FaceService] Enrollment failed for student {student_id} — FaceNet unavailable.")
        return False

    emb_bytes = emb.astype(np.float32).tobytes()
    db_conn.execute(
        "UPDATE students SET face_embedding = ? WHERE id = ?",
        (emb_bytes, student_id)
    )
    db_conn.commit()
    print(f"[FaceService] Student {student_id} enrolled successfully.")
    return True


def get_embedding(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    return _embed_face(frame)


def get_embedding_from_path(image_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    return _embed_face(frame)


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK FALLBACK (only used if FaceNet model file is missing)
# ══════════════════════════════════════════════════════════════════════════════

def _mock_predict(db_conn) -> dict:
    import random
    students = db_conn.execute("SELECT id FROM students").fetchall()
    if not students:
        return {'recognized': False, 'student_id': None, 'confidence': 0.0,
                'faces_found': 0, 'message': 'No students in database'}
    if random.random() < 0.70:
        s = random.choice(students)
        c = round(random.uniform(0.78, 0.99), 4)
        return {'recognized': True, 'student_id': s['id'], 'confidence': c,
                'faces_found': 1, 'message': 'Recognized (mock — FaceNet missing)'}
    c = round(random.uniform(0.10, 0.45), 4)
    return {'recognized': False, 'student_id': None, 'confidence': c,
            'faces_found': 0, 'message': 'Not recognized (mock — FaceNet missing)'}