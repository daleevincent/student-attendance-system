"""
services/face_recognition_service.py

FaceNet (InceptionResnetV1) face recognition service.

Model:  20180402-114759.pb   (trained on MS-Celeb-1M, 512-dim embeddings)
Input:  input:0              shape [batch, 160, 160, 3]  float32, pixel values in [-1, 1]
Output: embeddings:0         shape [batch, 512]           L2-normalised embeddings

HOW RECOGNITION WORKS — No dataset needed
==========================================
1. When a student is REGISTERED with a face photo, we run the photo through
   the model immediately and store the resulting 512-dim embedding in the DB.
2. At scan time we run the webcam frame through the model, compare the live
   embedding to every stored embedding using cosine similarity, and pick the
   closest match above a confidence threshold.

Dataset (training images) are NOT needed at runtime — the model is already
trained. You only need one clear face photo per student for enrollment.
"""

import os
import io
import pickle
import struct
import numpy as np
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', '20180402-114759.pb')

# Node names confirmed by inspection
INPUT_NODE      = 'input:0'
PHASE_TRAIN_NODE= 'phase_train:0'
OUTPUT_NODE     = 'embeddings:0'

FACE_INPUT_SIZE = 160          # model expects 160×160
COSINE_THRESHOLD = 0.60        # minimum cosine similarity to accept a match
                               # raise toward 0.75 for stricter matching

# Module-level TF session (loaded once)
_sess        = None
_input_t     = None
_phase_t     = None
_embed_t     = None
_tf_available = None           # None = not checked yet


# ── TensorFlow loader ─────────────────────────────────────────────────────────
def _load_model():
    """Load the frozen FaceNet graph into a TF1-compat session. Called once."""
    global _sess, _input_t, _phase_t, _embed_t, _tf_available

    if _sess is not None:
        return True     # already loaded

    if not os.path.exists(MODEL_PATH):
        print(f"[FaceService] Model not found at {MODEL_PATH}")
        _tf_available = False
        return False

    try:
        import tensorflow as tf
        tf.compat.v1.disable_eager_execution()

        graph_def = tf.compat.v1.GraphDef()
        with open(MODEL_PATH, 'rb') as f:
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as g:
            tf.import_graph_def(graph_def, name='')
            _sess    = tf.compat.v1.Session(graph=g)
            _input_t = g.get_tensor_by_name(INPUT_NODE)
            _phase_t = g.get_tensor_by_name(PHASE_TRAIN_NODE)
            _embed_t = g.get_tensor_by_name(OUTPUT_NODE)

        _tf_available = True
        print("[FaceService] FaceNet model loaded successfully.")
        return True

    except Exception as e:
        print(f"[FaceService] Failed to load model: {e}")
        _tf_available = False
        return False


# ── Image preprocessing ───────────────────────────────────────────────────────
def _preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a float32 array in [-1, 1] with shape (1, 160, 160, 3).
    FaceNet was trained with (pixel - 127.5) / 128 normalisation.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((FACE_INPUT_SIZE, FACE_INPUT_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr - 127.5) / 128.0          # normalise to [-1, 1]
    arr = np.expand_dims(arr, axis=0)    # → (1, 160, 160, 3)
    return arr


# ── Embedding extraction ──────────────────────────────────────────────────────
def get_embedding(image_bytes: bytes) -> np.ndarray | None:
    """
    Run the model on one image and return a 512-dim L2-normalised embedding.
    Returns None if the model is not available.
    """
    if not _load_model():
        return None

    arr = _preprocess(image_bytes)
    emb = _sess.run(_embed_t, feed_dict={_input_t: arr, _phase_t: False})
    return emb[0]   # shape (512,)


def get_embedding_from_path(image_path: str) -> np.ndarray | None:
    """Convenience wrapper: read a file from disk then embed it."""
    if not os.path.exists(image_path):
        return None
    return get_embedding(open(image_path, 'rb').read())


# ── Cosine similarity ─────────────────────────────────────────────────────────
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D vectors in [−1, 1]."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Main recognition entry point ──────────────────────────────────────────────
def predict_face(image_bytes: bytes, db_conn) -> dict:
    """
    Recognise a face in `image_bytes` against all enrolled students.

    Steps:
      1. Run FaceNet → 512-dim embedding
      2. Compare with every stored embedding in the students table
      3. Return the best match if cosine similarity ≥ COSINE_THRESHOLD

    Args:
        image_bytes: Raw JPEG/PNG bytes of the captured webcam frame.
        db_conn:     Open SQLite connection with row_factory set.

    Returns:
        {
          'recognized': bool,
          'student_id': int | None,
          'confidence': float,      # cosine similarity, 0–1
          'message':    str
        }
    """
    # ── Get live embedding ──────────────────────────────────────────────────
    live_emb = get_embedding(image_bytes)
    if live_emb is None:
        # Model unavailable → fall back to mock
        print("[FaceService] Model unavailable, using mock predictor.")
        return _mock_predict(db_conn)

    # ── Fetch all students that have a stored embedding ─────────────────────
    students = db_conn.execute(
        "SELECT id, face_embedding FROM students WHERE face_embedding IS NOT NULL"
    ).fetchall()

    if not students:
        return {
            'recognized': False,
            'student_id': None,
            'confidence': 0.0,
            'message': 'No enrolled faces to compare against. Upload a face photo for each student first.'
        }

    # ── Compare live embedding to each stored embedding ─────────────────────
    best_id    = None
    best_score = -1.0

    for row in students:
        stored_emb = np.frombuffer(row['face_embedding'], dtype=np.float32)
        score = _cosine_similarity(live_emb, stored_emb)
        if score > best_score:
            best_score = score
            best_id    = row['id']

    # ── Decision ─────────────────────────────────────────────────────────────
    if best_score >= COSINE_THRESHOLD:
        return {
            'recognized': True,
            'student_id': best_id,
            'confidence': best_score,
            'message': 'Face recognized'
        }
    else:
        return {
            'recognized': False,
            'student_id': None,
            'confidence': best_score,
            'message': f'No match (best similarity: {best_score:.2f}, threshold: {COSINE_THRESHOLD})'
        }


# ── Enrollment helper (called when a student photo is uploaded) ───────────────
def enroll_student_face(image_path: str, db_conn, student_id: int) -> bool:
    """
    Compute and store the face embedding for a student.
    Called automatically in app.py whenever a student photo is saved.

    Returns True on success, False if the model is unavailable or the
    image could not be processed.
    """
    full_path = os.path.join(BASE_DIR, 'static', 'uploads', image_path)
    emb = get_embedding_from_path(full_path)
    if emb is None:
        print(f"[FaceService] Could not enroll student {student_id} — embedding failed.")
        return False

    emb_bytes = emb.astype(np.float32).tobytes()   # 512 × 4 = 2048 bytes
    db_conn.execute(
        "UPDATE students SET face_embedding = ? WHERE id = ?",
        (emb_bytes, student_id)
    )
    db_conn.commit()
    print(f"[FaceService] Student {student_id} enrolled successfully.")
    return True


# ── Mock predictor (fallback when model is not available) ─────────────────────
def _mock_predict(db_conn) -> dict:
    """
    Simulate recognition for demo/testing — 70 % chance of a random hit.
    Remove this once real enrollment data exists.
    """
    import random
    students = db_conn.execute("SELECT id FROM students").fetchall()
    if not students:
        return {'recognized': False, 'student_id': None, 'confidence': 0.0,
                'message': 'No students in database'}

    if random.random() < 0.70:
        student    = random.choice(students)
        confidence = round(random.uniform(0.78, 0.99), 4)
        return {'recognized': True, 'student_id': student['id'],
                'confidence': confidence, 'message': 'Face recognized (mock)'}
    else:
        confidence = round(random.uniform(0.10, 0.45), 4)
        return {'recognized': False, 'student_id': None,
                'confidence': confidence, 'message': 'Face not recognized (mock)'}
