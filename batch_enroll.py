"""
batch_enroll.py — Batch Face Enrollment from Dataset
=====================================================
Reads photos from the dataset/ folder and enrolls each student
into the face recognition system automatically.

Dataset structure expected:
    dataset/
    ├── Dale/          ← folder name = student first name
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    ├── Maria/
    │   └── ...

Usage:
    python batch_enroll.py
    python batch_enroll.py --dataset path/to/dataset
    python batch_enroll.py --dry-run     (preview without enrolling)
"""

import os
import sys
import cv2
import numpy as np
import argparse
from datetime import datetime

# ── Setup paths ───────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
sys.path.insert(0, BASE_DIR)

from utils.database import get_db
from services.face_recognition_service import (
    _load_facenet, _embed_face, _load_detector, detect_faces,
    COSINE_THRESHOLD
)

# ── Folder name → student full name mapping ───────────────────────────────────
# Maps each dataset folder name to the exact full_name in the database
FOLDER_TO_NAME = {
    'Aldrea':    'Aldrea Sarmiento',
    'Psalm':     'Psalm Andal',   
    'Angel':     'Angel Binaluyo',
    'Charlene':  'Charlene De Chavez',
    'Chester':   'Chester Andaya',
    'Dale':      'Dale Vincent Montaño',
    'Diana':     'Diana Baduya',
    'James':     'James Byron',         
    'Jm':        'JM Reyes',
    'Kenneth':   'Kenneth Averion',
    'Kyla':      'Kyla Jamito',
    'Niccollo':  'Mark Niccollo L. Dayrit',
    'Patrick':   'Patrick Eva',
    'Philip':    'Philip Llave',
    'Omar':      'Omar Ghazal',
}

# Supported image extensions
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_image_files(folder_path):
    """Return all image files in a folder."""
    files = []
    for f in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMG_EXTS:
            files.append(os.path.join(folder_path, f))
    return files


def embed_image(img_path):
    """
    Load image, detect face, crop, embed with FaceNet.
    Returns 512-dim embedding or None if failed.
    """
    frame = cv2.imread(img_path)
    if frame is None:
        return None, 'could not read image'

    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    boxes = detect_faces(img_bytes)

    if boxes:
        # Use largest detected face
        boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        x1, y1, x2, y2 = boxes[0]
        h, w = frame.shape[:2]
        pad  = 20
        crop = frame[max(0,y1-pad):min(h,y2+pad),
                     max(0,x1-pad):min(w,x2+pad)]
        face_detected = True
    else:
        # No face box — use center crop
        h, w   = frame.shape[:2]
        margin = min(h, w) // 8
        crop   = frame[margin:h-margin, margin:w-margin]
        face_detected = False

    if crop.size == 0:
        return None, 'empty crop'

    emb = _embed_face(crop)
    return emb, ('face detected' if face_detected else 'no face box — used full image')


def enroll_student(db, student_row, embeddings, folder_name):
    """Average all embeddings and store in database."""
    avg_emb = np.mean(embeddings, axis=0)
    norm    = np.linalg.norm(avg_emb)
    final   = (avg_emb / norm).astype(np.float32) if norm > 0 else avg_emb.astype(np.float32)

    db.execute(
        "UPDATE students SET face_embedding = ? WHERE id = ?",
        (final.tobytes(), student_row['id'])
    )
    db.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Batch enroll students from dataset folder')
    parser.add_argument('--dataset', default=DATASET_DIR, help='Path to dataset folder')
    parser.add_argument('--dry-run', action='store_true', help='Preview without enrolling')
    args = parser.parse_args()

    dataset_path = args.dataset
    dry_run      = args.dry_run

    print("=" * 60)
    print("  AttendX — Batch Face Enrollment")
    print(f"  Dataset: {dataset_path}")
    print(f"  Mode:    {'DRY RUN (no changes)' if dry_run else 'LIVE ENROLLMENT'}")
    print("=" * 60)
    print()

    # Check dataset folder exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset folder not found: {dataset_path}")
        print("Make sure your dataset/ folder is in the same directory as this script.")
        sys.exit(1)

    # Load FaceNet model
    print("Loading FaceNet model...")
    if not _load_facenet():
        print("ERROR: FaceNet model not found.")
        print(f"Make sure 20180402-114759.pb is in: {os.path.join(BASE_DIR, 'models')}")
        sys.exit(1)
    print("FaceNet loaded OK.")

    # Load face detector
    print("Loading face detector...")
    _load_detector()
    print("Detector loaded OK.")
    print()

    # Connect to database
    db = get_db()

    # Get all subfolders in dataset
    folders = sorted([
        f for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f))
    ])

    print(f"Found {len(folders)} folder(s) in dataset: {', '.join(folders)}")
    print()

    # Track results
    results = {
        'enrolled':   [],
        'skipped':    [],
        'not_found':  [],
        'no_images':  [],
        'failed':     []
    }

    for folder_name in folders:
        folder_path = os.path.join(dataset_path, folder_name)
        print(f"── {folder_name} {'─'*(40-len(folder_name))}")

        # Map folder name to full name
        if folder_name not in FOLDER_TO_NAME:
            print(f"   ⚠  No mapping defined for folder '{folder_name}' — skipping.")
            print(f"      Add it to FOLDER_TO_NAME in batch_enroll.py")
            results['skipped'].append(folder_name)
            continue

        full_name = FOLDER_TO_NAME[folder_name]
        print(f"   Student: {full_name}")

        # Find student in database
        student = db.execute(
            "SELECT * FROM students WHERE full_name = ?", (full_name,)
        ).fetchone()

        if not student:
            # Try partial match
            student = db.execute(
                "SELECT * FROM students WHERE full_name LIKE ?", (f"%{folder_name}%",)
            ).fetchone()

        if not student:
            print(f"   ✗  Not found in database — skipping.")
            print(f"      Make sure '{full_name}' is registered in the Students page first.")
            results['not_found'].append(folder_name)
            continue

        print(f"   ID: {student['student_number']} (db id={student['id']})")

        # Get image files
        images = get_image_files(folder_path)
        if not images:
            print(f"   ✗  No images found in folder.")
            results['no_images'].append(folder_name)
            continue

        print(f"   Photos: {len(images)} image(s) found")

        if dry_run:
            print(f"   [DRY RUN] Would process {len(images)} images and enroll.")
            results['enrolled'].append(folder_name)
            continue

        # Embed each image
        embeddings  = []
        failed_imgs = 0

        for img_path in images:
            fname = os.path.basename(img_path)
            emb, note = embed_image(img_path)
            if emb is not None:
                embeddings.append(emb)
                print(f"   ✓  {fname} — {note}")
            else:
                failed_imgs += 1
                print(f"   ✗  {fname} — failed ({note})")

        if not embeddings:
            print(f"   ✗  Could not extract any embeddings — enrollment skipped.")
            results['failed'].append(folder_name)
            continue

        # Average and store
        enroll_student(db, student, embeddings, folder_name)

        already = '(overwrote existing enrollment)' if student['face_embedding'] else '(new enrollment)'
        print(f"   ✅ Enrolled from {len(embeddings)}/{len(images)} photos. {already}")
        if failed_imgs:
            print(f"      {failed_imgs} photo(s) failed embedding — excluded from average.")
        results['enrolled'].append(folder_name)
        print()

    db.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  ENROLLMENT SUMMARY")
    print("=" * 60)
    print(f"  ✅ Enrolled:        {len(results['enrolled'])} student(s)")
    print(f"  ⚠  No mapping:     {len(results['skipped'])} folder(s)")
    print(f"  ✗  Not in DB:      {len(results['not_found'])} student(s)")
    print(f"  ✗  No images:      {len(results['no_images'])} folder(s)")
    print(f"  ✗  Embed failed:   {len(results['failed'])} student(s)")
    print()

    if results['enrolled']:
        print(f"  Enrolled: {', '.join(results['enrolled'])}")
    if results['not_found']:
        print(f"  Not in DB (register first): {', '.join(results['not_found'])}")
    if results['skipped']:
        print(f"  No mapping (add to FOLDER_TO_NAME): {', '.join(results['skipped'])}")
    print()

    if not dry_run and results['enrolled']:
        print("  All enrolled students can now be recognized in Face Scan.")
    elif dry_run:
        print("  Dry run complete. Run without --dry-run to actually enroll.")
    print("=" * 60)


if __name__ == '__main__':
    main()