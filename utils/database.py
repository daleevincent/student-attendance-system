"""
utils/database.py
SQLite connection helper and schema initializer.
"""

import sqlite3
import os
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'attendance.db')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = get_db()
    cur  = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS admins (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    NOT NULL UNIQUE,
            password_hash TEXT    NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            student_number TEXT    NOT NULL UNIQUE,
            full_name      TEXT    NOT NULL,
            course         TEXT    NOT NULL,
            year_level     INTEGER NOT NULL,
            section        TEXT    NOT NULL,
            image_path     TEXT,
            face_embedding BLOB,
            created_at     DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add face_embedding column to existing DBs (migration)
    try:
        cur.execute("ALTER TABLE students ADD COLUMN face_embedding BLOB")
        print("[DB] Added face_embedding column.")
    except Exception:
        pass  # column already exists

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_samples (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id  INTEGER NOT NULL,
            image_path  TEXT    NOT NULL,
            captured_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        )
    """)

    # Add face_samples count migration info
    cur.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id       INTEGER NOT NULL,
            attendance_date  DATE    NOT NULL,
            time_in          TIME    NOT NULL,
            status           TEXT    NOT NULL DEFAULT 'Present',
            confidence_score REAL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        )
    """)

    # Default admin
    existing = cur.execute("SELECT id FROM admins WHERE username = 'admin'").fetchone()
    if not existing:
        cur.execute(
            "INSERT INTO admins (username, password_hash) VALUES (?, ?)",
            ('admin', generate_password_hash('admin123'))
        )

    # Sample students
    sample_count = cur.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    if sample_count == 0:
        sample_students = [  # COM221 sample students
            ('2024-0001', 'Maria Santos',     'BSCS', 4, 'COM221'),
            ('2024-0002', 'Juan Dela Cruz',   'BSCS', 4, 'COM221'),
            ('2024-0003', 'Ana Reyes',        'BSCS', 4, 'COM221'),
            ('2024-0004', 'Carlo Mendoza',    'BSCS', 4, 'COM221'),
            ('2024-0005', 'Sofia Garcia',     'BSCS', 4, 'COM221'),
            ('2024-0006', 'Miguel Torres',    'BSCS', 4, 'COM221'),
            ('2024-0007', 'Isabella Flores',  'BSCS', 4, 'COM221'),
            ('2024-0008', 'Rafael Villanueva','BSCS', 4, 'COM221'),
        ]
        cur.executemany(
            "INSERT INTO students (student_number, full_name, course, year_level, section) VALUES (?, ?, ?, ?, ?)",
            sample_students
        )

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")