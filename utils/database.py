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
        sample_students = [
            ('2024-CS-001', 'Maria Santos',     'BSCS', 3, 'A'),
            ('2024-CS-002', 'Juan Dela Cruz',   'BSCS', 3, 'A'),
            ('2024-IT-001', 'Ana Reyes',        'BSIT', 2, 'B'),
            ('2024-IT-002', 'Carlo Mendoza',    'BSIT', 2, 'B'),
            ('2024-CE-001', 'Sofia Garcia',     'BSCE', 1, 'C'),
            ('2024-CE-002', 'Miguel Torres',    'BSCE', 1, 'C'),
            ('2024-BA-001', 'Isabella Flores',  'BSBA', 4, 'A'),
            ('2024-BA-002', 'Rafael Villanueva','BSBA', 4, 'A'),
        ]
        cur.executemany(
            "INSERT INTO students (student_number, full_name, course, year_level, section) VALUES (?, ?, ?, ?, ?)",
            sample_students
        )

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")
