# AttendX — Face Recognition Student Attendance System

A modern, full-stack web application for automating student attendance using face recognition. Built with Flask + SQLite + Vanilla JS.

---

## 📁 Project Structure

```
student-attendance-system/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
│
├── database/
│   └── attendance.db               # SQLite database (auto-created on first run)
│
├── models/
│   └── face_model.h5               # Place your trained model here
│
├── services/
│   └── face_recognition_service.py # Face recognition logic (mock + real)
│
├── utils/
│   └── database.py                 # DB connection helper + schema init
│
├── static/
│   ├── css/
│   │   └── main.css                # Full CSS (dark/light theme)
│   ├── js/
│   │   └── main.js                 # Global JS (theme, sidebar, toasts)
│   ├── uploads/                    # Student face photos (auto-created)
│   └── images/                     # Static assets
│
└── templates/
    ├── layout.html                 # Base layout with sidebar
    ├── login.html                  # Admin login page
    ├── dashboard.html              # Dashboard with stats
    ├── students.html               # Student CRUD management
    ├── attendance.html             # Live webcam face scan page
    └── records.html                # Attendance logs with filters
```

---

## 🚀 Quick Start

### 1. Clone / Download the project

```bash
cd student-attendance-system
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

### 5. Open in browser

```
http://127.0.0.1:5000
```

### 6. Login with default credentials

| Field    | Value      |
|----------|------------|
| Username | `admin`    |
| Password | `admin123` |

---

## ✨ Features

| Feature               | Description                                              |
|-----------------------|----------------------------------------------------------|
| 🔐 Admin Auth         | Session-based login with hashed passwords                |
| 📊 Dashboard          | Live clock, stat cards, today's attendance rate          |
| 👤 Student Management | Full CRUD — add, edit, delete, search, photo upload      |
| 📷 Face Scan          | Webcam live feed, auto-scan toggle, session log          |
| 📋 Attendance Records | Filter by date / course / name, pagination, CSV export   |
| 🌙 Dark / Light Mode  | Theme toggle persisted to localStorage                   |
| 📱 Responsive         | Mobile-friendly sidebar + stacked layout                 |

---

## 🤖 Face Recognition Integration

The system ships with a **mock predictor** (70% random hit rate) so the full UI/API pipeline works immediately.

### To plug in a real model:

1. Drop your trained model file into `models/face_model.h5`
2. Open `services/face_recognition_service.py`
3. Edit `load_model()` to load your model
4. Edit `predict_face()` to run real inference

Supported model frameworks (uncomment in `requirements.txt`):
- **TensorFlow / Keras** — `.h5` models
- **DeepFace** — wraps multiple backends (VGG-Face, FaceNet, ArcFace)
- **OpenCV** — Haar cascades + LBPH

---

## 🗄️ Database Schema

### `admins`
| Column        | Type    |
|---------------|---------|
| id            | INTEGER |
| username      | TEXT    |
| password_hash | TEXT    |

### `students`
| Column         | Type     |
|----------------|----------|
| id             | INTEGER  |
| student_number | TEXT     |
| full_name      | TEXT     |
| course         | TEXT     |
| year_level     | INTEGER  |
| section        | TEXT     |
| image_path     | TEXT     |
| created_at     | DATETIME |

### `attendance`
| Column           | Type    |
|------------------|---------|
| id               | INTEGER |
| student_id       | INTEGER |
| attendance_date  | DATE    |
| time_in          | TIME    |
| status           | TEXT    |
| confidence_score | REAL    |

---

## 🔒 Security Notes

- Passwords hashed with **Werkzeug** `generate_password_hash` (PBKDF2-SHA256)
- File uploads validated by extension and size (max 5 MB)
- Duplicate attendance prevented per student per day
- All admin routes protected with `@login_required`
- Change `app.secret_key` and set via environment variable in production

---

## 📦 Sample Data

The database is automatically seeded with 8 sample students on first run:

| Student ID     | Name               | Course | Year | Section |
|----------------|--------------------|--------|------|---------|
| 2024-CS-001    | Maria Santos       | BSCS   | 3    | A       |
| 2024-CS-002    | Juan Dela Cruz     | BSCS   | 3    | A       |
| 2024-IT-001    | Ana Reyes          | BSIT   | 2    | B       |
| 2024-IT-002    | Carlo Mendoza      | BSIT   | 2    | B       |
| 2024-CE-001    | Sofia Garcia       | BSCE   | 1    | C       |
| 2024-CE-002    | Miguel Torres      | BSCE   | 1    | C       |
| 2024-BA-001    | Isabella Flores    | BSBA   | 4    | A       |
| 2024-BA-002    | Rafael Villanueva  | BSBA   | 4    | A       |
