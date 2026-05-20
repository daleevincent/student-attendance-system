# AttendX — Face Recognition Student Attendance System
### National University Philippines — Section COM221

A full-stack web application that automates student attendance for **Section COM221 (BSIT 2nd Year)** using FaceNet face recognition. Built with Flask + SQLite + Vanilla JavaScript.

---

## 📁 Project Structure

```
student-attendance-system/
├── app.py                           # Main Flask application & all routes
├── requirements.txt                 # Python dependencies
├── INSTALL.md                       # Detailed installation guide
│
├── database/
│   └── attendance.db                # SQLite database (auto-created on first run)
│
├── models/
│   └── 20180402-114759.pb           # FaceNet frozen graph — place here manually
│
├── services/
│   └── face_recognition_service.py  # FaceNet inference, enrollment, recognition
│
├── utils/
│   └── database.py                  # DB connection helper + schema initializer
│
├── static/
│   ├── css/
│   │   └── main.css                 # NU-themed CSS (dark/light mode)
│   ├── js/
│   │   └── main.js                  # Global JS (theme toggle, sidebar, toasts)
│   └── uploads/                     # Student face photos (auto-created)
│
└── templates/
    ├── layout.html                  # Base layout — NU navy sidebar + gold accents
    ├── login.html                   # Split-panel admin login
    ├── dashboard.html               # Stats, live clock, recent attendance
    ├── students.html                # Student CRUD — COM221 only
    ├── attendance.html              # Live webcam face scan page
    └── records.html                 # Attendance logs with date/name filter
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11** (recommended) — [Download here](https://www.python.org/downloads/release/python-3119/)
- The model file `20180402-114759.pb` placed in the `models/` folder

> ⚠️ Python 3.13 is **not supported** by TensorFlow. Use Python 3.11.

### 2. Create a virtual environment

```powershell
# Windows (PowerShell)
cd student-attendance-system
py -3.11 -m venv dale
dale\Scripts\activate
```

```bash
# macOS / Linux
python3.11 -m venv dale
source dale/bin/activate
```

### 3. Install dependencies

```powershell
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

> TensorFlow is ~500 MB — allow 3–5 minutes depending on internet speed.

### 4. Place the model file

```
student-attendance-system/
└── models/
    └── 20180402-114759.pb   ← copy here
```

### 5. Run the app

```powershell
python app.py
```

### 6. Open in browser

```
http://127.0.0.1:5000
```

> Use **Microsoft Edge** or **Chrome** — camera access works best on `127.0.0.1`.

### 7. Login

| Field    | Value      |
|----------|------------|
| Username | `admin`    |
| Password | `admin123` |

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔐 Admin Login | Session-based auth with hashed passwords, split-panel login UI |
| 📊 Dashboard | Live clock, stat cards (present/absent/total), today's attendance rate bar |
| 👤 Student Management | Add, edit, delete, search — scoped to COM221, no course/section fields needed |
| 📷 Face Scan | Webcam live feed, manual scan or auto-scan every 3s, session log |
| 📋 Attendance Logs | Filter by date or student name, pagination, CSV export |
| 🌙 Dark / Light Mode | NU navy dark theme by default, toggleable to light |
| 📱 Responsive | Mobile-friendly collapsible sidebar |

---

## 🤖 Face Recognition — How It Works

This system uses **FaceNet (InceptionResnetV1)** trained on MS-Celeb-1M via a frozen TensorFlow graph (`.pb` file). No dataset is required at runtime.

### Recognition pipeline

```
Student registered with photo
        ↓
Photo → FaceNet → 512-dim embedding stored in database
        ↓
Webcam frame → FaceNet → compare vs all stored embeddings (cosine similarity)
        ↓
Best match ≥ 0.60 threshold → Attendance marked ✅
```

### Model details

| Property | Value |
|----------|-------|
| File | `20180402-114759.pb` |
| Architecture | InceptionResnetV1 |
| Input node | `input:0` — shape `[batch, 160, 160, 3]`, normalized to `[-1, 1]` |
| Output node | `embeddings:0` — shape `[batch, 512]`, L2-normalized |
| Match threshold | `0.60` cosine similarity (adjustable in `face_recognition_service.py`) |

### Enrollment

When you add a student and upload their photo, the face embedding is computed and stored automatically. No manual steps needed.

### Tips for best accuracy

- Upload a **clear, well-lit, front-facing** photo for each student
- Student should look **directly at the camera** during scanning
- Avoid photos with sunglasses, masks, or heavy shadows
- Raise `COSINE_THRESHOLD` in `face_recognition_service.py` for stricter matching

---

## 🗄️ Database Schema

### `admins`
| Column | Type |
|--------|------|
| id | INTEGER PK |
| username | TEXT UNIQUE |
| password_hash | TEXT |

### `students`
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| student_number | TEXT UNIQUE | e.g. `2024-COM221-01` |
| full_name | TEXT | |
| course | TEXT | Fixed: `BSIT` |
| year_level | INTEGER | Fixed: `2` |
| section | TEXT | Fixed: `COM221` |
| image_path | TEXT | Filename in `static/uploads/` |
| face_embedding | BLOB | 512 × float32 = 2048 bytes |
| created_at | DATETIME | Auto |

### `attendance`
| Column | Type | Notes |
|--------|------|-------|
| id | INTEGER PK | |
| student_id | INTEGER FK | → students.id |
| attendance_date | DATE | e.g. `2024-11-20` |
| time_in | TIME | e.g. `08:15:32` |
| status | TEXT | `Present` or `Late` (after 08:30) |
| confidence_score | REAL | FaceNet cosine similarity 0–1 |

---

## 🎨 UI Theme

The interface uses the **National University Philippines** brand colors:

| Color | Hex | Usage |
|-------|-----|-------|
| NU Gold | `#F5A800` | Active nav, buttons, accents, clock |
| NU Navy Dark | `#001540` | Sidebar background, login banner |
| NU Navy | `#002366` | Datetime bar, gradients |

Font: **Inter** (Google Fonts)

---

## 🔒 Security Notes

- Passwords hashed with Werkzeug `generate_password_hash` (PBKDF2-SHA256)
- File uploads validated by extension (PNG, JPG, JPEG, GIF, WEBP) and size (max 5 MB)
- Duplicate attendance prevented per student per day
- All admin routes protected with `@login_required` decorator
- Change `app.secret_key` via environment variable before deploying

---

## 📦 Sample Data

On first run, 5 sample COM221 students are seeded automatically:

| Student ID | Name | Course | Year | Section |
|------------|------|--------|------|---------|
| 2024-COM221-01 | Maria Santos | BSIT | 2 | COM221 |
| 2024-COM221-02 | Juan Dela Cruz | BSIT | 2 | COM221 |
| 2024-COM221-03 | Ana Reyes | BSIT | 2 | COM221 |
| 2024-COM221-04 | Carlo Mendoza | BSIT | 2 | COM221 |
| 2024-COM221-05 | Sofia Garcia | BSIT | 2 | COM221 |

> These are placeholder names. Delete them and add your real COM221 students with their actual photos to enable face recognition.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---------|-----|
| `numpy build error` | Run `pip install --upgrade pip` then retry |
| `tensorflow not found` | Make sure you're using Python 3.11, not 3.13 |
| `Camera not working` | Use Edge/Chrome on `http://127.0.0.1:5000`, allow camera in browser |
| `Face not recognized` | Check that student has a face photo uploaded (badge shows "Face Enrolled") |
| `No match / low confidence` | Re-upload a clearer, better-lit front-facing photo |
| `Module not found` | Activate venv first — you should see `(dale)` in prompt |
| `Model not found` | Copy `20180402-114759.pb` into the `models/` folder |