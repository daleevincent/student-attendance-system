"""
app.py — AttendX Student Attendance System
Main Flask application entry point.

Run:
    python app.py
Then open:  http://127.0.0.1:5000
Default login: admin / admin123
"""

import os
import csv
import io
from datetime import date, datetime
from functools import wraps

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash, jsonify, Response
)
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

# Local helpers
from utils.database import get_db, init_db
from services.face_recognition_service import (
    predict_face, enroll_student_face,
    capture_face_sample, enroll_from_samples
)

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'attendx-secret-key-change-in-production')

UPLOAD_FOLDER   = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024   # 5 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('database', exist_ok=True)
os.makedirs('models',   exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(f):
    """Decorator: redirect to login if not authenticated.
    For API/AJAX routes (URL starts with /api/), returns JSON 401 instead of redirect.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'admin_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({'success': False, 'message': 'Session expired. Please refresh and log in again.'}), 401
            flash('Please log in to continue.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ── Auth routes ───────────────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'admin_id' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Username and password are required.', 'error')
            return render_template('login.html')

        db    = get_db()
        admin = db.execute(
            "SELECT * FROM admins WHERE username = ?", (username,)
        ).fetchone()
        db.close()

        if admin and check_password_hash(admin['password_hash'], password):
            session['admin_id']   = admin['id']
            session['admin_name'] = admin['username']
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid username or password.', 'error')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


# ── Dashboard ─────────────────────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    db    = get_db()
    today = date.today().isoformat()

    total_students   = db.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    today_attendance = db.execute(
        "SELECT COUNT(*) FROM attendance WHERE attendance_date = ?", (today,)
    ).fetchone()[0]
    total_records    = db.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]

    recent_attendance = db.execute("""
        SELECT a.time_in, s.full_name, s.student_number, s.course
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.attendance_date = ?
        ORDER BY a.time_in DESC
        LIMIT 8
    """, (today,)).fetchall()

    db.close()

    stats = {
        'total_students':   total_students,
        'today_attendance': today_attendance,
        'total_records':    total_records,
    }
    return render_template('dashboard.html', stats=stats, recent_attendance=recent_attendance)


# ── Student CRUD ──────────────────────────────────────────────────────────────
@app.route('/students')
@login_required
def students():
    db       = get_db()
    students = db.execute(
        "SELECT * FROM students ORDER BY created_at DESC"
    ).fetchall()
    db.close()
    return render_template('students.html', students=students)


# Section is fixed for this deployment
SECTION_NAME  = 'COM221'
COURSE_NAME   = 'BSCS'
YEAR_LEVEL    = 4

@app.route('/students/add', methods=['POST'])
@login_required
def add_student():
    student_number = request.form.get('student_number', '').strip()
    full_name      = request.form.get('full_name',      '').strip()

    # Fixed values — single section system
    course     = COURSE_NAME
    year_level = YEAR_LEVEL
    section    = SECTION_NAME

    # Basic validation
    if not all([student_number, full_name]):
        flash('Student number and full name are required.', 'error')
        return redirect(url_for('students'))

    # Handle optional profile photo — for display only, NOT used for face recognition
    # Face recognition requires burst webcam enrollment via the Enroll button
    image_path = None
    file = request.files.get('face_image')
    if file and file.filename and allowed_file(file.filename):
        filename   = secure_filename(f"{student_number}_{file.filename}")
        save_path  = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        image_path = filename

    db = get_db()
    try:
        db.execute("""
            INSERT INTO students (student_number, full_name, course, year_level, section, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (student_number, full_name, course, int(year_level), section, image_path))
        db.commit()
        if image_path:
            flash(f'Student "{full_name}" added with profile photo. Use the Enroll button to register their face for recognition.', 'success')
        else:
            flash(f'Student "{full_name}" added. Use the Enroll button to register their face for recognition.', 'success')
    except Exception as e:
        db.rollback()
        if 'UNIQUE' in str(e):
            flash(f'Student ID "{student_number}" already exists.', 'error')
        else:
            flash(f'Error adding student: {str(e)}', 'error')
    finally:
        db.close()

    return redirect(url_for('students'))


@app.route('/students/edit/<int:student_id>', methods=['POST'])
@login_required
def edit_student(student_id):
    student_number = request.form.get('student_number', '').strip()
    full_name      = request.form.get('full_name',      '').strip()

    # Fixed values — single section system
    course     = COURSE_NAME
    year_level = YEAR_LEVEL
    section    = SECTION_NAME

    if not all([student_number, full_name]):
        flash('Student number and full name are required.', 'error')
        return redirect(url_for('students'))

    db = get_db()
    # Check if a new image was uploaded
    file = request.files.get('face_image')
    if file and file.filename and allowed_file(file.filename):
        filename  = secure_filename(f"{student_number}_{file.filename}")
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        db.execute("""
            UPDATE students
            SET student_number=?, full_name=?, course=?, year_level=?, section=?, image_path=?
            WHERE id=?
        """, (student_number, full_name, course, int(year_level), section, filename, student_id))
    else:
        db.execute("""
            UPDATE students
            SET student_number=?, full_name=?, course=?, year_level=?, section=?
            WHERE id=?
        """, (student_number, full_name, course, int(year_level), section, student_id))

    db.commit()
    db.close()
    flash(f'Student "{full_name}" updated successfully!', 'success')
    return redirect(url_for('students'))


@app.route('/students/delete/<int:student_id>', methods=['POST'])
@login_required
def delete_student(student_id):
    db      = get_db()
    student = db.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    if student:
        db.execute("DELETE FROM students WHERE id = ?", (student_id,))
        db.commit()
        flash(f'Student "{student["full_name"]}" deleted.', 'success')
    else:
        flash('Student not found.', 'error')
    db.close()
    return redirect(url_for('students'))


# ── Face attendance page ───────────────────────────────────────────────────────
@app.route('/face-attendance')
@login_required
def face_attendance():
    return render_template('attendance.html')


# ── Face recognition API endpoint ────────────────────────────────────────────
@app.route('/api/recognize', methods=['POST'])
@login_required
def api_recognize():
    """
    Receives a webcam frame (multipart/form-data, field: 'frame'),
    runs face recognition, marks attendance, and returns JSON.
    Always returns JSON — never HTML — so the frontend catch() never fires.
    """
    db = None
    try:
        if 'frame' not in request.files:
            return jsonify({'success': False, 'message': 'No frame provided'}), 400

        frame_file  = request.files['frame']
        image_bytes = frame_file.read()

        if not image_bytes:
            return jsonify({'success': False, 'message': 'Empty frame'}), 400

        db     = get_db()
        result = predict_face(image_bytes, db)

        if not result['recognized']:
            db.close()
            return jsonify({
                'success':     False,
                'message':     result['message'],
                'faces_found': result['faces_found']
            })

        # Process EVERY matched face and mark attendance for each
        today    = date.today().isoformat()
        time_now = datetime.now().strftime('%H:%M:%S')
        late_cutoff = '08:30:00'

        recognized_students = []

        for match in result['matches']:
            student = db.execute(
                "SELECT * FROM students WHERE id = ?", (match['student_id'],)
            ).fetchone()

            if not student:
                continue

            # Duplicate check per student per day
            existing = db.execute("""
                SELECT id FROM attendance
                WHERE student_id = ? AND attendance_date = ?
            """, (student['id'], today)).fetchone()

            already_marked = bool(existing)

            if not existing:
                status = 'Late' if time_now > late_cutoff else 'Present'
                db.execute("""
                    INSERT INTO attendance
                    (student_id, attendance_date, time_in, status, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (student['id'], today, time_now, status, match['confidence']))
                db.commit()
            else:
                status = db.execute(
                    "SELECT status FROM attendance WHERE student_id = ? AND attendance_date = ?",
                    (student['id'], today)
                ).fetchone()['status']

            recognized_students.append({
                'student_number': student['student_number'],
                'full_name':      student['full_name'],
                'course':         student['course'],
                'year_level':     student['year_level'],
                'section':        student['section'],
                'image_path':     student['image_path'],
                'confidence':     match['confidence'],
                'time_in':        time_now,
                'status':         status,
                'already_marked': already_marked
            })

        db.close()
        return jsonify({
            'success':    True,
            'faces_found': result['faces_found'],
            'total_recognized': len(recognized_students),
            'message':    result['message'],
            'students':   recognized_students      # list — supports 1 or many
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        if db:
            try:
                db.close()
            except Exception:
                pass
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


# ── Attendance records ────────────────────────────────────────────────────────
@app.route('/attendance-records')
@login_required
def attendance_records():
    # Collect filter params
    search   = request.args.get('search', '').strip()
    page     = max(1, int(request.args.get('page', 1)))
    per_page = 20

    # Default date is TODAY — only override if admin explicitly picks another date
    today_str = date.today().isoformat()
    f_date    = request.args.get('date', '').strip()
    if not f_date:
        f_date = today_str   # always show today by default

    # Export CSV?
    if request.args.get('export') == 'csv':
        return export_csv(search, f_date)

    # Build query — always filter to COM221 section
    where_clauses = ["s.section = 'COM221'"]
    params        = []

    if search:
        where_clauses.append("(s.full_name LIKE ? OR s.student_number LIKE ?)")
        params += [f'%{search}%', f'%{search}%']

    # Always filter by date (today by default, or chosen date)
    where_clauses.append("a.attendance_date = ?")
    params.append(f_date)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    base_query = f"""
        SELECT a.id, a.attendance_date, a.time_in, a.status, a.confidence_score,
               s.full_name, s.student_number, s.course, s.year_level, s.section
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        {where_sql}
    """

    db          = get_db()
    total_count = db.execute(f"SELECT COUNT(*) FROM ({base_query})", params).fetchone()[0]
    total_pages = max(1, (total_count + per_page - 1) // per_page)
    page        = min(page, total_pages)
    offset      = (page - 1) * per_page

    records = db.execute(
        f"{base_query} ORDER BY a.attendance_date DESC, a.time_in DESC LIMIT ? OFFSET ?",
        params + [per_page, offset]
    ).fetchall()

    db.close()

    filters = {'search': search, 'date': f_date}
    return render_template(
        'records.html',
        records=records,
        filters=filters,
        page=page,
        per_page=per_page,
        total_count=total_count,
        total_pages=total_pages,
        today=today_str
    )


def export_csv(search='', f_date=''):
    """Build and stream a CSV file of attendance records."""
    where_clauses = ["s.section = 'COM221'"]
    params        = []

    if search:
        where_clauses.append("(s.full_name LIKE ? OR s.student_number LIKE ?)")
        params += [f'%{search}%', f'%{search}%']
    if f_date:
        where_clauses.append("a.attendance_date = ?")
        params.append(f_date)

    where_sql = "WHERE " + " AND ".join(where_clauses)

    db      = get_db()
    records = db.execute(f"""
        SELECT a.id, s.student_number, s.full_name, s.course,
               s.year_level, s.section,
               a.attendance_date, a.time_in, a.status, a.confidence_score
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        {where_sql}
        ORDER BY a.attendance_date DESC, a.time_in DESC
    """, params).fetchall()
    db.close()

    output  = io.StringIO()
    writer  = csv.writer(output)
    writer.writerow([
        'Attendance ID', 'Student Number', 'Full Name', 'Course',
        'Year Level', 'Section', 'Date', 'Time In', 'Status', 'Confidence Score'
    ])
    for r in records:
        conf = f"{r['confidence_score']:.2%}" if r['confidence_score'] else ''
        writer.writerow([
            r['id'], r['student_number'], r['full_name'], r['course'],
            r['year_level'], r['section'],
            r['attendance_date'], r['time_in'], r['status'], conf
        ])

    filename = f"attendance_{date.today().isoformat()}.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


# ── Burst enrollment API ──────────────────────────────────────────────────────

@app.route('/api/enroll/capture', methods=['POST'])
@login_required
def api_enroll_capture():
    """
    Receive a single webcam frame during burst enrollment.
    Detects + crops the face and saves it as a sample.
    """
    try:
        student_id = request.form.get('student_id', type=int)
        if not student_id:
            return jsonify({'success': False, 'message': 'student_id required'}), 400

        if 'frame' not in request.files:
            return jsonify({'success': False, 'message': 'No frame provided'}), 400

        image_bytes = request.files['frame'].read()
        if not image_bytes:
            return jsonify({'success': False, 'message': 'Empty frame'}), 400

        # Save sample to uploads/samples/<student_id>/
        sample_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'samples', str(student_id))
        result     = capture_face_sample(image_bytes, student_id, sample_dir)

        if not result['success']:
            return jsonify(result), 400

        # Record in DB
        db = get_db()
        db.execute(
            "INSERT INTO face_samples (student_id, image_path) VALUES (?, ?)",
            (student_id, result['filename'])
        )
        db.commit()
        count = db.execute(
            "SELECT COUNT(*) as n FROM face_samples WHERE student_id = ?",
            (student_id,)
        ).fetchone()['n']
        db.close()

        return jsonify({
            'success':       True,
            'face_detected': result['face_detected'],
            'filename':      result['filename'],
            'total_samples': count,
            'message':       result['message']
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/enroll/finish', methods=['POST'])
@login_required
def api_enroll_finish():
    """
    Called after burst capture is complete.
    Averages all captured samples into a single robust embedding.
    """
    try:
        data       = request.get_json() or {}
        student_id = data.get('student_id')
        if not student_id:
            return jsonify({'success': False, 'message': 'student_id required'}), 400

        db     = get_db()
        result = enroll_from_samples(int(student_id), db)
        db.close()
        return jsonify(result)

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/enroll/clear', methods=['POST'])
@login_required
def api_enroll_clear():
    """Delete all captured samples for a student (e.g. to re-do enrollment)."""
    try:
        data       = request.get_json() or {}
        student_id = data.get('student_id')
        if not student_id:
            return jsonify({'success': False, 'message': 'student_id required'}), 400

        db = get_db()
        db.execute("DELETE FROM face_samples WHERE student_id = ?", (student_id,))
        db.commit()
        db.close()

        # Also delete saved sample images
        import shutil
        sample_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'samples', str(student_id))
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)

        return jsonify({'success': True, 'message': 'Samples cleared'})

    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template('login.html'), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'message': 'File too large (max 5 MB)'}), 413


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    init_db()          # create tables + seed default data on first run
    app.run(debug=True, host='0.0.0.0', port=5000)