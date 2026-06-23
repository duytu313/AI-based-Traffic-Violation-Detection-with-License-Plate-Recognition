"""
Test Database - Separate database for image/video/webcam processing testing
This is isolated from parking, logistics, and smartcity databases
"""
import sqlite3
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "test.db")

def get_test_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_test_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS test_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            license_plate TEXT,
            vehicle_type TEXT,
            color TEXT,
            fingerprint TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            entry_vehicle_img TEXT,
            entry_plate_img TEXT,
            exit_vehicle_img TEXT,
            exit_plate_img TEXT,
            fraud_alert BOOLEAN DEFAULT 0,
            fraud_reason TEXT
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS test_violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vehicle_id INTEGER,
            violation_type TEXT,
            timestamp TIMESTAMP,
            details TEXT,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def test_insert_vehicle(track_id, license_plate, vehicle_type, color, entry_vehicle_img, entry_plate_img):
    fingerprint = f"{license_plate}_{vehicle_type}_{color}".upper()
    conn = get_test_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO test_vehicles (track_id, license_plate, vehicle_type, color, fingerprint, entry_time, entry_vehicle_img, entry_plate_img)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (track_id, license_plate, vehicle_type, color, fingerprint, datetime.now(), entry_vehicle_img, entry_plate_img))
    vehicle_id = cur.lastrowid
    conn.commit()
    conn.close()
    return vehicle_id

def test_update_exit(vehicle_id, exit_vehicle_img=None, exit_plate_img=None):
    conn = get_test_connection()
    conn.execute('''
        UPDATE test_vehicles SET exit_time = ?, exit_vehicle_img = ?, exit_plate_img = ?
        WHERE id = ?
    ''', (datetime.now(), exit_vehicle_img, exit_plate_img, vehicle_id))
    conn.commit()
    conn.close()

def test_insert_violation(vehicle_id, violation_type, details, image_path=None):
    conn = get_test_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO test_violations (vehicle_id, violation_type, timestamp, details, image_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (vehicle_id, violation_type, datetime.now(), details, image_path))
    violation_id = cur.lastrowid
    conn.commit()
    conn.close()
    return violation_id

def test_get_vehicles(limit=100, offset=0, license_plate=None, start_time=None, end_time=None):
    conn = get_test_connection()
    query = "SELECT * FROM test_vehicles WHERE 1=1"
    params = []
    if license_plate:
        query += " AND license_plate LIKE ?"
        params.append(f"%{license_plate}%")
    if start_time:
        query += " AND DATE(entry_time) >= ?"
        params.append(start_time)
    if end_time:
        query += " AND DATE(entry_time) <= ?"
        params.append(end_time)
    query += " ORDER BY entry_time DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def test_get_violations(limit=100, offset=0, violation_type=None):
    conn = get_test_connection()
    query = """
        SELECT v.*, ve.license_plate, ve.vehicle_type, ve.color
        FROM test_violations v
        LEFT JOIN test_vehicles ve ON v.vehicle_id = ve.id
        WHERE 1=1
    """
    params = []
    if violation_type:
        query += " AND v.violation_type = ?"
        params.append(violation_type)
    query += " ORDER BY v.timestamp DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def test_get_stats():
    conn = get_test_connection()
    total_vehicles = conn.execute('SELECT COUNT(*) FROM test_vehicles').fetchone()[0]
    total_violations = conn.execute('SELECT COUNT(*) FROM test_violations').fetchone()[0]
    fraud = conn.execute('SELECT COUNT(*) FROM test_vehicles WHERE fraud_alert = 1').fetchone()[0]
    conn.close()
    return {"total_vehicles": total_vehicles, "total_violations": total_violations, "fraud_alerts": fraud}

def test_check_fraud(vehicle_id, license_plate, vehicle_type, color):
    conn = get_test_connection()
    entry = conn.execute('SELECT license_plate, vehicle_type, color FROM test_vehicles WHERE id = ?', (vehicle_id,)).fetchone()
    if not entry:
        conn.close()
        return False, None
    fingerprint_entry = f"{entry['license_plate']}_{entry['vehicle_type']}_{entry['color']}"
    fingerprint_exit = f"{license_plate}_{vehicle_type}_{color}"
    if fingerprint_entry != fingerprint_exit:
        reason = f"Không khớp: Vào ({entry['license_plate']}/{entry['vehicle_type']}/{entry['color']}) - Ra ({license_plate}/{vehicle_type}/{color})"
        conn.execute('UPDATE test_vehicles SET fraud_alert = 1, fraud_reason = ? WHERE id = ?', (reason, vehicle_id))
        conn.commit()
        conn.close()
        return True, reason
    conn.close()
    return False, None