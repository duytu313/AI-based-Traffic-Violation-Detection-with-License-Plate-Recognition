"""
Logistics Database - Separate database for warehouse/industrial zone management
"""
import sqlite3
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "logistics.db")

def get_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    # Vehicles entering/exiting logistics zone
    conn.execute('''
        CREATE TABLE IF NOT EXISTS logistics_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            license_plate TEXT,
            vehicle_type TEXT,
            color TEXT,
            camera TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            entry_vehicle_img TEXT,
            entry_plate_img TEXT,
            exit_vehicle_img TEXT,
            exit_plate_img TEXT,
            status TEXT DEFAULT 'inside',
            is_unknown BOOLEAN DEFAULT 0,
            fraud_alert BOOLEAN DEFAULT 0,
            fraud_reason TEXT
        )
    ''')
    # Unknown vehicle alerts
    conn.execute('''
        CREATE TABLE IF NOT EXISTS unknown_vehicle_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            vehicle_type TEXT,
            color TEXT,
            camera TEXT,
            detected_time TIMESTAMP,
            image_path TEXT,
            resolved BOOLEAN DEFAULT 0,
            resolution TEXT
        )
    ''')
    # Truck visit records
    conn.execute('''
        CREATE TABLE IF NOT EXISTS truck_visits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            license_plate TEXT,
            vehicle_type TEXT,
            color TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            duration_hours REAL,
            cargo_notes TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_logistics_entry(track_id, license_plate, vehicle_type, color, camera, entry_vehicle_img, entry_plate_img):
    is_unknown = 1 if not license_plate or license_plate.strip() == "" else 0
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO logistics_entries (track_id, license_plate, vehicle_type, color, camera, entry_time, entry_vehicle_img, entry_plate_img, status, is_unknown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'inside', ?)
    ''', (track_id, license_plate, vehicle_type, color, camera, datetime.now(), entry_vehicle_img, entry_plate_img, is_unknown))
    entry_id = cur.lastrowid
    conn.commit()
    conn.close()
    return entry_id

def update_logistics_exit(entry_id, exit_vehicle_img=None, exit_plate_img=None):
    conn = get_connection()
    cur = conn.cursor()
    entry = conn.execute('SELECT * FROM logistics_entries WHERE id=?', (entry_id,)).fetchone()
    cur.execute('''
        UPDATE logistics_entries SET exit_time=?, exit_vehicle_img=?, exit_plate_img=?, status='exited'
        WHERE id=?
    ''', (datetime.now(), exit_vehicle_img, exit_plate_img, entry_id))
    # If it's a truck, record truck visit
    if entry and entry['vehicle_type'] and 'truck' in entry['vehicle_type'].lower():
        duration = 0
        if entry['entry_time']:
            duration = round((datetime.now() - datetime.fromisoformat(entry['entry_time'])).total_seconds() / 3600, 1)
        cur.execute('''
            INSERT INTO truck_visits (license_plate, vehicle_type, color, entry_time, exit_time, duration_hours)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (entry['license_plate'], entry['vehicle_type'], entry['color'], entry['entry_time'], datetime.now(), duration))
    conn.commit()
    conn.close()

def insert_unknown_alert(track_id, vehicle_type, color, camera, image_path=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO unknown_vehicle_alerts (track_id, vehicle_type, color, camera, detected_time, image_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (track_id, vehicle_type, color, camera, datetime.now(), image_path))
    alert_id = cur.lastrowid
    conn.commit()
    conn.close()
    return alert_id

def get_logistics_stats():
    conn = get_connection()
    inside = conn.execute('SELECT COUNT(*) FROM logistics_entries WHERE status="inside"').fetchone()[0]
    total = conn.execute('SELECT COUNT(*) FROM logistics_entries').fetchone()[0]
    unknown = conn.execute('SELECT COUNT(*) FROM unknown_vehicle_alerts WHERE resolved=0').fetchone()[0]
    trucks = conn.execute('SELECT COUNT(*) FROM truck_visits').fetchone()[0]
    fraud = conn.execute('SELECT COUNT(*) FROM logistics_entries WHERE fraud_alert=1').fetchone()[0]
    conn.close()
    return {"inside": inside, "total": total, "unknown_alerts": unknown, "truck_visits": trucks, "fraud": fraud}

def get_logistics_entries(limit=100, offset=0):
    conn = get_connection()
    rows = conn.execute('SELECT * FROM logistics_entries ORDER BY entry_time DESC LIMIT ? OFFSET ?', (limit, offset)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_unknown_alerts(limit=50, offset=0):
    conn = get_connection()
    rows = conn.execute('SELECT * FROM unknown_vehicle_alerts ORDER BY detected_time DESC LIMIT ? OFFSET ?', (limit, offset)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_truck_visits(limit=50, offset=0):
    conn = get_connection()
    rows = conn.execute('SELECT * FROM truck_visits ORDER BY entry_time DESC LIMIT ? OFFSET ?', (limit, offset)).fetchall()
    conn.close()
    return [dict(r) for r in rows]