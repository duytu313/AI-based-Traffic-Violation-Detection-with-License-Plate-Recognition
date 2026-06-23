"""
SmartCity Database - Separate database for urban traffic management
"""
import sqlite3
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "smartcity.db")

def get_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    # Traffic flow records
    conn.execute('''
        CREATE TABLE IF NOT EXISTS traffic_flow (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT,
            track_id INTEGER,
            license_plate TEXT,
            vehicle_type TEXT,
            color TEXT,
            detected_time TIMESTAMP,
            image_path TEXT,
            direction TEXT
        )
    ''')
    # Violations detected by city cameras
    conn.execute('''
        CREATE TABLE IF NOT EXISTS city_violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT,
            track_id INTEGER,
            license_plate TEXT,
            vehicle_type TEXT,
            color TEXT,
            violation_type TEXT,
            details TEXT,
            confidence REAL,
            detected_time TIMESTAMP,
            image_path TEXT,
            resolved BOOLEAN DEFAULT 0
        )
    ''')
    # Traffic light violations
    conn.execute('''
        CREATE TABLE IF NOT EXISTS red_light_violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera TEXT,
            track_id INTEGER,
            license_plate TEXT,
            vehicle_type TEXT,
            color TEXT,
            detected_time TIMESTAMP,
            image_path TEXT,
            fine_amount REAL DEFAULT 0
        )
    ''')
    # Hourly traffic statistics
    conn.execute('''
        CREATE TABLE IF NOT EXISTS hourly_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            hour INTEGER,
            camera TEXT,
            vehicle_count INTEGER DEFAULT 0,
            violation_count INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def insert_traffic_flow(camera, track_id, license_plate, vehicle_type, color, image_path=None, direction=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO traffic_flow (camera, track_id, license_plate, vehicle_type, color, detected_time, image_path, direction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (camera, track_id, license_plate, vehicle_type, color, datetime.now(), image_path, direction))
    flow_id = cur.lastrowid
    conn.commit()
    conn.close()
    return flow_id

def insert_city_violation(camera, track_id, license_plate, vehicle_type, color, violation_type, details, confidence, image_path=None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO city_violations (camera, track_id, license_plate, vehicle_type, color, violation_type, details, confidence, detected_time, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (camera, track_id, license_plate, vehicle_type, color, violation_type, details, confidence, datetime.now(), image_path))
    viol_id = cur.lastrowid
    conn.commit()
    conn.close()
    return viol_id

def insert_red_light_violation(camera, track_id, license_plate, vehicle_type, color, image_path=None, fine_amount=0):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO red_light_violations (camera, track_id, license_plate, vehicle_type, color, detected_time, image_path, fine_amount)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (camera, track_id, license_plate, vehicle_type, color, datetime.now(), image_path, fine_amount))
    viol_id = cur.lastrowid
    conn.commit()
    conn.close()
    return viol_id

def get_smartcity_stats():
    conn = get_connection()
    total_flow = conn.execute('SELECT COUNT(*) FROM traffic_flow').fetchone()[0]
    violations = conn.execute('SELECT COUNT(*) FROM city_violations').fetchone()[0]
    red_light = conn.execute('SELECT COUNT(*) FROM red_light_violations').fetchone()[0]
    unresolved = conn.execute('SELECT COUNT(*) FROM city_violations WHERE resolved=0').fetchone()[0]
    conn.close()
    return {"total_flow": total_flow, "violations": violations, "red_light": red_light, "unresolved": unresolved}

def get_traffic_flow(limit=100, offset=0):
    conn = get_connection()
    rows = conn.execute('SELECT * FROM traffic_flow ORDER BY detected_time DESC LIMIT ? OFFSET ?', (limit, offset)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_city_violations(limit=100, offset=0):
    conn = get_connection()
    rows = conn.execute('SELECT * FROM city_violations ORDER BY detected_time DESC LIMIT ? OFFSET ?', (limit, offset)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_flow_by_hour():
    conn = get_connection()
    rows = conn.execute('''
        SELECT hour, COUNT(*) as count FROM (
            SELECT CAST(strftime('%H', detected_time) AS INTEGER) as hour FROM traffic_flow
        ) GROUP BY hour ORDER BY hour
    ''').fetchall()
    conn.close()
    result = {i: 0 for i in range(24)}
    for r in rows:
        result[r['hour']] = r['count']
    return result