"""
Parking Database - Separate database for parking lot management
"""
import sqlite3
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "parking.db")

def get_connection():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    # Vehicles entering/exiting parking
    conn.execute('''
        CREATE TABLE IF NOT EXISTS parking_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            license_plate TEXT,
            vehicle_type TEXT,
            color TEXT,
            slot_id TEXT,
            zone TEXT,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            entry_vehicle_img TEXT,
            entry_plate_img TEXT,
            exit_vehicle_img TEXT,
            exit_plate_img TEXT,
            fee REAL DEFAULT 0,
            status TEXT DEFAULT 'active',
            fraud_alert BOOLEAN DEFAULT 0,
            fraud_reason TEXT
        )
    ''')
    # Parking slots
    conn.execute('''
        CREATE TABLE IF NOT EXISTS parking_slots (
            id TEXT PRIMARY KEY,
            zone TEXT NOT NULL,
            status TEXT DEFAULT 'available',
            vehicle_type TEXT,
            current_plate TEXT,
            entry_time TIMESTAMP,
            fee REAL DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()

def insert_parking_entry(track_id, license_plate, vehicle_type, color, slot_id, zone, entry_vehicle_img, entry_plate_img):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO parking_entries (track_id, license_plate, vehicle_type, color, slot_id, zone, entry_time, entry_vehicle_img, entry_plate_img, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
    ''', (track_id, license_plate, vehicle_type, color, slot_id, zone, datetime.now(), entry_vehicle_img, entry_plate_img))
    entry_id = cur.lastrowid
    # Update slot status
    cur.execute('''
        UPDATE parking_slots SET status='occupied', vehicle_type=?, current_plate=?, entry_time=?
        WHERE id=?
    ''', (vehicle_type, license_plate, datetime.now(), slot_id))
    conn.commit()
    conn.close()
    return entry_id

def update_parking_exit(entry_id, exit_vehicle_img=None, exit_plate_img=None, fee=0):
    conn = get_connection()
    cur = conn.cursor()
    # Get entry info to free slot
    entry = conn.execute('SELECT slot_id, license_plate FROM parking_entries WHERE id=?', (entry_id,)).fetchone()
    cur.execute('''
        UPDATE parking_entries SET exit_time=?, exit_vehicle_img=?, exit_plate_img=?, fee=?, status='exited'
        WHERE id=?
    ''', (datetime.now(), exit_vehicle_img, exit_plate_img, fee, entry_id))
    if entry:
        # Free up the slot
        cur.execute('''
            UPDATE parking_slots SET status='available', vehicle_type=NULL, current_plate=NULL, entry_time=NULL, fee=0
            WHERE id=?
        ''', (entry['slot_id'],))
    conn.commit()
    conn.close()

def init_parking_slots():
    """Initialize default parking slots (A01-C06)"""
    conn = get_connection()
    existing = conn.execute('SELECT COUNT(*) FROM parking_slots').fetchone()[0]
    if existing == 0:
        zones = {'A': 6, 'B': 6, 'C': 6}
        for zone, count in zones.items():
            for i in range(1, count + 1):
                slot_id = f"{zone}{i:02d}"
                conn.execute('INSERT OR IGNORE INTO parking_slots (id, zone) VALUES (?, ?)', (slot_id, zone))
        conn.commit()
    conn.close()

def get_parking_stats():
    conn = get_connection()
    total = conn.execute('SELECT COUNT(*) FROM parking_entries WHERE status="active"').fetchone()[0]
    total_all = conn.execute('SELECT COUNT(*) FROM parking_entries').fetchone()[0]
    revenue = conn.execute('SELECT COALESCE(SUM(fee), 0) FROM parking_entries').fetchone()[0]
    fraud = conn.execute('SELECT COUNT(*) FROM parking_entries WHERE fraud_alert=1').fetchone()[0]
    conn.close()
    return {"active": total, "total": total_all, "revenue": revenue, "fraud": fraud}

def get_parking_entries(limit=100, offset=0):
    conn = get_connection()
    rows = conn.execute('SELECT * FROM parking_entries ORDER BY entry_time DESC LIMIT ? OFFSET ?', (limit, offset)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_parking_slots():
    conn = get_connection()
    rows = conn.execute('SELECT * FROM parking_slots ORDER BY zone, id').fetchall()
    conn.close()
    return [dict(r) for r in rows]