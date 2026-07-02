"""
Main Database Manager - Coordinates all separate databases
Dashboard uses this to query across all modules
"""
import os
import sqlite3
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Import all separate database modules
from database_parking import (
    init_db as init_parking_db,
    get_parking_stats,
    get_parking_entries,
    get_parking_slots,
    init_parking_slots,
    insert_parking_entry,
    update_parking_exit,
)
from database_logistics import (
    init_db as init_logistics_db,
    get_logistics_stats,
    get_logistics_entries,
    get_unknown_alerts,
    get_truck_visits,
    insert_logistics_entry,
    update_logistics_exit,
    insert_unknown_alert,
)
from database_smartcity import (
    init_db as init_smartcity_db,
    get_smartcity_stats,
    get_traffic_flow,
    get_city_violations,
    get_flow_by_hour,
    insert_traffic_flow,
    insert_city_violation,
    insert_red_light_violation,
)
from database_test import (
    init_db as init_test_db,
    get_test_connection,
    test_insert_vehicle,
    test_insert_violation,
    test_get_vehicles,
    test_get_violations,
    test_get_stats,
    test_update_exit,
    test_check_fraud,
    insert_speed_violation,
    get_speed_violations,
    get_speed_violation_stats,
)


def init_all_databases():
    """Initialize all separate databases"""
    os.makedirs(DATA_DIR, exist_ok=True)
    init_parking_db()
    init_logistics_db()
    init_smartcity_db()
    init_test_db()
    init_parking_slots()


def get_global_stats():
    """Get combined stats from all databases for Dashboard"""
    parking = get_parking_stats()
    logistics = get_logistics_stats()
    smartcity = get_smartcity_stats()
    test = test_get_stats()

    return {
        "global": {
            "total_vehicles": parking["active"] + logistics["inside"] + smartcity["total_flow"] + test["total_vehicles"],
            "total_violations": smartcity["violations"] + test["total_violations"],
            "fraud_alerts": parking["fraud"] + logistics["fraud"],
            "total_revenue": parking.get("revenue", 0),
        },
        "parking": parking,
        "logistics": logistics,
        "smartcity": smartcity,
        "test": test,
    }


# For backward compatibility - test processing still uses original functions
# These are aliases pointing to test database functions
def init_db():
    init_all_databases()

# Test database functions (for image/video/webcam processing)
def insert_vehicle_entry(track_id, license_plate, vehicle_type, color, entry_vehicle_img, entry_plate_img):
    return test_insert_vehicle(track_id, license_plate, vehicle_type, color, entry_vehicle_img, entry_plate_img)

def update_exit_vehicle(vehicle_id, exit_vehicle_img=None, exit_plate_img=None):
    return test_update_exit(vehicle_id, exit_vehicle_img, exit_plate_img)

def insert_violation(vehicle_id, violation_type, details, image_path=None):
    return test_insert_violation(vehicle_id, violation_type, details, image_path)

def get_connection():
    return get_test_connection()

def check_fraud_on_exit(vehicle_id, license_plate, vehicle_type, color):
    return test_check_fraud(vehicle_id, license_plate, vehicle_type, color)

# Speed violation functions
def db_insert_speed_violation(vehicle_id, track_id, license_plate, vehicle_type, color, speed_kmh, speed_limit, image_path=None):
    return insert_speed_violation(vehicle_id, track_id, license_plate, vehicle_type, color, speed_kmh, speed_limit, image_path)

def db_get_speed_violations(limit=100, offset=0, license_plate=None):
    return get_speed_violations(limit, offset, license_plate)

def db_get_speed_violation_stats():
    return get_speed_violation_stats()
