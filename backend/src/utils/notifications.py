"""
Notifications - Telegram notification sending logic
"""
import os
import cv2
import requests
import numpy as np
from typing import Tuple

# Telegram configuration (hidden)
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8964633567:AAEsY1dXmxbc2Oa7pfoho5Af9Tdzo4RfCJo")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "6928155306")


def send_telegram_notification(vehicle_img: np.ndarray, plate_img, plate_text: str, vehicle_type: str, color: str) -> Tuple[bool, str]:
    """Send vehicle detection notification to Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        return False, "Missing token or chat_id"
    _, vehicle_encoded = cv2.imencode('.jpg', vehicle_img)
    vehicle_bytes = vehicle_encoded.tobytes()
    caption = f"🚗 VEHICLE DETECTED 🚗\n\nVehicle type: {vehicle_type}\nColor: {color}\nLicense plate: {plate_text if plate_text else 'Unknown'}"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {'photo': ('vehicle.jpg', vehicle_bytes, 'image/jpeg')}
    data = {'chat_id': CHAT_ID, 'caption': caption}
    try:
        r = requests.post(url, files=files, data=data)
        if not r.ok:
            return False, r.text
        result = r.json()
        if not result.get('ok'):
            return False, result.get('description', 'Unknown error')
    except Exception as e:
        return False, str(e)
    if plate_img is not None:
        _, plate_encoded = cv2.imencode('.jpg', plate_img)
        plate_bytes = plate_encoded.tobytes()
        files_plate = {'photo': ('plate.jpg', plate_bytes, 'image/jpeg')}
        data_plate = {'chat_id': CHAT_ID, 'caption': f'License plate: {plate_text}'}
        try:
            requests.post(url, files=files_plate, data=data_plate)
        except:
            pass
    return True, "Notification sent"


def send_violation_telegram(vehicle_img: np.ndarray, plate_text: str, violation_type: str, details: str) -> Tuple[bool, str]:
    """Send violation notification to Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        return False, "Missing token or chat_id"
    _, vehicle_encoded = cv2.imencode('.jpg', vehicle_img)
    vehicle_bytes = vehicle_encoded.tobytes()

    violation_icon = get_violation_icon(violation_type)
    caption = f"{violation_icon} VIOLATION DETECTED {violation_icon}\n\nType: {details}\nLicense plate: {plate_text if plate_text else 'Unknown'}"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {'photo': ('violation.jpg', vehicle_bytes, 'image/jpeg')}
    data = {'chat_id': CHAT_ID, 'caption': caption}
    try:
        r = requests.post(url, files=files, data=data)
        if not r.ok:
            return False, r.text
    except Exception as e:
        return False, str(e)
    return True, "Violation notification sent"


def send_test_telegram() -> Tuple[bool, str]:
    """Send test message to check Telegram connection."""
    if not BOT_TOKEN or not CHAT_ID:
        return False, "Missing token or chat_id"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': '✅ Connection successful! Bot will send images when new vehicles are detected.'}
    try:
        r = requests.post(url, json=payload)
        if r.ok:
            return True, "Test message sent successfully!"
        else:
            return False, r.text
    except Exception as e:
        return False, str(e)


def get_violation_icon(vtype: str) -> str:
    """Return icon corresponding to violation type."""
    if 'USING_MOBILE' in vtype:
        return "\U0001F4F1"  # 📱
    elif 'MORE_THAN_TWO_PERSONS' in vtype:
        return "\U0001F6F5"  # 🛵
    elif 'RED_LIGHT_VIOLATION' in vtype:
        return "\U0001F6A8"  # 🚨
    else:
        return "\u26D1\uFE0F"  # ⛑️