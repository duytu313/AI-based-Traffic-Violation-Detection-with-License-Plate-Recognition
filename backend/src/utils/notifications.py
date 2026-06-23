"""
Notifications - Logic gửi thông báo Telegram
"""
import os
import cv2
import requests
import numpy as np
from typing import Tuple

# Cấu hình Telegram (ẩn)
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8964633567:AAEsY1dXmxbc2Oa7pfoho5Af9Tdzo4RfCJo")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "6928155306")


def send_telegram_notification(vehicle_img: np.ndarray, plate_img, plate_text: str, vehicle_type: str, color: str) -> Tuple[bool, str]:
    """Gửi thông báo phát hiện phương tiện lên Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        return False, "Thiếu token hoặc chat_id"
    _, vehicle_encoded = cv2.imencode('.jpg', vehicle_img)
    vehicle_bytes = vehicle_encoded.tobytes()
    caption = f"🚗 PHÁT HIỆN PHƯƠNG TIỆN 🚗\n\nLoại xe: {vehicle_type}\nMàu sắc: {color}\nBiển số: {plate_text if plate_text else 'Không rõ'}"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {'photo': ('vehicle.jpg', vehicle_bytes, 'image/jpeg')}
    data = {'chat_id': CHAT_ID, 'caption': caption}
    try:
        r = requests.post(url, files=files, data=data)
        if not r.ok:
            return False, r.text
        result = r.json()
        if not result.get('ok'):
            return False, result.get('description', 'Lỗi không xác định')
    except Exception as e:
        return False, str(e)
    if plate_img is not None:
        _, plate_encoded = cv2.imencode('.jpg', plate_img)
        plate_bytes = plate_encoded.tobytes()
        files_plate = {'photo': ('plate.jpg', plate_bytes, 'image/jpeg')}
        data_plate = {'chat_id': CHAT_ID, 'caption': f'Biển số: {plate_text}'}
        try:
            requests.post(url, files=files_plate, data=data_plate)
        except:
            pass
    return True, "Đã gửi thông báo"


def send_violation_telegram(vehicle_img: np.ndarray, plate_text: str, violation_type: str, details: str) -> Tuple[bool, str]:
    """Gửi thông báo vi phạm lên Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        return False, "Thiếu token hoặc chat_id"
    _, vehicle_encoded = cv2.imencode('.jpg', vehicle_img)
    vehicle_bytes = vehicle_encoded.tobytes()

    violation_icon = get_violation_icon(violation_type)
    caption = f"{violation_icon} PHÁT HIỆN VI PHẠM {violation_icon}\n\nLoại: {details}\nBiển số: {plate_text if plate_text else 'Không rõ'}"

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    files = {'photo': ('violation.jpg', vehicle_bytes, 'image/jpeg')}
    data = {'chat_id': CHAT_ID, 'caption': caption}
    try:
        r = requests.post(url, files=files, data=data)
        if not r.ok:
            return False, r.text
    except Exception as e:
        return False, str(e)
    return True, "Đã gửi thông báo vi phạm"


def send_test_telegram() -> Tuple[bool, str]:
    """Gửi tin nhắn test kiểm tra kết nối Telegram."""
    if not BOT_TOKEN or not CHAT_ID:
        return False, "Thiếu token hoặc chat_id"
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': '✅ Kết nối thành công! Bot sẽ gửi ảnh khi phát hiện xe mới.'}
    try:
        r = requests.post(url, json=payload)
        if r.ok:
            return True, "Đã gửi tin nhắn test thành công!"
        else:
            return False, r.text
    except Exception as e:
        return False, str(e)


def get_violation_icon(vtype: str) -> str:
    """Trả về icon tương ứng với loại vi phạm."""
    if 'USING_MOBILE' in vtype:
        return "\U0001F4F1"  # 📱
    elif 'MORE_THAN_TWO_PERSONS' in vtype:
        return "\U0001F6F5"  # 🛵
    elif 'RED_LIGHT_VIOLATION' in vtype:
        return "\U0001F6A8"  # 🚨
    else:
        return "\u26D1\uFE0F"  # ⛑️