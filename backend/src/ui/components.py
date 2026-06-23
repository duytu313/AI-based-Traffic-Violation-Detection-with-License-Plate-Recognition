"""
UI Components - Các hàm vẽ UI Streamlit, sidebar, display cards
"""
import streamlit as st
import cv2
import time
import numpy as np
from typing import List, Tuple, Optional


def show_traffic_light_summary(recognizer, traffic_lights, red_light_violations):
    """Hiển thị tóm tắt trạng thái đèn giao thông."""
    active_light = "Đỏ" if recognizer.red_light_is_active(traffic_lights) else "Không đỏ"
    st.write(f"🚦 Trạng thái đèn: **{active_light}**")
    if traffic_lights:
        st.caption("Đèn phát hiện: " + ", ".join(
            f"{light['class_name']} → {light['state']} ({light['conf']*100:.1f}%)"
            for light in traffic_lights
        ))
    if red_light_violations:
        st.error(f"🚨 **Phát hiện {len(red_light_violations)} đối tượng vượt đèn đỏ!**")
        for item in red_light_violations:
            st.write(f"- {item['class_name']} ({item['conf']*100:.1f}%): {item['details']}")
    elif traffic_lights:
        st.success("Không phát hiện vượt đèn đỏ.")
    else:
        st.info("Chưa phát hiện đèn giao thông.")


def display_vehicle_card(vehicle_img, plate_img, plate_text, color, vtype, idx=0):
    """Hiển thị thông tin 1 xe (ảnh xe + biển số)."""
    caption = f"Xe {idx + 1}: {color} {vtype}"
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB),
                 caption=caption, use_container_width=True)
    with col_b:
        if plate_img is not None:
            st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB),
                     caption=f"Biển số: {plate_text}", use_container_width=True)
        else:
            st.write("Không có biển số")


def display_violation_detail(recognizer, violations):
    """Hiển thị chi tiết các vi phạm của 1 xe."""
    for vtype_v, details, _, conf in violations:
        icon = recognizer.get_violation_icon(vtype_v)
        st.error(f"{icon} {details} ({conf*100:.1f}%)")


def render_sidebar_controls(recognizer):
    """
    Render các control ở sidebar.
    Trả về dict chứa các giá trị cấu hình người dùng đã chọn.
    """
    st.sidebar.header("Chế độ")
    mode = st.sidebar.radio("Chọn chế độ", ("Image Upload", "Video Upload", "Webcam (local)", "RTSP / IP Camera"))

    display_fps = st.sidebar.checkbox("Hiển thị FPS", value=True)
    show_boxes = st.sidebar.checkbox("Hiển thị khung xe và biển số", value=True)
    max_items = st.sidebar.slider("Số xe tối đa hiển thị mỗi lần", 1, 10, 1)
    process_every_n_frame = st.sidebar.slider("Xử lý mỗi N frame (video)", 1, 30, 5)

    # --- Cấu hình phát hiện vi phạm ---
    st.sidebar.header("⚠️ Phát hiện vi phạm")
    enable_violation_detection = st.sidebar.checkbox("Bật phát hiện vi phạm", value=True)
    violation_conf_limit = st.sidebar.slider("Ngưỡng độ tin cậy vi phạm (%)", 10, 90, 15, 5)
    recognizer.set_violation_conf_limit(violation_conf_limit / 100.0)
    conf_more_than_two = st.sidebar.slider("Ngưỡng chở quá 2 người (%)", 10, 90, 50, 5)
    conf_no_helmet = st.sidebar.slider("Ngưỡng không mũ bảo hiểm (%)", 10, 90, 15, 5)
    conf_using_mobile = st.sidebar.slider("Ngưỡng sử dụng điện thoại (%)", 10, 90, 15, 5)
    recognizer.violation_conf_more_than_two = conf_more_than_two / 100.0
    recognizer.violation_conf_without_helmet = conf_no_helmet / 100.0
    recognizer.violation_conf_using_mobile = conf_using_mobile / 100.0

    st.sidebar.header("🚦 Vượt đèn đỏ")
    enable_red_light_detection = st.sidebar.checkbox("Bật phát hiện vượt đèn đỏ", value=False)
    traffic_light_conf = st.sidebar.slider("Ngưỡng nhận dạng đèn/người/xe (%)", 10, 90, 25, 5)
    recognizer.set_traffic_light_conf(traffic_light_conf / 100.0)
    st.sidebar.caption("Vượt đèn đỏ được phát hiện khi xe nằm trong vùng ROI được vẽ thủ công.")

    debug_vehicle = st.sidebar.checkbox("🔍 Debug phát hiện vi phạm", value=False)

    # --- Cấu hình Telegram ---
    st.sidebar.subheader("🤖 Thông báo Telegram")
    enable_telegram = st.sidebar.checkbox("Gửi thông báo Telegram khi phát hiện xe mới", value=True)
    enable_violation_telegram = st.sidebar.checkbox("Gửi thông báo vi phạm Telegram", value=True)

    from src.utils.notifications import send_test_telegram
    if st.sidebar.button("📨 Gửi tin nhắn test"):
        success, msg = send_test_telegram()
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(f"Lỗi: {msg}")

    return {
        "mode": mode,
        "display_fps": display_fps,
        "show_boxes": show_boxes,
        "max_items": max_items,
        "process_every_n_frame": process_every_n_frame,
        "enable_violation_detection": enable_violation_detection,
        "enable_red_light_detection": enable_red_light_detection,
        "debug_vehicle": debug_vehicle,
        "enable_telegram": enable_telegram,
        "enable_violation_telegram": enable_violation_telegram,
    }
