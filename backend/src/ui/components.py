"""
UI Components - Streamlit UI functions, sidebar, display cards
"""
import streamlit as st
import cv2
import time
import numpy as np
from typing import List, Tuple, Optional


def show_traffic_light_summary(recognizer, traffic_lights, red_light_violations):
    """Display traffic light status summary."""
    active_light = "Red" if recognizer.red_light_is_active(traffic_lights) else "Not red"
    st.write(f"🚦 Light status: **{active_light}**")
    if traffic_lights:
        st.caption("Detected lights: " + ", ".join(
            f"{light['class_name']} → {light['state']} ({light['conf']*100:.1f}%)"
            for light in traffic_lights
        ))
    if red_light_violations:
        st.error(f"🚨 **Detected {len(red_light_violations)} red light violations!**")
        for item in red_light_violations:
            st.write(f"- {item['class_name']} ({item['conf']*100:.1f}%): {item['details']}")
    elif traffic_lights:
        st.success("No red light violations detected.")
    else:
        st.info("No traffic lights detected.")


def display_vehicle_card(vehicle_img, plate_img, plate_text, color, vtype, idx=0):
    """Display information for 1 vehicle (vehicle image + license plate)."""
    caption = f"Vehicle {idx + 1}: {color} {vtype}"
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2RGB),
                 caption=caption, use_container_width=True)
    with col_b:
        if plate_img is not None:
            st.image(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB),
                     caption=f"License plate: {plate_text}", use_container_width=True)
        else:
            st.write("No license plate")


def display_violation_detail(recognizer, violations):
    """Display violation details for 1 vehicle."""
    for vtype_v, details, _, conf in violations:
        icon = recognizer.get_violation_icon(vtype_v)
        st.error(f"{icon} {details} ({conf*100:.1f}%)")


def render_sidebar_controls(recognizer):
    """
    Render sidebar controls.
    Returns dict containing user-selected configuration values.
    """
    st.sidebar.header("Mode")
    mode = st.sidebar.radio("Select mode", ("Image Upload", "Video Upload", "Webcam (local)", "RTSP / IP Camera"))

    display_fps = st.sidebar.checkbox("Display FPS", value=True)
    show_boxes = st.sidebar.checkbox("Display vehicle and plate boxes", value=True)
    max_items = st.sidebar.slider("Maximum vehicles to display per frame", 1, 10, 1)
    process_every_n_frame = st.sidebar.slider("Process every N frames (video)", 1, 30, 5)

    # --- Violation detection configuration ---
    st.sidebar.header("⚠️ Violation Detection")
    enable_violation_detection = st.sidebar.checkbox("Enable violation detection", value=True)
    violation_conf_limit = st.sidebar.slider("Violation confidence threshold (%)", 10, 90, 15, 5)
    recognizer.set_violation_conf_limit(violation_conf_limit / 100.0)
    conf_more_than_two = st.sidebar.slider("More than 2 people threshold (%)", 10, 90, 50, 5)
    conf_no_helmet = st.sidebar.slider("No helmet threshold (%)", 10, 90, 15, 5)
    conf_using_mobile = st.sidebar.slider("Phone usage threshold (%)", 10, 90, 15, 5)
    recognizer.violation_conf_more_than_two = conf_more_than_two / 100.0
    recognizer.violation_conf_without_helmet = conf_no_helmet / 100.0
    recognizer.violation_conf_using_mobile = conf_using_mobile / 100.0

    st.sidebar.header("🚦 Red Light Running")
    enable_red_light_detection = st.sidebar.checkbox("Enable red light running detection", value=False)
    traffic_light_conf = st.sidebar.slider("Light/person/vehicle detection threshold (%)", 10, 90, 25, 5)
    recognizer.set_traffic_light_conf(traffic_light_conf / 100.0)
    st.sidebar.caption("Red light running is detected when vehicle is inside the manually drawn ROI zone.")

    debug_vehicle = st.sidebar.checkbox("🔍 Debug violation detection", value=False)

    # --- Telegram configuration ---
    st.sidebar.subheader("🤖 Telegram Notifications")
    enable_telegram = st.sidebar.checkbox("Send Telegram notification when new vehicle detected", value=True)
    enable_violation_telegram = st.sidebar.checkbox("Send Telegram violation notification", value=True)

    from backend.src.utils.notifications import send_test_telegram
    if st.sidebar.button("📨 Send test message"):
        success, msg = send_test_telegram()
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(f"Error: {msg}")

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