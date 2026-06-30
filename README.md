# 🚦 Traffic AI - Smart Traffic Management System

**Traffic AI** is an AI-powered traffic violation detection and license plate recognition system (YOLOv8 + FastPlateOCR), combining **FastAPI backend** and **Next.js 16 frontend**.

The system supports **4 business modules**:
- 🚦 **Traffic Monitoring** – Traffic surveillance, violation detection
- 🅿️ **Parking Management** – Vehicle entry/exit management
- 🚛 **Logistics** – Warehouse monitoring, unknown vehicle detection
- 🏙️ **SmartCity** – Multi-camera urban traffic monitoring

---

## 🏗️ Architecture Overview

```
├── backend/                        # FastAPI Backend (Python)
│   ├── main.py                     # Main API server (1927 lines)
│   ├── requirements.txt            # Python dependencies
│   ├── database.py                 # SQLite database (traffic)
│   ├── database_parking.py         # Parking database
│   ├── database_logistics.py       # Logistics database
│   ├── database_smartcity.py       # SmartCity database
│   ├── database_test.py            # Test database
│   ├── data/
│   │   └── traffic.db              # SQLite file
│   ├── files_model/                # YOLO model weights
│   └── src/
│       ├── core/
│       │   ├── engine.py           # License plate recognizer, violations
│       │   ├── tracker.py          # Object tracking (ByteTrack)
│       │   ├── zones.py            # Zone-based red light config
│       │   ├── roi_detector.py     # ROI polygon violation detection
│       │   └── birds_eye_detector.py  # Bird's Eye View 3D red light
│       ├── utils/
│       │   ├── image_utils.py      # Image processing utilities
│       │   └── notifications.py    # Telegram notifications
│       └── ui/
│           └── components.py       # Streamlit components (legacy)
│
├── frontend/                       # Next.js 16 Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx          # Root layout
│   │   │   ├── page.tsx            # Main page with 9 tabs
│   │   │   └── globals.css         # Dark theme styles
│   │   ├── components/
│   │   │   ├── Header.tsx          # Header with API status
│   │   │   ├── Tabs.tsx            # Tab navigation
│   │   │   ├── ConfigPanel.tsx     # Detection config controls
│   │   │   ├── ImageProcessor.tsx  # Image upload & processing
│   │   │   ├── VideoProcessor.tsx  # Video processing
│   │   │   ├── WebcamProcessor.tsx # Webcam/RTSP streaming
│   │   │   ├── BEVEditor.tsx       # Bird's Eye View config editor
│   │   │   ├── ROIEditor.tsx       # Region of Interest editor
│   │   │   ├── Dashboard.tsx       # Analytics dashboard
│   │   │   ├── DatabaseView.tsx    # Database viewer
│   │   │   ├── Settings.tsx        # Telegram configuration
│   │   │   ├── ParkingManager.tsx  # Parking management UI
│   │   │   ├── LogisticsOperations.tsx  # Logistics UI
│   │   │   └── SmartCityDashboard.tsx   # SmartCity UI
│   │   └── lib/
│   │       ├── api.ts              # API client (axios)
│   │       └── types.ts            # TypeScript definitions
│   ├── .env.local                  # Environment config
│   └── package.json
│
├── files_model/                    # YOLO model weights
│   ├── helmet.pt                   # Violation detection (helmet, overload, phone)
│   ├── license_plate_detector.pt   # License plate detection
│   ├── traffic_light.pt            # Traffic light detection
│   ├── vehicle_color_n_cls.pt     # Vehicle color classification
│   ├── yolov8n.pt                 # Base YOLOv8 nano (vehicle detection)
│   └── yolov8n-cls.pt            # YOLOv8 classification
│
└── data/                           # Runtime data
    ├── traffic.db                  # SQLite database
    ├── plates/                     # Saved plate images
    ├── vehicles/                   # Saved vehicle images
    └── evidence/                   # Violation evidence images
```

---

## ✨ Detailed Features

### 🚦 Traffic Monitoring

| Feature | Description |
|---------|-------------|
| ✅ **Vehicle Detection** | Cars, motorcycles, buses, trucks (YOLOv8 COCO) |
| ✅ **License Plate Recognition** | Automatic ANPR/LPR (YOLO + FastPlateOCR `cct-s-v2-global-model`) |
| ✅ **Vehicle Color Classification** | 15 different colors |
| ✅ **Violation Detection** | No helmet, carrying more than 2 people, using phone |
| ✅ **Red Light Violation Detection** | 3 methods: Zone-based, ROI Polygon, Bird's Eye View 3D |
| ✅ **Object Tracking** | ByteTrack for real-time video |
| ✅ **Telegram Notifications** | Instant violation alerts with images |
| ✅ **Static Image Processing** | Upload image, analyze, display results |
| ✅ **Video Processing** | Upload video, streaming with bounding boxes |
| ✅ **Webcam/RTSP** | Real-time streaming from IP camera or webcam |

### 🅿️ Parking Management

| Feature | Description |
|---------|-------------|
| ✅ **Entry/Exit Gate Cameras** | 2 separate camera streams |
| ✅ **License Plate Recognition** | Automatic recognition on entry/exit |
| ✅ **Parking Time Calculation** | Calculate duration and parking fee |
| ✅ **Fraud Alerts** | Detect duplicate plates, unknown vehicles |
| ✅ **Statistics** | Vehicles in lot, entry/exit history |

### 🚛 Logistics

| Feature | Description |
|---------|-------------|
| ✅ **Gate + Construction Site Cameras** | 2 monitoring camera streams |
| ✅ **Vehicle Entry/Exit Recognition** | Automatic logging of trucks, cargo vehicles |
| ✅ **Unknown Vehicle Detection** | Alert when vehicle has no plate or unreadable |
| ✅ **Truck Visit Tracking** | History of truck entry/exit trips |

### 🏙️ SmartCity

| Feature | Description |
|---------|-------------|
| ✅ **Multi-Camera** | Support up to 4 monitoring cameras |
| ✅ **Traffic Flow Analysis** | Hourly vehicle flow measurement |
| ✅ **Violation Detection** | Aggregate violations from multiple cameras |
| ✅ **Urban Statistics** | City traffic overview dashboard |

---

## 🧠 Red Light Violation Detection Methods

### 1. Zone-Based Detection
Analyzes **Waiting → Stop → Intersection** zones based on Y coordinates. Used for angled cameras with direction `"down"` or `"up"`.

### 2. ROI Polygon Detection
User draws ROI polygon on image, system detects vehicles in ROI when red light.

### 3. Bird's Eye View 3D (BEV)
Converts perspective to **top-down view**, draws 3D stop line in real space. Detects vehicles crossing boundary when red light (similar to Colab implementation).

---

## 🚀 Installation & Setup

### System Requirements
- **Python** 3.10+
- **Node.js** 18+
- **npm** 9+
- **pip** (Python package manager)

### 1. Backend (FastAPI)

```bash
# Activate virtual environment (Windows PowerShell)
venv\Scripts\Activate.ps1

# Or (CMD)
venv\Scripts\activate.bat

cd backend
pip install -r requirements.txt
python main.py
```

Backend runs at **http://localhost:8000** with Swagger docs at **http://localhost:8000/docs**

### 2. Frontend (Next.js 16)

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at **http://localhost:3000**

### 3. Environment Configuration

Create file `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 📡 API Endpoints

### Core (Traffic Monitoring)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Check API connection |
| POST | `/api/process-image` | Process image: detect vehicles, plates, violations |
| POST | `/api/process-video` | Process video file frame-by-frame |
| GET | `/api/vehicles` | Vehicle list (paginated) |
| GET | `/api/violations` | Violation list |
| GET | `/api/stats` | Overall statistics |
| GET | `/api/fraud` | Fraud alerts |

### Red Light Detection

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/zone-config` | Update zone-based detection config |
| GET | `/api/zone-config` | Get current zone config |
| GET | `/api/zone-stats` | Zone detector statistics |
| POST | `/api/zone-reset` | Reset zone detector |
| POST | `/api/roi/set` | Set ROI zone |
| GET | `/api/roi/get` | Get ROI config |
| POST | `/api/roi/clear` | Clear ROI zone |
| POST | `/api/roi/detect` | Detect violations in ROI |
| POST | `/api/bev/config` | Update BEV config |
| GET | `/api/bev/config` | Get BEV config |
| POST | `/api/bev/reset` | Reset BEV detector |
| GET | `/api/bev/stats` | BEV statistics |

### Webcam / RTSP

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/webcam/start` | Start webcam/RTSP stream |
| POST | `/api/webcam/stop` | Stop stream |
| GET | `/api/webcam/frame` | Get current frame (JPEG) |
| GET | `/api/webcam/status` | Stream status |
| GET | `/api/webcam/detected` | List of detected vehicles |

### Video Streaming

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/video/stream` | Upload video & start stream |
| POST | `/api/video/stop` | Stop video stream |
| GET | `/api/video/frame` | Get current frame (JPEG) |
| GET | `/api/video/status` | Video stream status |
| GET | `/api/video/detected` | List of detected vehicles |

### Parking

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/parking/stats` | Parking statistics |
| GET | `/api/parking/entries` | List of entered vehicles |
| GET | `/api/parking/slots` | Parking slot status |
| POST | `/api/parking/{gate}/start` | Start gate camera (entry/exit) |
| POST | `/api/parking/{gate}/stop` | Stop gate camera |
| GET | `/api/parking/{gate}/frame` | Frame from gate camera |
| GET | `/api/parking/{gate}/status` | Gate camera status |
| GET | `/api/parking/{gate}/detected` | Vehicles detected at gate |

### Logistics

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/logistics/stats` | Logistics statistics |
| GET | `/api/logistics/entries` | Vehicle list |
| GET | `/api/logistics/unknown-alerts` | Unknown vehicle alerts |
| GET | `/api/logistics/truck-visits` | Truck visits |
| POST | `/api/logistics/{camera}/start` | Start camera (gate/construction_site) |
| POST | `/api/logistics/{camera}/stop` | Stop camera |
| GET | `/api/logistics/{camera}/frame` | Frame from camera |
| GET | `/api/logistics/{camera}/status` | Camera status |
| GET | `/api/logistics/{camera}/detected` | Detected vehicles |

### SmartCity

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/smartcity/stats` | Urban statistics |
| GET | `/api/smartcity/flow` | Traffic flow |
| GET | `/api/smartcity/violations` | City-wide violations |
| GET | `/api/smartcity/flow-by-hour` | Hourly traffic flow |
| POST | `/api/smartcity/{camera}/start` | Start camera (cam1-cam4) |
| POST | `/api/smartcity/{camera}/stop` | Stop camera |
| GET | `/api/smartcity/{camera}/frame` | Frame from camera |
| GET | `/api/smartcity/{camera}/status` | Camera status |
| GET | `/api/smartcity/{camera}/detected` | Detected vehicles |

### Image Processing Example

```bash
curl -X POST http://localhost:8000/api/process-image \
  -F "file=@image.jpg" \
  -F "enable_violation_detection=true" \
  -F "enable_red_light_detection=false" \
  -F "enable_bev_detection=true" \
  -F "violation_conf_limit=0.15" \
  -F "conf_no_helmet=0.15" \
  -F "conf_using_mobile=0.15" \
  -F "conf_more_than_two=0.50" \
  -F "traffic_light_conf=0.25" \
  -F "debug=false" \
  -F "show_zones=false" \
  -F "show_bev=true" \
  -F "camera_direction=down"
```

---

## ⚙️ Configuration

### Backend
- **Detection thresholds**: Adjustable via form data when calling API
- **Telegram**: Configure token in `backend/src/utils/notifications.py`
- **Model paths**: Default in `backend/files_model/`
- **CORS**: Allows all origins (can be restricted in `main.py`)

### Frontend
- **API URL**: Configure in `frontend/.env.local`
- **Detection config**: UI in **Image Processing** tab (ConfigPanel)
- **ROI Editor**: Draw detection zone directly on image
- **BEV Editor**: Configure Bird's Eye View 3D

---

## 🧠 AI Models

| Model | File | Purpose |
|-------|------|---------|
| **YOLOv8n** | `yolov8n.pt` | Vehicle detection (car, motorcycle, bus, truck) |
| **License Plate** | `license_plate_detector.pt` | License plate detection |
| **Helmet** | `helmet.pt` | Violation detection: no helmet, overloading, phone use |
| **Traffic Light** | `traffic_light.pt` | Traffic light detection (red/yellow/green) |
| **Color Classification** | `vehicle_color_n_cls.pt` | Vehicle color classification |
| **Classification** | `yolov8n-cls.pt` | YOLOv8 classification backbone |
| **FastPlateOCR** | (hub model: `cct-s-v2-global-model`) | OCR engine for license plate character recognition |

---

## 🖥️ Frontend Interface

The system has **9 functional tabs**:

1. **Dashboard** 📊 – Overview statistics from all modules
2. **Parking** 🅿️ – Parking lot management (2 entry/exit cameras)
3. **Logistics** 🚛 – Warehouse monitoring (2 gate/construction_site cameras)
4. **SmartCity** 🏙️ – Urban monitoring (4 cameras cam1-cam4)
5. **Image Processing** 📸 – Upload image, configure detection, view results
6. **Video Processing** 🎥 – Upload video, real-time streaming
7. **Webcam** 📹 – Webcam / RTSP streaming
8. **Database** 🗄️ – View vehicle, violation data
9. **Settings** ⚙️ – Telegram configuration, data refresh

---

## 📦 Database

The system uses **SQLite** with separate databases for each module:

| Database | File | Purpose |
|----------|------|---------|
| Traffic | `data/traffic.db` | Vehicles, traffic violations |
| Parking | `database_parking.py` | Vehicle entry/exit from parking lot |
| Logistics | `database_logistics.py` | Warehouse vehicle entry/exit, unknown vehicle alerts |
| SmartCity | `database_smartcity.py` | Traffic flow, urban violations |
| Test | `database_test.py` | Test data from image/video processing |

---

## 🔔 Telegram Notifications

The system automatically sends Telegram notifications when detecting:
- ✅ New vehicles with license plates
- 🚨 Traffic violations (with captured images)

Configure in `backend/src/utils/notifications.py`:
```python
BOT_TOKEN = "your_bot_token"
CHAT_ID = "your_chat_id"
```

---

## 📝 Notes

- When using **ROI detection**, you need to set up the ROI polygon first via the ROI Editor tab
- **Bird's Eye View (BEV)** requires configuration of 4 source points (src_points) and 4 destination points (dst_points)
- For **zone-based detection**, you need to adjust Y coordinates of zones to match camera angle
- Image processing results are returned as **base64** for direct frontend display
- All camera streaming supports both **webcam** (`source=0`) and **RTSP URL**

---

## 📄 License

Project developed by [duytu313](https://github.com/duytu313) for research and learning purposes in computer vision and AI for intelligent transportation.
