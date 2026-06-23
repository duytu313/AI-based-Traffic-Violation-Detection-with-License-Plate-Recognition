# 🚦 Traffic AI - Hệ thống quản lý giao thông thông minh

Hệ thống phát hiện vi phạm giao thông và nhận dạng biển số xe sử dụng AI (YOLOv8 + OCR).

## 🏗️ Kiến trúc

```
├── backend/                 # FastAPI Backend (Python)
│   ├── main.py             # API server
│   ├── requirements.txt    # Python dependencies
│   ├── database.py         # SQLite database
│   ├── data/               # Database & evidences
│   ├── files_model/        # YOLO model files
│   └── src/                # Core AI logic
│       ├── core/
│       │   ├── engine.py    # License plate recognizer, violations
│       │   └── tracker.py   # Object tracking (ByteTrack)
│       ├── utils/
│       │   ├── image_utils.py    # Image processing utilities
│       │   └── notifications.py  # Telegram notifications
│       └── ui/
│           └── components.py     # Streamlit components (legacy)
│
├── frontend/                # Next.js 16 Frontend
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx       # Root layout
│   │   │   ├── page.tsx         # Main page with tabs
│   │   │   └── globals.css      # Global styles (dark theme)
│   │   ├── components/
│   │   │   ├── Header.tsx       # Header with API status
│   │   │   ├── Tabs.tsx         # Tab navigation
│   │   │   ├── ConfigPanel.tsx  # Detection config controls
│   │   │   ├── ImageProcessor.tsx  # Image upload & processing
│   │   │   ├── VideoProcessor.tsx  # Video processing (WIP)
│   │   │   ├── Dashboard.tsx    # Analytics dashboard
│   │   │   ├── DatabaseView.tsx # Database viewer
│   │   │   └── Settings.tsx     # Telegram configuration
│   │   └── lib/
│   │       ├── api.ts          # API client (axios)
│   │       └── types.ts        # TypeScript definitions
│   ├── .env.local              # Environment config
│   └── package.json
│
├── data/                    # Runtime data
│   ├── traffic.db          # SQLite database
│   ├── plates/             # Saved plate images
│   ├── vehicles/           # Saved vehicle images
│   └── evidence/           # Violation evidence images
│
├── files_model/            # YOLO model weights
│   ├── helmet.pt
│   ├── license_plate_detector.pt
│   ├── light_traffic.pt
│   └── vehicle_color_n_cls.pt
│
└── yolov8n.pt              # Base YOLOv8 model
```

## ✨ Tính năng

| Tính năng | Mô tả |
|-----------|-------|
| ✅ **Phát hiện phương tiện** | Ô tô, xe máy, xe buýt, xe tải (YOLOv8) |
| ✅ **Nhận dạng biển số** | ANPR/LPR tự động (YOLO + FastPlateOCR) |
| ✅ **Phân loại màu xe** | 15 màu sắc khác nhau |
| ✅ **Phát hiện vi phạm** | Không mũ bảo hiểm, chở quá 2 người, dùng điện thoại |
| ✅ **Phát hiện vượt đèn đỏ** | Phân tích trạng thái đèn giao thông |
| ✅ **Tracking đối tượng** | ByteTrack cho video real-time |
| ✅ **Thông báo Telegram** | Gửi ảnh và thông tin vi phạm |
| ✅ **Cơ sở dữ liệu** | Lưu trữ lịch sử phương tiện và vi phạm |
| ✅ **Dashboard thống kê** | Tổng quan hệ thống |

## 🚀 Cài đặt & Chạy

### Yêu cầu
- Python 3.10+
- Node.js 18+
- npm 9+

### 1. Backend (Python)

```bash
venv\Scripts\Activate.ps1
cd backend
pip install -r requirements.txt
python main.py
```

Backend chạy tại **http://localhost:8000**

### 2. Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Frontend chạy tại **http://localhost:3000**

## 📡 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/api/health` | Kiểm tra kết nối |
| POST | `/api/process-image` | Xử lý ảnh: phát hiện xe, biển số, vi phạm |
| GET | `/api/vehicles` | Danh sách phương tiện (có phân trang) |
| GET | `/api/violations` | Danh sách vi phạm |
| GET | `/api/stats` | Thống kê tổng quan |
| GET | `/api/fraud` | Cảnh báo gian lận |

### Xử lý ảnh

```bash
curl -X POST http://localhost:8000/api/process-image \
  -F "file=@image.jpg" \
  -F "enable_violation_detection=true" \
  -F "enable_red_light_detection=false"
```

## ⚙️ Cấu hình

### Backend
- Các ngưỡng phát hiện được truyền qua form data khi gọi API
- Có thể cấu hình Telegram token trong `src/utils/notifications.py`

### Frontend
- Cấu hình API URL trong `frontend/.env.local`
- Các ngưỡng vi phạm điều chỉnh qua UI trong tab **Xử lý ảnh**

## 🧠 Models

- `yolov8n.pt` - YOLOv8 nano cho phát hiện phương tiện
- `license_plate_detector.pt` - Phát hiện biển số
- `helmet.pt` - Phát hiện vi phạm (mũ bảo hiểm, chở quá người, điện thoại)
- `light_traffic.pt` - Phát hiện đèn giao thông
- `vehicle_color_n_cls.pt` - Phân loại màu xe
- `fast-plate-ocr` - OCR engine (hub model: cct-s-v2-global-model)

## 📝 Ghi chú

- Tính năng **xử lý video** đang được phát triển (tab hiển thị UI placeholder)
- Với **vượt đèn đỏ**, cần bật checkbox trong cấu hình
- Ảnh kết quả được trả về dưới dạng base64 để hiển thị trực tiếp


