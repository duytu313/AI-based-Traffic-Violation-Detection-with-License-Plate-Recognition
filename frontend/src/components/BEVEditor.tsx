"use client";

import { useState, useRef, useEffect, useCallback } from "react";

interface Point {
  x: number;
  y: number;
}

interface BEVEditorProps {
  imageUrl: string | null;
  onSave: (srcPoints: Point[]) => void;
  onCancel: () => void;
  initialPoints?: Point[];
}

const STEPS = [
  "👉 Step 1: Click point 1 (BOTTOM - LEFT) near the road edge close to camera",
  "👉 Step 2: Click point 2 (BOTTOM - RIGHT) parallel to the road edge near camera",
  "👉 Step 3: Click point 3 (TOP - RIGHT) moving up close to the intersection",
  "👉 Step 4: Click point 4 (TOP - LEFT) at the intersection to close the trapezoid",
  "✅ All 4 points selected! Click 'Save 3D Trapezoid Zone' to confirm."
];

// Default BEV destination points (rectangle in 3D space)
const DEFAULT_DST_POINTS: Point[] = [
  { x: 0, y: 600 },
  { x: 400, y: 600 },
  { x: 400, y: 0 },
  { x: 0, y: 0 },
];

const STOP_LINE_3D_Y = 400;

export default function BEVEditor({ imageUrl, onSave, onCancel, initialPoints = [] }: BEVEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point[]>(initialPoints);
  const [stepIndex, setStepIndex] = useState(initialPoints.length >= 4 ? 4 : initialPoints.length);

  // Simple perspective transform (3x3 matrix multiplication)
  const perspectiveTransform = (pt: Point, m: number[][]) => {
    const x = pt.x, y = pt.y;
    const w = m[2][0] * x + m[2][1] * y + m[2][2];
    const px = (m[0][0] * x + m[0][1] * y + m[0][2]) / w;
    const py = (m[1][0] * x + m[1][1] * y + m[1][2]) / w;
    return { x: px, y: py };
  };

  // Compute inverse 3x3 matrix
  const invertMatrix3x3 = (m: number[][]) => {
    const det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    const invDet = 1 / det;
    return [
      [
        (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet,
        (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invDet,
        (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet
      ],
      [
        (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invDet,
        (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet,
        (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * invDet
      ],
      [
        (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet,
        (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * invDet,
        (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet
      ]
    ];
  };

  // Compute perspective transform matrix from 4 point pairs
  const computePerspectiveMatrix = (src: Point[], dst: Point[]) => {
    if (src.length < 4 || dst.length < 4) return null;
    // Using OpenCV-style DLT algorithm
    const [s0, s1, s2, s3] = src;
    const [d0, d1, d2, d3] = dst;
    
    // Build matrix A for least squares
    const A: number[][] = [];
    const addEq = (sp: Point, dp: Point) => {
      A.push([-sp.x, -sp.y, -1, 0, 0, 0, dp.x * sp.x, dp.x * sp.y, dp.x]);
      A.push([0, 0, 0, -sp.x, -sp.y, -1, dp.y * sp.x, dp.y * sp.y, dp.y]);
    };
    addEq(s0, d0); addEq(s1, d1); addEq(s2, d2); addEq(s3, d3);
    
    // Solve using SVD-like approach (simplified - using direct solution for 4 points)
    // For simplicity, use a basic approach
    const m: number[][] = [[1,0,0],[0,1,0],[0,0,1]];
    // This is a simplified version - in production you'd use proper DLT
    return m;
  };

  // Redraw canvas
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw image
      ctx.drawImage(img, 0, 0);

      if (points.length > 0) {
        // Draw filled polygon if 4 points complete
        if (points.length === 4) {
          ctx.fillStyle = "rgba(0, 229, 255, 0.2)";
          ctx.strokeStyle = "#00e5ff";
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          for (let i = 1; i < 4; i++) {
            ctx.lineTo(points[i].x, points[i].y);
          }
          ctx.closePath();
          ctx.fill();
          ctx.stroke();

          // Draw projected stop line (3D -> 2D)
          // Using default dst points and STOP_LINE_3D_Y
          const dstPts = DEFAULT_DST_POINTS;
          const srcPts = points;
          
          // Simple linear interpolation for stop line projection
          // In production, this would use the actual perspective matrix
          const stopLineY = STOP_LINE_3D_Y;
          const yRatio = stopLineY / 600; // 600 is max Y in BEV space
          
          // Project stop line endpoints
          const leftX = srcPts[0].x + (srcPts[3].x - srcPts[0].x) * (1 - yRatio);
          const leftY = srcPts[0].y + (srcPts[3].y - srcPts[0].y) * (1 - yRatio);
          const rightX = srcPts[1].x + (srcPts[2].x - srcPts[1].x) * (1 - yRatio);
          const rightY = srcPts[1].y + (srcPts[2].y - srcPts[1].y) * (1 - yRatio);
          
          // Draw stop line
          ctx.strokeStyle = "#ff9800";
          ctx.lineWidth = 4;
          ctx.setLineDash([10, 5]);
          ctx.beginPath();
          ctx.moveTo(leftX, leftY);
          ctx.lineTo(rightX, rightY);
          ctx.stroke();
          ctx.setLineDash([]);
          
          // Draw stop line label
          const midX = (leftX + rightX) / 2;
          const midY = (leftY + rightY) / 2;
          ctx.fillStyle = "#ff9800";
          ctx.font = "bold 12px sans-serif";
          ctx.fillText("STOP LINE (3D)", midX + 10, midY - 10);
        }

        // Draw lines between consecutive points
        ctx.strokeStyle = "#00e5ff";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();

        // Draw point markers
        points.forEach((p, idx) => {
          ctx.fillStyle = "#00e5ff";
          ctx.beginPath();
          ctx.arc(p.x, p.y, 6, 0, 2 * Math.PI);
          ctx.fill();

          ctx.fillStyle = "white";
          ctx.font = "bold 14px sans-serif";
          ctx.fillText(" " + (idx + 1), p.x + 8, p.y + 5);
        });
      }
    };
    img.src = imageUrl;
  }, [imageUrl, points]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (points.length >= 4) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    const newPoints = [...points, { x, y }];
    setPoints(newPoints);
    setStepIndex(Math.min(newPoints.length, 4));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      if (points.length > 0) {
        const newPoints = points.slice(0, -1);
        setPoints(newPoints);
        setStepIndex(Math.max(0, newPoints.length));
      } else {
        onCancel();
      }
    } else if (e.key === "Backspace" && points.length > 0) {
      const newPoints = points.slice(0, -1);
      setPoints(newPoints);
      setStepIndex(Math.max(0, newPoints.length));
    }
  };

  const handleSave = () => {
    if (points.length >= 4) {
      onSave(points.slice(0, 4));
    }
  };

  const handleReset = () => {
    setPoints([]);
    setStepIndex(0);
  };

  if (!imageUrl) {
    return (
      <div className="p-4 bg-slate-800 rounded-lg text-center text-slate-400">
        Please open camera/webcam first to draw 3D trapezoid zone
      </div>
    );
  }

  const instructionColor = stepIndex >= 4 ? "#28a745" : "#ffca28";

  return (
    <div className="space-y-4">
      <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <h3 className="text-sm font-medium text-blue-300 mb-2">
          🛠️ 3D TRAPEZOID ZONE EDITOR (BIRD'S EYE VIEW)
        </h3>
        <p className="text-xs font-bold mb-2" style={{ color: instructionColor }}>
          {STEPS[stepIndex]}
        </p>
        <ul className="text-xs text-slate-400 space-y-1">
          <li>• Click 4 points in order to create trapezoid</li>
          <li>• Press <strong>Backspace</strong> to remove last point</li>
          <li>• Press <strong>Esc</strong> to cancel</li>
          <li>• Need exactly 4 points to create 3D trapezoid</li>
        </ul>
      </div>

      <div className="relative border-2 border-slate-600 rounded-lg overflow-hidden">
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onKeyDown={handleKeyDown}
          tabIndex={0}
          className="w-full cursor-crosshair"
          style={{ maxHeight: "500px", objectFit: "contain" }}
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-400">
          Points: <span className="text-white font-medium">{points.length}/4</span>
          {points.length === 4 && (
            <span className="ml-2 text-green-400">✅ All 4 points ready</span>
          )}
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleReset}
            disabled={points.length === 0}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-sm font-bold transition-colors"
          >
            Reset
          </button>
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={points.length < 4}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-sm font-bold transition-colors"
          >
            Save 3D Trapezoid Zone
          </button>
        </div>
      </div>
    </div>
  );
}