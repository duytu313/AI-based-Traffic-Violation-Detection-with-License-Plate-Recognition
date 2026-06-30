"use client";

import { useState, useRef, useEffect, useCallback } from "react";

interface Point {
  x: number;
  y: number;
}

interface ROIEditorProps {
  imageUrl: string | null;
  onSave: (points: Point[]) => void;
  onCancel: () => void;
  initialPoints?: Point[];
}

export default function ROIEditor({ imageUrl, onSave, onCancel, initialPoints = [] }: ROIEditorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<Point[]>(initialPoints);
  const [isComplete, setIsComplete] = useState(initialPoints.length >= 3);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });

  // Redraw canvas
  const redraw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
      // Set canvas size to match image
      canvas.width = img.width;
      canvas.height = img.height;
      setImageSize({ width: img.width, height: img.height });

      // Draw image
      ctx.drawImage(img, 0, 0);

      // Draw existing points
      if (points.length > 0) {
        // Draw polygon fill
        if (points.length >= 3) {
          ctx.fillStyle = "rgba(255, 255, 0, 0.2)";
          ctx.strokeStyle = "rgba(255, 255, 0, 0.8)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          for (let i = 1; i < points.length; i++) {
            ctx.lineTo(points[i].x, points[i].y);
          }
          if (isComplete) {
            ctx.closePath();
            ctx.fill();
          }
          ctx.stroke();
        }

        // Draw lines between points
        ctx.strokeStyle = "rgba(255, 255, 0, 0.8)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.stroke();

        // Draw points
        for (const point of points) {
          ctx.fillStyle = "rgba(255, 0, 0, 0.8)";
          ctx.strokeStyle = "white";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }

        // Draw point numbers
        ctx.fillStyle = "white";
        ctx.font = "bold 14px Arial";
        for (let i = 0; i < points.length; i++) {
          ctx.fillText(`${i + 1}`, points[i].x + 10, points[i].y - 10);
        }
      }
    };
    img.src = imageUrl;
  }, [imageUrl, points, isComplete]);

  useEffect(() => {
    redraw();
  }, [redraw]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isComplete) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    setPoints([...points, { x, y }]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && points.length >= 3) {
      setIsComplete(true);
    } else if (e.key === "Escape") {
      if (points.length > 0) {
        setPoints(points.slice(0, -1));
      } else {
        onCancel();
      }
    } else if (e.key === "Backspace" && points.length > 0) {
      setPoints(points.slice(0, -1));
    }
  };

  const handleSave = () => {
    if (points.length >= 3) {
      // Round to integers to avoid 422 errors
      const roundedPoints = points.map(p => ({ x: Math.round(p.x), y: Math.round(p.y) }));
      onSave(roundedPoints);
    }
  };

  const handleReset = () => {
    setPoints([]);
    setIsComplete(false);
  };

  if (!imageUrl) {
    return (
      <div className="p-4 bg-slate-800 rounded-lg text-center text-slate-400">
        Please open camera/webcam first to draw violation zone
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
        <h3 className="text-sm font-medium text-blue-300 mb-2">
          🎯 Violation Zone Drawing Tool (ROI Editor)
        </h3>
        <ul className="text-xs text-slate-400 space-y-1">
          <li>• Click to add violation zone points</li>
          <li>• Press <strong>Enter</strong> to complete zone</li>
          <li>• Press <strong>Backspace</strong> to remove last point</li>
          <li>• Press <strong>Esc</strong> to cancel/remove last point</li>
          <li>• Need at least 3 points to create zone</li>
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
          Points: <span className="text-white font-medium">{points.length}</span>
          {isComplete && (
            <span className="ml-2 text-green-400">✓ Complete</span>
          )}
        </div>

        <div className="flex gap-2">
          <button
            onClick={handleReset}
            disabled={points.length === 0}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-sm transition-colors"
          >
            Clear All
          </button>
          <button
            onClick={onCancel}
            className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-sm transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={!isComplete}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-800 disabled:text-slate-500 text-white rounded-lg text-sm transition-colors"
          >
            Save Zone
          </button>
        </div>
      </div>
    </div>
  );
}