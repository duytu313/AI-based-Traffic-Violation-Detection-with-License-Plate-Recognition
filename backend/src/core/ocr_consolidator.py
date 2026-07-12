"""
OCR Consolidator - Frame-level OCR result consolidation per track ID.
Solves the problem of duplicate/inconsistent OCR reads across frames.

Cùng một xe đi qua nhưng OCR đọc 20-100 lần với kết quả khác nhau.
Module này dùng voting/consensus mechanism để chọn ra kết quả đúng nhất.

Design principles:
1. Mỗi track_id có một bộ nhớ OCR riêng (history)
2. Kết quả cuối cùng = voting (most common OCR text)
3. Chỉ ghi DB 1 lần cho mỗi: plate entry, violation type per track
4. Tự động cleanup các track cũ sau timeout
"""
import time
from collections import defaultdict, Counter
from typing import Optional, Set, Tuple, Dict


class TrackOCRMemory:
    """OCR memory for a single tracked vehicle across its lifetime."""
    
    def __init__(self, track_id: int, min_vote_frames: int = 5, vote_ratio: float = 0.6):
        self.track_id = track_id
        self.min_vote_frames = min_vote_frames  # Minimum frames before finalizing
        self.vote_ratio = vote_ratio  # Minimum ratio (60%) for consensus
        
        # OCR history: plate_text -> count
        self.ocr_results: Dict[str, int] = defaultdict(int)
        self.total_ocr_reads: int = 0
        
        # Best plate after consolidation
        self._best_plate: str = ""
        self._best_plate_count: int = 0
        self._finalized_plate: bool = False
        
        # State tracking
        self.plate_entry_created: bool = False
        self.violations_sent: Set[str] = set()  # Violation types already recorded
        self.last_seen: float = time.time()
        self.first_seen: float = time.time()
        
        # Store the best vehicle crop for evidence
        self.best_vehicle_crop = None
        self.best_plate_img = None

    def add_ocr_result(self, plate_text: str, confidence: float = 1.0) -> None:
        """
        Add an OCR reading for this track.
        
        Args:
            plate_text: The OCR result text
            confidence: OCR confidence score (unused in voting, but reserved)
        """
        if not plate_text or plate_text.strip() == "":
            return
        
        plate_text = plate_text.strip().upper()
        self.ocr_results[plate_text] += 1
        self.total_ocr_reads += 1
        self.last_seen = time.time()
        
        # Update best plate if this one is now more frequent
        current_best = self._get_most_common_plate()
        if current_best and current_best[1] > self._best_plate_count:
            self._best_plate = current_best[0]
            self._best_plate_count = current_best[1]

    def _get_most_common_plate(self) -> Optional[Tuple[str, int]]:
        """Get the most common plate text and its count."""
        if not self.ocr_results:
            return None
        best_text = max(self.ocr_results, key=self.ocr_results.get)
        return (best_text, self.ocr_results[best_text])

    def get_best_plate(self) -> Tuple[str, bool]:
        """
        Get the best (consolidated) plate text.
        
        Returns:
            (plate_text, is_finalized)
            - plate_text: The most common OCR result
            - is_finalized: True if we've seen enough frames to be confident
        """
        if not self.ocr_results:
            return ("", False)
        
        best = self._get_most_common_plate()
        if best is None:
            return ("", False)
        
        best_text, best_count = best
        total = self.total_ocr_reads
        
        # Only finalize if we've seen enough frames and the winner has clear majority
        if total >= self.min_vote_frames:
            vote_share = best_count / total
            if vote_share >= self.vote_ratio:
                self._finalized_plate = True
                self._best_plate = best_text
        
        return (best_text, self._finalized_plate)

    def is_plate_stable(self) -> bool:
        """Check if OCR results have stabilized (enough frames, clear winner)."""
        self.get_best_plate()  # Side effect: updates _finalized_plate
        return self._finalized_plate

    def has_violation_been_sent(self, violation_type: str) -> bool:
        """Check if this violation type was already recorded for this track."""
        return violation_type in self.violations_sent

    def mark_violation_sent(self, violation_type: str) -> None:
        """Mark a violation type as recorded for this track."""
        self.violations_sent.add(violation_type)

    def is_plate_entry_created(self) -> bool:
        return self.plate_entry_created

    def mark_plate_entry_created(self) -> None:
        self.plate_entry_created = True

    def get_age(self) -> float:
        """Get age of this track in seconds since first seen."""
        return time.time() - self.first_seen


class OCRConsolidator:
    """
    Manages OCR consolidation across all tracked vehicles.
    
    Usage:
        consolidator = OCRConsolidator()
        
        # Per frame:
        for each vehicle:
            track_memory = consolidator.get_track_memory(track_id)
            track_memory.add_ocr_result(plate_text)
            best_plate, is_final = track_memory.get_best_plate()
            
            if track_memory.is_plate_stable() and not track_memory.is_plate_entry_created():
                # Save to DB once
                track_memory.mark_plate_entry_created()
            
            if not track_memory.has_violation_been_sent(violation_type):
                # Save violation to DB once
                track_memory.mark_violation_sent(violation_type)
    """
    
    def __init__(self, min_vote_frames: int = 5, vote_ratio: float = 0.6,
                 track_timeout: float = 10.0, cleanup_interval: float = 30.0):
        """
        Args:
            min_vote_frames: Minimum frames before finalizing plate (default: 5)
            vote_ratio: Minimum vote share for consensus (default: 0.6 = 60%)
            track_timeout: Remove tracks unseen for this many seconds
            cleanup_interval: Run cleanup every N seconds
        """
        self.tracks: Dict[int, TrackOCRMemory] = {}
        self.min_vote_frames = min_vote_frames
        self.vote_ratio = vote_ratio
        self.track_timeout = track_timeout
        self._last_cleanup = time.time()
        self.cleanup_interval = cleanup_interval

    def get_track_memory(self, track_id: int) -> TrackOCRMemory:
        """Get or create OCR memory for a track."""
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackOCRMemory(
                track_id,
                min_vote_frames=self.min_vote_frames,
                vote_ratio=self.vote_ratio
            )
        return self.tracks[track_id]

    def remove_track(self, track_id: int) -> None:
        """Remove a track's memory (e.g., after vehicle exits)."""
        if track_id in self.tracks:
            del self.tracks[track_id]

    def cleanup(self) -> int:
        """
        Remove old tracks that haven't been seen recently.
        
        Returns:
            Number of tracks removed
        """
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return 0
        
        self._last_cleanup = now
        to_remove = [
            tid for tid, mem in self.tracks.items()
            if now - mem.last_seen > self.track_timeout
        ]
        for tid in to_remove:
            del self.tracks[tid]
        return len(to_remove)

    def get_active_track_ids(self) -> list:
        """Get all active track IDs."""
        self.cleanup()
        return list(self.tracks.keys())

    def get_stats(self) -> dict:
        """Get consolidation statistics."""
        active = len(self.tracks)
        finalized = sum(1 for m in self.tracks.values() if m._finalized_plate)
        total_violations = sum(len(m.violations_sent) for m in self.tracks.values())
        return {
            "active_tracks": active,
            "finalized_plates": finalized,
            "total_violations_recorded": total_violations,
            "min_vote_frames": self.min_vote_frames,
            "vote_ratio": self.vote_ratio,
        }

    def finalize_plate_for_track(self, track_id: int) -> Optional[str]:
        """
        Force-finalize the best plate for a track and return it.
        Used when vehicle exits the frame.
        Returns the final plate text or None.
        """
        if track_id not in self.tracks:
            return None
        mem = self.tracks[track_id]
        best_plate, _ = mem.get_best_plate()
        if best_plate:
            mem._finalized_plate = True
            mem._best_plate = best_plate
        return best_plate if best_plate else None