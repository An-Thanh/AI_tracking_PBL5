import cv2
import numpy as np
from typing import Optional, Tuple
from collections import deque
import time
from ..config import Config

class FaceVisualizer:
    """Handles visualization of face recognition results."""
    
    def __init__(self, config: Config):
        self.config = config
        self.face_recognition_history = {}
        self.recognition_timestamps = {}
        self.blink_detection_state = {}
    
    def get_face_id(self, fa: dict) -> str:
        """Generate a unique face ID based on facial area."""
        return f"f_{int(fa['x'] / 50)}_{int(fa['y'] / 50)}_{int(fa['w'] / 50)}_{int(fa['h'] / 50)}"
    
    def assess_face_quality(self, face_img_crop: np.ndarray) -> float:
        """Assess the quality of a face crop."""
        try:
            if face_img_crop is None or face_img_crop.size == 0:
                return 0.0
            
            h_crop, w_crop = face_img_crop.shape[:2]
            if w_crop < 10 or h_crop < 10:
                return 0.1
            
            gray_face = cv2.cvtColor(face_img_crop, cv2.COLOR_BGR2GRAY) if len(face_img_crop.shape) == 3 else face_img_crop
            brightness = np.mean(gray_face)
            brightness_score = 1.0
            if brightness < 40:
                brightness_score = max(0.0, brightness / 80.0)
            elif brightness > 215:
                brightness_score = max(0.0, (255.0 - brightness) / 80.0)
            
            contrast = np.std(gray_face.astype(np.float32))
            contrast_score = 1.0
            if contrast < 30:
                contrast_score = max(0.0, contrast / 60.0)
            
            laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
            blur_score = min(laplacian_var / 700.0, 1.0)
            
            quality = (blur_score * 0.5) + (brightness_score * 0.3) + (contrast_score * 0.2)
            return min(max(quality, 0.0), 1.0)
        except Exception as e:
            self.config.logger.error(f"Error in face quality assessment: {e}")
            return 0.2
    
    def detect_blink(self, face_id: str) -> Tuple[bool, bool]:
        """Simulate blink detection for a face."""
        if not self.config.enable_blink_detection:
            return False, False
        now = time.time()
        state = self.blink_detection_state.setdefault(face_id, {'last_check': 0, 'blinked_in_interval': False})
        if now - state['last_check'] < self.config.blink_check_interval:
            return state['blinked_in_interval'], False
        state['last_check'] = now
        state['blinked_in_interval'] = random.random() < 0.2
        return state['blinked_in_interval'], True
    
    def update_recognition_history(self, face_id: str, name: str, distance: float) -> dict:
        """Update recognition history for a face."""
        now = time.time()
        entry = self.face_recognition_history.setdefault(
            face_id, {"h": deque(maxlen=self.config.smoothing_window), "cc": 0, "lvn": None, "v": False, "lst": now}
        )
        entry["lst"] = now
        entry["h"].append((name, distance))
        entry["cc"] = sum(1 for rn, _ in reversed(entry["h"]) if rn == name) if name != "Unknown" else 0
        if entry["cc"] >= self.config.consecutive_recognitions_needed and name != "Unknown":
            entry["v"], entry["lvn"] = True, name
            self.recognition_timestamps[face_id] = now
        elif entry["cc"] < self.config.consecutive_recognitions_needed and not (
                face_id in self.recognition_timestamps and
                (now - self.recognition_timestamps.get(face_id, 0) < self.config.verified_display_duration)):
            entry["v"] = False
        return entry
    
    def draw_face_info(self, frame: np.ndarray, faces_data: list[dict], known_faces: list[dict], scales: Optional[Tuple[float, float]], threshold: float):
        """Draw face information on the frame."""
        disp_frame = frame.copy()
        for face_d in faces_data or []:
            if not (isinstance(face_d, dict) and "embedding" in face_d and face_d.get("embedding") and "facial_area" in face_d):
                self.config.logger.debug("Invalid face data structure. Skipping.")
                continue
            
            emb = np.array(face_d["embedding"], dtype=np.float32)
            fa_raw = face_d["facial_area"]
            x, y, w, h = fa_raw['x'], fa_raw['y'], fa_raw['w'], fa_raw['h']
            
            if scales:
                sx, sy = scales
                x_orig, y_orig, w_orig, h_orig = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
            else:
                x_orig, y_orig, w_orig, h_orig = x, y, w, h
            
            if w_orig < self.config.minimum_face_size or h_orig < self.config.minimum_face_size:
                self.config.logger.debug(f"Face filtered by size: w={w_orig}, h={h_orig}")
                continue
            
            crop = disp_frame[max(0, y_orig):y_orig + h_orig, max(0, x_orig):x_orig + w_orig]
            if crop.size == 0:
                self.config.logger.debug(f"Empty face crop: x={x_orig}, y={y_orig}, w={w_orig}, h={h_orig}")
                continue
            
            quality_score = self.assess_face_quality(crop)
            if quality_score < self.config.face_quality_threshold:
                self.config.logger.debug(f"Face filtered by quality: {quality_score:.2f}")
                continue
            
            face_id = self.get_face_id({'x': x_orig, 'y': y_orig, 'w': w_orig, 'h': h_orig})
            blinked, checked_blink = self.detect_blink(face_id)
            
            min_distance, best_name = float('inf'), None
            if known_faces:
                for entry in known_faces:
                    distance = self.calculate_face_distance(emb, entry["embedding"])
                    if distance < min_distance:
                        min_distance, best_name = distance, entry["person_name"]
            
            match = (best_name is not None) and (min_distance <= threshold)
            hist = self.update_recognition_history(face_id, best_name if match else "Unknown", min_distance)
            
            display_name, color = "Unknown", (0, 0, 255)
            blink_ok = self.blink_detection_state.get(face_id, {}).get('blinked_in_interval', False)
            
            if hist.get("v", False):
                display_name = hist["lvn"]
                color = (50, 180, 50) if self.config.enable_blink_detection and not blink_ok else (0, 255, 0)
                if time.time() - self.recognition_timestamps.get(face_id, 0) < self.config.verified_display_duration:
                    cv2.putText(disp_frame, "VERIFIED", (x_orig, y_orig + h_orig + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif match:
                display_name, color = f"{best_name}?", (0, 165, 255)
            
            dist_str = f" ({min_distance:.2f})" if best_name and best_name != "Unknown" else ""
            blink_str = (" B:" + ("Y" if blinked else "N")) if self.config.enable_blink_detection and checked_blink else \
                (" B:PrevY" if self.config.enable_blink_detection and blink_ok else "")
            
            cv2.rectangle(disp_frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), color, 2)
            cv2.putText(disp_frame, f"{display_name}{dist_str}{blink_str}",
                        (x_orig, y_orig - 7 if y_orig - 7 > 7 else y_orig + h_orig + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            if self.config.enable_blink_detection and (checked_blink or face_id in self.blink_detection_state):
                cv2.circle(disp_frame, (x_orig + w_orig - 10, y_orig + 10), 5,
                           (0, 255, 0) if blink_ok else (0, 0, 255), -1)
        
        return disp_frame