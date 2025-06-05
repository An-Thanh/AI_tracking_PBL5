import cv2
import numpy as np
from queue import Queue, Empty, Full
import threading
from typing import Optional, Tuple
from ..config import Config
from ..face_processing.recognition import FaceRecognizer

class InferenceProcessor(threading.Thread):
    """Processes frames for face detection and embedding extraction."""
    
    def __init__(self, frame_queue: Queue, result_queue: Queue, config: Config):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.config = config
        self.recognizer = FaceRecognizer(config)
        self.frame_count = 0
        self.name = "InferenceProcessorThread"
    
    def run(self):
        self.config.logger.info(f"{self.name} starting...")
        try:
            self.recognizer.initialize_models()
        except Exception as e:
            self.config.logger.error(f"Failed to initialize models: {e}")
            return
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except Empty:
                continue
            
            self.frame_count += 1
            if self.config.frame_skip_rate > 1 and self.frame_count % self.config.frame_skip_rate != 0:
                try:
                    self.result_queue.put((frame, [], None), timeout=0.1)
                except Full:
                    pass
                continue
            
            faces_data, resize_scales = self.recognizer.process_faces(frame, optimize=True)
            try:
                self.result_queue.put((frame, faces_data, resize_scales), timeout=0.5)
            except Full:
                pass
        
        self.config.logger.info(f"{self.name} stopped.")