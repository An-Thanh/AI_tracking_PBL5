import cv2
import threading
from queue import Queue, Full
from ..config import Config

class FrameReader(threading.Thread):
    """Reads frames from a video source."""
    
    def __init__(self, video_source: int, frame_queue: Queue, stop_event: threading.Event, config: Config):
        super().__init__(daemon=True)
        self.video_source = video_source
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.config = config
        self.cap = None
        self.name = "FrameReaderThread"
    
    def run(self):
        self.config.logger.info(f"{self.name} starting...")
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            if not self.cap.isOpened():
                self.config.logger.error(f"Cannot open video source: {self.video_source}")
                self.stop_event.set()
                return
            
            while not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.1)
                    if isinstance(self.video_source, int):
                        if not self.cap.isOpened():
                            self.config.logger.warning("Webcam disconnected. Reopening...")
                            self.cap.release()
                            self.cap = cv2.VideoCapture(self.video_source)
                            if not self.cap.isOpened():
                                self.config.logger.error("Failed to reopen. Stopping.")
                                self.stop_event.set()
                                break
                    else:
                        self.config.logger.info("End of video file.")
                        self.stop_event.set()
                        break
                    continue
                
                try:
                    self.frame_queue.put(frame, timeout=0.5)
                except Full:
                    pass
        except Exception as e:
            self.config.logger.error(f"{self.name} exception: {e}")
        finally:
            if self.cap and self.cap.isOpened():
                self.cap.release()
            self.config.logger.info(f"{self.name} stopped.")