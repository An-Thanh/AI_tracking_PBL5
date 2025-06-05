import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import Optional

load_dotenv()

class Config:
    """Configuration class for the face recognition system."""
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        self.dataset_path = self.base_dir / 'dataset/images'
        self.metadata_path = self.base_dir / 'dataset/metadata/metadata.json'
        
        # Database configuration
        self.db_config = {
            "host": os.getenv("HOST"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
        }
        
        # Face processing settings
        self.face_extraction_model = "ArcFace"
        self.detector_backend = "ssd"
        self.distance_metric = "cosine"
        self.verification_threshold_multiplier = 0.6
        self.consecutive_recognitions_needed = 3
        self.minimum_face_size = 10
        self.face_quality_threshold = 0.05
        self.smoothing_window = 5
        self.verified_display_duration = 1
        self.inactive_face_timeout = 10
        self.enable_blink_detection = True
        self.blink_check_interval = 3
        self.frame_skip_rate = 1
        self.resize_for_processing = True
        self.processing_frame_width = 320
        self.frame_queue_size = 5
        self.result_queue_size = 5
        
        # Table and column names
        self.face_embeddings_table = "face_embeddings"
        self.column_person_name = "person_name"
        self.column_person_id = "person_id"
        self.column_embedding = "embedding"
        self.column_model = "model"
        
        self.logger = self._setup_logger()
        self.valid_exts = {'.jpg', '.jpeg', '.png'}
        self.max_workers = os.cpu_count() or 4

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for the application."""
        logger = logging.getLogger("FaceRecognition")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger