import numpy as np
from deepface import DeepFace
from deepface.commons import functions, distance as dst_functions
from typing import Optional, Tuple
from ..config import Config

class FaceRecognizer:
    """Handles real-time face recognition."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embedding_model_obj = None
        self.embedding_target_size = None
    
    def initialize_models(self):
        """Initialize face detection and embedding models."""
        try:
            self.config.logger.info(f"Initializing embedding model: {self.config.face_extraction_model}")
            self.embedding_model_obj = DeepFace.build_model(self.config.face_extraction_model)
            keras_input_shape = self.embedding_model_obj.input_shape
            if isinstance(keras_input_shape, tuple) and len(keras_input_shape) == 4:
                self.embedding_target_size = (keras_input_shape[1], keras_input_shape[2])
            elif isinstance(keras_input_shape, list) and len(keras_input_shape) > 0 and isinstance(keras_input_shape[0], tuple):
                self.embedding_target_size = (keras_input_shape[0][1], keras_input_shape[0][2])
            else:
                self.embedding_target_size = functions.get_input_shape(self.embedding_model_obj)
            self.config.logger.info(f"Embedding target size: {self.embedding_target_size}")
        except Exception as e:
            self.config.logger.error(f"Failed to initialize models: {e}")
            raise
    
    def get_recognition_threshold(self) -> float:
        """Get recognition threshold for the model and distance metric."""
        try:
            from deepface.commons.thresholding import get_threshold
            threshold = get_threshold(self.config.face_extraction_model, self.config.distance_metric)
            source = "API"
        except Exception:
            maps = {
                "ArcFace": {"cosine": 0.68, "euclidean_l2": 1.13},
                "SFace": {"cosine": 0.593, "euclidean_l2": 1.055},
                "VGG-Face": {"cosine": 0.40, "euclidean_l2": 0.86},
                "Facenet": {"cosine": 0.40, "euclidean_l2": 0.80}
            }
            model_map = maps.get(self.config.face_extraction_model, {})
            threshold = model_map.get(self.config.distance_metric, 0.4 if self.config.distance_metric == "cosine" else 1.0)
            source = "map"
        
        final_thresh = threshold * self.config.verification_threshold_multiplier
        self.config.logger.info(
            f"Base threshold ({source}): {threshold:.4f}, Adjusted threshold "
            f"({self.config.verification_threshold_multiplier}): {final_thresh:.4f}"
        )
        return final_thresh
    
    def calculate_face_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate distance between two face embeddings."""
        if self.config.distance_metric == "cosine":
            return dst_functions.findCosineDistance(emb1, emb2)
        elif self.config.distance_metric == "euclidean_l2":
            return dst_functions.findEuclideanDistance(
                dst_functions.l2_normalize(emb1), dst_functions.l2_normalize(emb2)
            )
        elif self.config.distance_metric == "euclidean":
            return dst_functions.findEuclideanDistance(emb1, emb2)
        self.config.logger.warning(f"Unknown distance metric: {self.config.distance_metric}. Defaulting to cosine.")
        return dst_functions.findCosineDistance(emb1, emb2)
    
    def process_faces(self, frame: np.ndarray, optimize: bool = True) -> list[dict]:
        """Process faces in a frame to extract embeddings."""
        processed_frame = frame
        resize_scales = None
        
        if optimize and self.config.resize_for_processing:
            h_orig, w_orig = frame.shape[:2]
            if w_orig > self.config.processing_frame_width:
                ratio = self.config.processing_frame_width / w_orig
                h_target = int(h_orig * ratio)
                processed_frame = cv2.resize(frame, (self.config.processing_frame_width, h_target),
                                            interpolation=cv2.INTER_AREA)
                resize_scales = (w_orig / self.config.processing_frame_width, h_orig / h_target)
        
        faces_output_data = []
        try:
            if optimize and self.embedding_model_obj and self.embedding_target_size:
                extracted_faces_info = DeepFace.extract_faces(
                    img_path=processed_frame,
                    detector_backend=self.config.detector_backend,
                    enforce_detection=False,
                    align=True,
                    target_size=self.embedding_target_size
                )
                
                face_batch, temp_infos = [], []
                for idx, face_info in enumerate(extracted_faces_info):
                    face_img = face_info.get('face')
                    if face_img is None or face_img.size == 0:
                        self.config.logger.debug(f"Empty face array for face {idx}. Skipping.")
                        continue
                    face_batch.append(face_img)
                    temp_infos.append({"facial_area": face_info["facial_area"], "confidence": face_info["confidence"]})
                
                if face_batch:
                    np_batch = np.array(face_batch)
                    if np_batch.ndim == 3:
                        np_batch = np.expand_dims(np_batch, axis=-1)
                    if np_batch.ndim == 4 and np_batch.shape[0] > 0:
                        if np_batch.dtype != np.float32:
                            np_batch = np_batch.astype(np.float32)
                        model_in_shape = self.embedding_model_obj.input_shape
                        expected_ch = model_in_shape[0][-1] if isinstance(model_in_shape, list) else model_in_shape[-1]
                        if expected_ch == 3 and np_batch.shape[-1] == 1:
                            np_batch = np.concatenate([np_batch] * 3, axis=-1)
                        
                        embeddings = self.embedding_model_obj.predict(np_batch)
                        for i, emb_vec in enumerate(embeddings):
                            faces_output_data.append({
                                "embedding": emb_vec.tolist(),
                                "facial_area": temp_infos[i]["facial_area"],
                                "confidence": temp_infos[i]["confidence"]
                            })
            else:
                faces_output_data = DeepFace.represent(
                    img_path=processed_frame,
                    model_name=self.config.face_extraction_model,
                    detector_backend=self.config.detector_backend,
                    enforce_detection=False,
                    align=True
                )
            return faces_output_data, resize_scales
        except Exception as e:
            self.config.logger.error(f"Error in face processing: {e}")
            return [], resize_scales