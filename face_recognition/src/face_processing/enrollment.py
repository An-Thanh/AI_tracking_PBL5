from pathlib import Path
from typing import List, Dict, Optional, Union
import numpy as np
from mtcnn import MTCNN
from deepface import DeepFace
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ..config import Config
from ..utils.file_utils import load_metadata

class FaceEnrollmentSystem:
    """Face enrollment system that detects, extracts, and stores face embeddings."""
    
    def __init__(self, config: Config):
        self.config = config
        self.face_detector = MTCNN()
        self.known_face_embeddings = []
        self.known_face_names = []
        self.person_metadata = {}
        self.enrollment_stats = {
            "total_images_processed": 0,
            "successful_enrollments": 0,
            "failed_enrollments": 0,
            "unique_individuals": 0
        }
    
    def load_person_metadata(self, json_file_path: Optional[str] = None) -> Dict[str, Dict]:
        """Load person metadata from JSON file."""
        self.person_metadata = load_metadata(json_file_path or self.config.metadata_path, self.config.logger)
        return self.person_metadata
    
    def process_single_image(self, image_path: Path, person_name: str) -> Optional[Dict]:
        """Process a single image to extract face embedding."""
        if not image_path.is_file() or image_path.suffix.lower() not in self.config.valid_exts:
            return None
        
        try:
            embedding_objs = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.config.face_extraction_model,
                detector_backend="mtcnn",
                enforce_detection=True,
                align=True
            )
            
            if (isinstance(embedding_objs, list) and len(embedding_objs) > 0 and
                    isinstance(embedding_objs[0], dict) and 'embedding' in embedding_objs[0]):
                embedding = embedding_objs[0]['embedding']
                metadata = self.person_metadata.get(person_name, {})
                
                return {
                    "person_name": person_name,
                    "image_path": str(image_path),
                    "embedding": embedding,
                    "id": metadata.get("id", ""),
                    "birthday": metadata.get("birthday", ""),
                    "model": self.config.face_extraction_model,
                    **{k: v for k, v in metadata.items() if k not in ["id", "birthday"]}
                }
            
            self.config.logger.warning(f"No valid embedding found in {image_path}")
            return None
        except Exception as e:
            self.config.logger.error(f"Failed to process {image_path}: {str(e)}")
            return None
    
    def enroll_faces_from_dataset(self, dataset_path: Optional[Union[str, Path]] = None) -> List[Dict]:
        """Enroll faces from the dataset."""
        path = Path(dataset_path) if dataset_path else self.config.dataset_path
        if not path.exists():
            self.config.logger.error(f"Dataset path {path} not found.")
            return []
        
        self.config.logger.info(f"Enrolling faces from {path} using {self.config.face_extraction_model}...")
        self.known_face_embeddings = []
        self.known_face_names = []
        embeddings_for_db = []
        person_stats = {}
        
        person_folders = [f for f in path.iterdir() if f.is_dir()]
        for person_folder in person_folders:
            person_name = person_folder.name
            person_stats[person_name] = {"processed": 0, "successful": 0}
            image_files = [f for f in person_folder.iterdir() if f.is_file() and f.suffix.lower() in self.config.valid_exts]
            
            self.config.logger.info(f"Processing {len(image_files)} images for {person_name}")
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_image = {
                    executor.submit(self.process_single_image, img_path, person_name): img_path
                    for img_path in image_files
                }
                
                for future in tqdm(as_completed(future_to_image), total=len(future_to_image), desc=f"Enrolling {person_name}"):
                    image_path = future_to_image[future]
                    person_stats[person_name]["processed"] += 1
                    self.enrollment_stats["total_images_processed"] += 1
                    
                    try:
                        embedding_obj = future.result()
                        if embedding_obj:
                            embeddings_for_db.append(embedding_obj)
                            self.known_face_embeddings.append({
                                "person_name": person_name,
                                "embedded": embedding_obj["embedding"]
                            })
                            self.known_face_names.append(person_name)
                            person_stats[person_name]["successful"] += 1
                            self.enrollment_stats["successful_enrollments"] += 1
                        else:
                            self.enrollment_stats["failed_enrollments"] += 1
                    except Exception as e:
                        self.config.logger.error(f"Error retrieving result for {image_path}: {e}")
                        self.enrollment_stats["failed_enrollments"] += 1
        
        self.enrollment_stats["unique_individuals"] = len(set(self.known_face_names))
        
        for person, stats in person_stats.items():
            self.config.logger.info(
                f"Enrolled {stats['successful']}/{stats['processed']} "
                f"images for {person} ({stats['successful'] / max(stats['processed'], 1):.1%})"
            )
        
        if not embeddings_for_db:
            self.config.logger.error("No faces enrolled. Check dataset structure and image quality.")
        else:
            self.config.logger.info(
                f"Enrollment complete: {len(embeddings_for_db)} embeddings, "
                f"{self.enrollment_stats['unique_individuals']} individuals."
            )
        
        return embeddings_for_db
    
    def format_embeddings_for_db(self, embeddings: List[Dict]) -> List[Dict]:
        """Format embeddings for vector database storage."""
        formatted_embeddings = []
        
        for embedding_obj in embeddings:
            try:
                embedding_vector = embedding_obj["embedding"]
                if isinstance(embedding_vector, np.ndarray):
                    embedding_vector = embedding_vector.tolist()
                
                formatted_embedding = {
                    "vector": embedding_vector,
                    "metadata": {
                        "person_name": embedding_obj["person_name"],
                        "image_path": embedding_obj["image_path"],
                        "id": embedding_obj.get("id", ""),
                        "birthday": embedding_obj.get("birthday", ""),
                        "model": embedding_obj.get("model", self.config.face_extraction_model)
                    }
                }
                
                for key, value in embedding_obj.items():
                    if key not in ["embedding", "person_name", "image_path", "id", "birthday", "model"]:
                        formatted_embedding["metadata"][key] = value
                        
                formatted_embeddings.append(formatted_embedding)
            except Exception as e:
                self.config.logger.error(f"Failed to format embedding for {embedding_obj.get('person_name', 'unknown')}: {e}")
        
        return formatted_embeddings