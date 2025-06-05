import psycopg2
from psycopg2.extras import RealDictCursor
import pgvector.psycopg2
from typing import Optional
from ..config import Config

class DatabaseConnection:
    """Handles PostgreSQL database connections and queries."""
    
    def __init__(self, config: Config):
        self.config = config
        self.conn = None
    
    def connect(self) -> Optional[psycopg2.extensions.connection]:
        """Establish a connection to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**self.config.db_config)
            pgvector.psycopg2.register_vector(self.conn)
            self.config.logger.info("Database connected & pgvector registered.")
            return self.conn
        except Exception as e:
            self.config.logger.error(f"Database connection failed: {e}")
            return None
    
    def load_known_faces(self) -> list[dict]:
        """Load face embeddings from the database."""
        known_faces = []
        conn = self.connect()
        if not conn:
            self.config.logger.error("No database connection for loading faces.")
            return known_faces
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    f"SELECT {self.config.column_person_name}, {self.config.column_person_id}, "
                    f"{self.config.column_embedding} FROM {self.config.face_embeddings_table} "
                    f"WHERE {self.config.column_model} = %s",
                    (self.config.face_extraction_model,)
                )
                rows = cur.fetchall()
                expected_dims = self._get_expected_dimensions()
                
                for row in rows:
                    try:
                        emb = row[self.config.column_embedding]
                        if not isinstance(emb, np.ndarray):
                            emb = self._parse_embedding(str(emb))
                        if expected_dims != -1 and (emb.ndim != 1 or emb.shape[0] != expected_dims):
                            self.config.logger.warning(
                                f"Bad embedding dim for {row.get(self.config.column_person_name, 'N/A')}. "
                                f"Shape: {emb.shape}. Skip."
                            )
                            continue
                        known_faces.append({
                            "person_name": row[self.config.column_person_name],
                            "person_id": row[self.config.column_person_id],
                            "embedding": emb.astype(np.float32)
                        })
                    except Exception as e:
                        self.config.logger.warning(
                            f"Parse fail for {row.get(self.config.column_person_name, 'N/A')}: {e}. Skip."
                        )
            self.config.logger.info(f"Loaded {len(known_faces)} faces for model '{self.config.face_extraction_model}'.")
        except Exception as e:
            self.config.logger.error(f"Database load error: {e}")
        finally:
            if conn:
                conn.close()
        return known_faces
    
    def _get_expected_dimensions(self) -> int:
        """Return expected embedding dimensions for the model."""
        dims = {
            "ArcFace": 512, "VGG-Face": 2622, "Facenet": 128, "Facenet512": 512,
            "SFace": 128, "OpenFace": 128, "DeepFace": 4096, "DeepID": 160, "Dlib": 128
        }
        return dims.get(self.config.face_extraction_model, -1)
    
    def _parse_embedding(self, emb_str: str) -> np.ndarray:
        """Parse embedding string to numpy array."""
        import ast
        if emb_str.startswith("vector:"):
            emb_str = emb_str.split(":", 1)[1]
        try:
            return np.array(ast.literal_eval(emb_str), dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Parse error: '{emb_str[:30]}...': {e}")