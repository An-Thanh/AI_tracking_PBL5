from pathlib import Path
import json
from typing import Dict, Optional
import logging

def load_metadata(json_file_path: Path, logger: logging.Logger) -> Dict[str, Dict]:
    """Load person metadata from JSON file."""
    try:
        if json_file_path.exists():
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            logger.info(f"Loaded metadata from {json_file_path}")
        else:
            logger.warning(f"Metadata file {json_file_path} not found. Using fallback data.")
            json_data = [
                {"id": "01", "person_name": "Phuc", "birthday": "08/07/2004"},
                {"id": "02", "person_name": "An", "birthday": "01/01/2004"}
            ]
        
        metadata = {
            person["person_name"]: {
                "id": person.get("id", ""),
                "birthday": person.get("birthday", ""),
                **{k: v for k, v in person.items() if k not in ["id", "person_name", "birthday"]}
            }
            for person in json_data
        }
        logger.info(f"Loaded metadata for {len(metadata)} persons")
        return metadata
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return {}