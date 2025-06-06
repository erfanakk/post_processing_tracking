# data_loader.py

import cv2
import numpy as np
from .video_processor import VideoFrameLoader
from .utils_post import is_isolated
from typing import List, Dict

import json
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

def load_json2(file_path: str) -> Dict:
    try:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        logger.info(f"Loaded JSON data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON data from {file_path}: {e}")
        raise e


def load_json(file_path: str) -> Dict:
    try:

        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON data from {file_path}: {e}")
        raise e



def save_json(data: Dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as f:
            if isinstance(data, list):
                # Handle list of dictionaries (JSONL format)
                for item in data:
                    f.write(json.dumps(item) + '\n')
            else:
                # Handle single dictionary
                f.write(json.dumps(data) + '\n')
        logger.info(f"Saved JSON data to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON data to {file_path}: {e}")
        raise e
def extract_isolated_object_crops(video_path: str, tracking_data: List[Dict], entity_types: List[str],
                                  max_frames: int = 20, frame_step: int = 1, start_frame: int = 0,
                                  distance_threshold: float = 25) -> Dict[str, List[np.ndarray]]:
    """
    Extract crops of isolated objects (players, referees, goalkeepers) from a video.

    Args:
        video_path (str): Path to the video file.
        tracking_data (List[Dict]): Tracking data for frames.
        entity_types (List[str]): List of entity types to process (e.g., ["player", "referee", "goalkeeper"]).
        max_frames (int): Maximum number of frames to process.
        frame_step (int): Process every `frame_step` frames.
        start_frame (int): Starting frame for processing.
        distance_threshold (float): Distance threshold for determining isolation.

    Returns:
        Dict[str, List[np.ndarray]]: Dictionary of isolated crops categorized by entity type.
    """
    video_loader = VideoFrameLoader(video_path)
    isolated_crops = {entity: [] for entity in entity_types}

    for frame_idx in range(start_frame, start_frame + max_frames, frame_step):
        frame_data = tracking_data[frame_idx - start_frame] if frame_idx - start_frame < len(tracking_data) else {}
        frame_image = video_loader.get_frame(frame_idx)
        if frame_image is None or not frame_data:
            continue

        for entity_type in entity_types:
            entities = frame_data.get(entity_type, {})
            for entity_id, entity_info in entities.items():
                if is_isolated(frame_data, entity_id, entity_type, distance_threshold):
                    bbox = entity_info.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        h, w, _ = frame_image.shape
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(w, x_max)
                        y_max = min(h, y_max)
                        if x_min < x_max and y_min < y_max:
                            crop = frame_image[y_min:y_max, x_min:x_max]
                            isolated_crops[entity_type].append(crop)
    video_loader.release()
    return isolated_crops


