# utils.py

import numpy as np
import logging
from typing import Dict



import cv2

from .video_processor import VideoFrameLoader
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from . import config
logger = logging.getLogger(__name__)

def is_isolated(frame_data: Dict, object_id: str, entity_type: str, distance_threshold: float = 30.0) -> bool:
    target = frame_data.get(entity_type, {}).get(object_id)
    if not target:
        return False
    target_center = target.get('center', [])
    if len(target_center) != 2:
        return False
    for other_type in ['player', 'referee', 'goalkeeper']:
        for other_id, other_obj in frame_data.get(other_type, {}).items():
            if other_id == object_id and other_type == entity_type:
                continue
            other_center = other_obj.get('center', [])
            if len(other_center) != 2:
                continue
            distance = np.linalg.norm(np.array(target_center) - np.array(other_center))
            if distance < distance_threshold:
                return False
    return True




def is_isolated_with_score(frame_data: Dict, object_id: str, entity_type: str, distance_threshold: float = 30.0) -> bool:
    """
    Enhanced isolation check with scoring mechanism.
    Returns True if the object is isolated beyond the threshold,
    and also calculates an isolation score for the frame.
    """
    target = frame_data.get(entity_type, {}).get(object_id)
    if not target:
        return False, 0.0  # Return score of 0 if target not found
    
    target_center = target.get('center', [])
    target_bbox = target.get('bbox', [])
    if len(target_center) != 2 or len(target_bbox) != 4:
        return False, 0.0  # Invalid data format
    
    # Calculate isolation score components
    score_components = {
        'min_distance': float('inf'),  # Lower is better
        'nearby_objects': 0,           # Fewer is better
        'size_ratio': 1.0              # Closer to 1 is better
    }
    
    for other_type in ['player', 'referee', 'goalkeeper']:
        for other_id, other_obj in frame_data.get(other_type, {}).items():
            if other_id == object_id and other_type == entity_type:
                continue
            
            other_center = other_obj.get('center', [])
            other_bbox = other_obj.get('bbox', [])
            if len(other_center) != 2 or len(other_bbox) != 4:
                continue
            
            # Calculate distance
            distance = np.linalg.norm(np.array(target_center) - np.array(other_center))
            if distance < score_components['min_distance']:
                score_components['min_distance'] = distance
                
            if distance < distance_threshold:
                score_components['nearby_objects'] += 1
                
            # Calculate size ratio
            target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
            other_area = (other_bbox[2] - other_bbox[0]) * (other_bbox[3] - other_bbox[1])
            if other_area > 0:
                size_ratio = target_area / other_area
                score_components['size_ratio'] = min(size_ratio, 1.0 / size_ratio)
    
    distance_score = np.clip(1.0 - (score_components['min_distance'] / distance_threshold), 0, 1)
    
    # Dynamic threshold based on frame characteristics
    isolation_score = (
        0.5 * distance_score +
        0.3 * (1.0 / (1.0 + score_components['nearby_objects'])) +
        0.2 * score_components['size_ratio']
    )
    
    # Adaptive threshold based on number of objects
    threshold = 0.7 - (0.05 * len(frame_data.get(entity_type, {})))
    threshold = max(0.5, min(0.7, threshold))
    
    return isolation_score > threshold, isolation_score






def extract_isolated_object_crops(video_path: str, tracking_data: List[Dict], entity_types: List[str],
                                  metadata_video , max_frames: int = 20, frame_step: int = 1, start_frame: int = 0,
                                  distance_threshold: float = 25, ) -> Dict[str, List[np.ndarray]]:
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
    
    video_loader  = VideoFrameLoader(video_path)
    isolated_crops = {entity: [] for entity in entity_types}

    if metadata_video is not None:
        invalid_frames = get_invalid_frames(metadata_video, video_loader.frame_rate)
    else:
        invalid_frames = None

    frame_indices = list(range(start_frame, start_frame + max_frames, frame_step))
    BATCH_SIZE = config.BATCH_SIZE
    executor = ThreadPoolExecutor(max_workers=1)
    future = None

    for i in range(0, len(frame_indices), BATCH_SIZE):
        batch_idx = frame_indices[i:i + BATCH_SIZE]
        if future is None:
            frames = video_loader.get_batch(batch_idx)
        else:
            frames = future.result()
        next_idx = frame_indices[i + BATCH_SIZE:i + 2 * BATCH_SIZE]
        future = executor.submit(video_loader.get_batch, next_idx) if next_idx else None

        for frame_idx, frame_image in zip(batch_idx, frames):
            frame_data = tracking_data[frame_idx - start_frame] if frame_idx - start_frame < len(tracking_data) else {}
            if (frame_image is None) or (not frame_data):
                continue
            if invalid_frames is not None and frame_idx in invalid_frames:
                continue

            for entity_type in entity_types:
                entities = frame_data.get(entity_type, {})
                for entity_id, entity_info in entities.items():
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

    if future is not None:
        future.result()
    executor.shutdown(wait=True)
    video_loader.release()
    return isolated_crops





# def extract_isolated_object_crops(video_path: str, tracking_data: List[Dict], entity_types: List[str],
#                                   metadata_video , max_frames: int = 2000, frame_step: int = 1, start_frame: int = 0,
#                                   distance_threshold: float = 25, ) -> Dict[str, List[np.ndarray]]:
#     """
#     Extract crops of isolated objects (players, referees, goalkeepers) from a video.
#     When metadata exists, extracts 1000 images from each half.
#     When metadata doesn't exist, extracts 2000 images total.

#     Args:
#         video_path (str): Path to the video file.
#         tracking_data (List[Dict]): Tracking data for frames.
#         entity_types (List[str]): List of entity types to process (e.g., ["player", "referee", "goalkeeper"]).
#         metadata_video (Dict): Video metadata containing half timings.
#         max_frames (int): Maximum number of frames to process (default 2000).
#         frame_step (int): Process every `frame_step` frames.
#         start_frame (int): Starting frame for processing.
#         distance_threshold (float): Distance threshold for determining isolation.

#     Returns:
#         Dict[str, List[np.ndarray]]: Dictionary of isolated crops categorized by entity type.
#     """
    
#     video_loader = VideoFrameLoader(video_path)
#     isolated_crops = {entity: [] for entity in entity_types}

#     def time_to_frame(time_str: str) -> int:
#         """Convert time string (MM:SS) to frame number"""
#         if not time_str:
#             return None
#         minutes, seconds = map(int, time_str.split(':'))
#         return int((minutes * 60 + seconds) * video_loader.frame_rate)

#     # Define frame ranges based on metadata
#     if metadata_video is not None:
#         # Get first half frames
#         first_half_start = time_to_frame(metadata_video.get('start_first_half', '00:00'))
#         first_half_end = time_to_frame(metadata_video.get('end_first_half', '45:00'))
        
#         # Get second half frames
#         second_half_start = time_to_frame(metadata_video.get('start_second_half', '45:00'))
#         second_half_end = time_to_frame(metadata_video.get('end_second_half', '90:00'))

#         # Calculate frame steps for each half to get approximately 1000 images per half
#         first_half_frames = first_half_end - first_half_start
#         second_half_frames = second_half_end - second_half_start
        
#         first_half_step = max(1, first_half_frames // 1000)
#         second_half_step = max(1, second_half_frames // 1000)

#         frame_ranges = [
#             (first_half_start, first_half_end, first_half_step),
#             (second_half_start, second_half_end, second_half_step)
#         ]
#     else:
#         # Without metadata, process entire video
#         total_frames = len(tracking_data)
#         frame_step = max(1, total_frames // 2000)  # Adjust step to get ~2000 images
#         frame_ranges = [(start_frame, start_frame + total_frames, frame_step)]

#     # Process frames according to ranges
#     for start, end, step in frame_ranges:
#         if start is None or end is None:
#             continue
            
#         for frame_idx in range(start, end, step):
#             if frame_idx >= len(tracking_data):
#                 break
                
#             frame_data = tracking_data[frame_idx]
#             frame_image = video_loader.get_frame(frame_idx)
            
#             if (frame_image is None) or (not frame_data):
#                 continue

#             for entity_type in entity_types:
#                 entities = frame_data.get(entity_type, {})
#                 for entity_id, entity_info in entities.items():
#                     bbox = entity_info.get("bbox")
#                     if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
#                         x_min, y_min, x_max, y_max = map(int, bbox)
#                         h, w, _ = frame_image.shape
#                         x_min = max(0, x_min)
#                         y_min = max(0, y_min)
#                         x_max = min(w, x_max)
#                         y_max = min(h, y_max)
#                         if x_min < x_max and y_min < y_max:
#                             crop = frame_image[y_min:y_max, x_min:x_max]
#                             isolated_crops[entity_type].append(crop)

#     video_loader.release()
    # return isolated_crops




def get_invalid_frames(metadata: Dict, fps: int) -> List[int]:
    """
    Identify invalid frames based on metadata and FPS.
    Invalid frames are those that fall outside the defined time zones in metadata.
    
    Args:
        metadata (Dict): Dictionary containing video metadata with time zones
        fps (int): Frames per second of the video
        
    Returns:
        List[int]: List of frame numbers that should be considered invalid
    """
    def time_to_frame(time_str: str) -> int:
        """Convert time string (MM:SS) to frame number"""
        if not time_str:
            return None
        minutes, seconds = map(int, time_str.split(':'))
        return int((minutes * 60 + seconds) * fps)

    # Define valid time zones
    valid_zones = []
    
    # First half
    start_first = time_to_frame(metadata.get('start_first_half'))
    end_first = time_to_frame(metadata.get('end_first_half'))
    if start_first is not None and end_first is not None:
        valid_zones.append((start_first, end_first))
    
    # Second half
    start_second = time_to_frame(metadata.get('start_second_half'))
    end_second = time_to_frame(metadata.get('end_second_half'))
    if start_second is not None and end_second is not None:
        valid_zones.append((start_second, end_second))
    
    # Extra time first half
    start_extra_first = time_to_frame(metadata.get('start_extra_time_first'))
    end_extra_first = time_to_frame(metadata.get('end_extra_time_first'))
    if start_extra_first is not None and end_extra_first is not None:
        valid_zones.append((start_extra_first, end_extra_first))
    
    # Extra time second half
    start_extra_second = time_to_frame(metadata.get('start_extra_time_second'))
    end_extra_second = time_to_frame(metadata.get('end_extra_time_second'))
    if start_extra_second is not None and end_extra_second is not None:
        valid_zones.append((start_extra_second, end_extra_second))
    
    # Penalty shootout
    start_penalty = time_to_frame(metadata.get('start_penalty_shootout'))
    end_penalty = time_to_frame(metadata.get('end_penalty_shootout'))
    if start_penalty is not None and end_penalty is not None:
        valid_zones.append((start_penalty, end_penalty))
    
    # If no valid zones defined, return empty list
    if not valid_zones:
        return []
    
    # Sort zones by start time
    valid_zones.sort(key=lambda x: x[0])
    
    # Find invalid frames
    invalid_frames = []
    current_frame = 0
    
    for start, end in valid_zones:
        # Add frames before this zone
        invalid_frames.extend(range(current_frame, start))
        current_frame = end + 1
    
    return invalid_frames




