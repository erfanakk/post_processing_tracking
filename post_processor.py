#post_processor.py

import math 
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from . import config
from .data_loader import load_json, save_json
from .video_processor import VideoFrameLoader
from .utils_post import is_isolated , is_isolated_with_score
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from collections import defaultdict
import random
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import cv2
import math
from cachetools import LRUCache
import gc


from scipy.spatial.distance import cdist


random.seed(42)
logger = logging.getLogger(__name__)


import numpy as np
from sklearn.cluster import DBSCAN

from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TrackletSplitter:
    """
    Tracklet Splitter component for the GTA (Global Tracklet Association) method.
    This class implements the first stage of the GTA method, which splits tracklets

    
    The splitter uses appearance features extracted by a ReID model and clusters them
    using DBSCAN to identify potential ID swaps within a single tracklet.
    """
    
    def __init__(self, reid_model, eps=0.4, min_samples=3, min_cluster_size=3):
        """
        Initialize the TrackletSplitter.
        
        Args:
            reid_model: ReID model for extracting appearance features
            eps: DBSCAN epsilon parameter (maximum distance between samples)
            min_samples: DBSCAN min_samples parameter (min points to form dense region)
            min_cluster_size: Minimum size of a cluster to be considered valid
        """
        self.reid_model = reid_model
        self.eps = eps
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size


    def split_tracklet(
        self,
        entity_type: str,
        track_id: str,
        tracking_data: list,
        video_loader,              # must have .get_batch_safe()
        batch_size: int = 64       # tune to fit your RAM/GPU
    ) -> dict:
        """
        Returns {frame_num: new_track_id} if the tracklet should be split,
        else an empty dict.  All frames are decoded in memory-bounded chunks
        via VideoFrameLoader.get_batch_safe().
        """
        logger.info(f"Analyzing {track_id} ({entity_type}) for splitting")

        # -----------------------------------------------------------------
        # 1. collect every frame idx where this ID is present
        # -----------------------------------------------------------------
        frames = [
            f for f, fdata in enumerate(tracking_data)
            if track_id in fdata.get(entity_type, {})
        ]
        if len(frames) < self.min_samples * 2:
            logger.debug(f"{track_id}: only {len(frames)} frames – skip")
            return {}

        # -----------------------------------------------------------------
        # 2. decode those frames in CHUNKS
        #    (get_batch_safe yields (idx, image) pairs)
        # -----------------------------------------------------------------
        # 2. decode those frames in CHUNKS
        sorted_unique = sorted(set(frames))

        # use a thread pool to preload next chunk
        executor = ThreadPoolExecutor(max_workers=1)
        future = None

        crops, crop_frames = [], []
        for i in range(0, len(sorted_unique), batch_size):
            batch = sorted_unique[i:i + batch_size]
            if future is None:
                imgs = video_loader.get_batch(batch)
            else:
                imgs = future.result()

            next_batch = sorted_unique[i + batch_size:i + 2 * batch_size]
            future = executor.submit(video_loader.get_batch, next_batch) if next_batch else None

            for fnum, img in zip(batch, imgs):
                bbox = tracking_data[fnum][entity_type][track_id].get("bbox")
                if not (bbox and len(bbox) == 4):
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                if img is None:
                    continue

                h, w, _ = img.shape
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crops.append(img[y1:y2, x1:x2])
                    crop_frames.append(fnum)

        if future is not None:
            future.result()
        executor.shutdown(wait=True)

        if len(crops) < self.min_samples * 2:
            logger.debug(f"{track_id}: only {len(crops)} valid crops – skip")
            return {}

        # -----------------------------------------------------------------
        # 4. Re-ID features  ➜  DBSCAN clustering
        # -----------------------------------------------------------------
        feats   = self.reid_model.extract_features(crops)
        labels  = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(feats)

        # keep clusters with ≥ min_cluster_size samples (ignore label -1)
        clusters = {
            lbl for lbl in set(labels)
            if lbl != -1 and (labels == lbl).sum() >= self.min_cluster_size
        }
        if len(clusters) <= 1:
            logger.debug(f"{track_id}: ≤1 cluster – no split")
            return {}

        # -----------------------------------------------------------------
        # 5. build {frame_num: new_track_id}
        # -----------------------------------------------------------------
        largest = max(clusters, key=lambda l: (labels == l).sum())
        frame2id = {}
        for lbl in clusters:
            new_id = track_id if lbl == largest else f"{track_id}_split_{lbl}"
            for i, fnum in enumerate(crop_frames):
                if labels[i] == lbl:
                    frame2id[fnum] = new_id

        logger.info(
            f"Split {track_id} into {len(set(frame2id.values()))} tracklets "
            f"(clusters={len(clusters)})"
        )
        return frame2id


    # def split_tracklet(self, entity_type, track_id, tracking_data, video_loader):
    #     """
    #     Split a tracklet if it contains multiple identities.
        
    #     Args:
    #         entity_type: Type of entity (player, referee, etc.)
    #         track_id: ID of the tracklet to split
    #         tracking_data: List of frame data dictionaries
    #         video_loader: VideoFrameLoader instance for accessing video frames
            
    #     Returns:
    #         dict: Dictionary mapping frame numbers to new track IDs
    #              Empty dict if no split is needed
    #     """
    #     logger.info(f"Analyzing tracklet {track_id} of type {entity_type} for potential splits")
        
    #     # 1. Collect all frames where this tracklet appears
    #     frames = []
    #     for frame_num, frame_data in enumerate(tracking_data):
    #         if entity_type in frame_data and track_id in frame_data[entity_type]:
    #             frames.append(frame_num)
        
    #     if len(frames) < self.min_samples * 2:  # Need enough frames to potentially form at least 2 clusters
    #         logger.debug(f"Tracklet {track_id} has too few frames ({len(frames)}) for splitting")
    #         return {}
        
    #     # 2. Extract crops for each frame
    #     crops = []
    #     frame_indices = []
    #     for frame_num in frames:
    #         frame_data = tracking_data[frame_num]
    #         entity_data = frame_data.get(entity_type, {}).get(track_id, {})
    #         bbox = entity_data.get("bbox")
            
    #         if bbox and len(bbox) == 4:
    #             try:
    #                 x_min, y_min, x_max, y_max = map(int, bbox)
    #                 frame_image = video_loader.get_frame(frame_num)
                    
    #                 if frame_image is None:
    #                     continue
                        
    #                 h, w, _ = frame_image.shape
    #                 x_min, x_max = max(0, x_min), min(w, x_max)
    #                 y_min, y_max = max(0, y_min), min(h, y_max)
                    
    #                 if x_min < x_max and y_min < y_max:
    #                     crop = frame_image[y_min:y_max, x_min:x_max]
    #                     crops.append(crop)
    #                     frame_indices.append(frame_num)
    #             except Exception as e:
    #                 logger.error(f"Error extracting crop for frame {frame_num}, tracklet {track_id}: {str(e)}")
    #                 continue
        
    #     if len(crops) < self.min_samples * 2:
    #         logger.debug(f"Tracklet {track_id} has too few valid crops ({len(crops)}) for splitting")
    #         return {}
            
    #     # 3. Extract ReID features for each crop
    #     features = self.reid_model.extract_features(crops)

    #     # 4. Cluster features using DBSCAN
    #     clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(features)
    #     labels = clustering.labels_
        
    #     # Count number of samples in each cluster
    #     unique_labels = set(labels)
    #     if -1 in unique_labels:  # Remove noise label
    #         unique_labels.remove(-1)
            
    #     # If only one cluster or no valid clusters, no need to split
    #     if len(unique_labels) <= 1:
    #         logger.debug(f"Tracklet {track_id} does not need splitting (found {len(unique_labels)} clusters)")
    #         return {}
            
    #     # Check if clusters are large enough
    #     cluster_sizes = {}
    #     for label in unique_labels:
    #         size = np.sum(labels == label)
    #         cluster_sizes[label] = size
            
    #     valid_clusters = [label for label, size in cluster_sizes.items() if size >= self.min_cluster_size]
        
    #     if len(valid_clusters) <= 1:
    #         logger.debug(f"Tracklet {track_id} does not have enough valid clusters for splitting")
    #         return {}
            
    #     # 5. Create new track IDs for each cluster
    #     logger.info(f"Splitting tracklet {track_id} into {len(valid_clusters)} new tracklets")
        
    #     # Map frames to new track IDs
    #     frame_to_new_id = {}
        
    #     # Keep the original ID for the largest cluster
    #     largest_cluster = max(valid_clusters, key=lambda x: cluster_sizes[x])
    #     new_track_ids = {}
        
    #     for i, label in enumerate(valid_clusters):
    #         if label == largest_cluster:
    #             new_id = track_id  # Keep original ID for largest cluster
    #         else:
    #             # Generate a new unique ID
    #             new_id = f"{track_id}_split_{i}"
                
    #         new_track_ids[label] = new_id
            
    #         # Assign frames to new track IDs
    #         for idx, frame_num in enumerate(frame_indices):
    #             if labels[idx] == label:
    #                 frame_to_new_id[frame_num] = new_id
        
    #     logger.info(f"Split tracklet {track_id} into {len(valid_clusters)} parts: {new_track_ids}")
    #     return frame_to_new_id
        
    def apply_splits(self, entity_type, tracking_data, video_loader, summary_data=None):
        """
        Apply tracklet splitting to all tracklets of a given entity type.
        
        Args:
            entity_type: Type of entity (player, referee, etc.)
            tracking_data: List of frame data dictionaries
            video_loader: VideoFrameLoader instance for accessing video frames
            summary_data: Optional summary data to update
            
        Returns:
            tuple: (updated_tracking_data, updated_summary_data, split_info)
                  where split_info contains statistics about the splits
        """
        # Find all unique track IDs for this entity type
        track_ids = set()
        for frame_data in tracking_data:
            if entity_type in frame_data:
                track_ids.update(frame_data[entity_type].keys())
        
        # Process each tracklet
        split_info = {
            "total_tracklets": len(track_ids),
            "split_tracklets": 0,
            "new_tracklets": 0
        }
        
        # We need to process one tracklet at a time to avoid ID conflicts
        for track_id in sorted(track_ids):
            # Skip already split tracklets (those with "_split_" in the name)
            if "_split_" in track_id:
                continue
                
            # Try to split this tracklet
            frame_to_new_id = self.split_tracklet(entity_type, track_id, tracking_data, video_loader)
            
            if not frame_to_new_id:
                continue  # No split needed
                
            # Count unique new IDs (excluding the original ID)
            new_ids = set(frame_to_new_id.values())
            if track_id in new_ids:
                new_ids.remove(track_id)
                
            split_info["split_tracklets"] += 1
            split_info["new_tracklets"] += len(new_ids)
            
            # Apply the split to tracking data
            self._apply_split_to_tracking_data(entity_type, track_id, frame_to_new_id, tracking_data)
            
            # Update summary data if provided
            if summary_data:
                self._update_summary_data(entity_type, track_id, frame_to_new_id, tracking_data, summary_data)
        
        logger.info(f"Split {split_info['split_tracklets']} tracklets, created {split_info['new_tracklets']} new tracklets")
        return tracking_data, summary_data, split_info
    
    def _apply_split_to_tracking_data(self, entity_type, track_id, frame_to_new_id, tracking_data):
        """
        Apply the split to the tracking data.
        
        Args:
            entity_type: Type of entity
            track_id: Original track ID
            frame_to_new_id: Dictionary mapping frame numbers to new track IDs
            tracking_data: List of frame data dictionaries
        """
        # Process each frame where we need to change the track ID
        for frame_num, new_id in frame_to_new_id.items():
            if new_id == track_id:
                continue  # No change needed for this frame
                
            # Get the frame data
            if frame_num >= len(tracking_data):
                continue
                
            frame_data = tracking_data[frame_num]
            
            if entity_type not in frame_data or track_id not in frame_data[entity_type]:
                continue
                
            # Create a new entry with the new ID
            entity_data = frame_data[entity_type][track_id].copy()
            frame_data[entity_type][new_id] = entity_data
            
            # Remove the original entry
            del frame_data[entity_type][track_id]
    
    def _update_summary_data(self, entity_type, track_id, frame_to_new_id, tracking_data, summary_data):
        """
        Update the summary data after splitting a tracklet.
        
        Args:
            entity_type: Type of entity
            track_id: Original track ID
            frame_to_new_id: Dictionary mapping frame numbers to new track IDs
            tracking_data: List of frame data dictionaries
            summary_data: Summary data to update
        """
        if entity_type not in summary_data or track_id not in summary_data[entity_type]:
            return
            
        # Get all unique new IDs
        new_ids = set(frame_to_new_id.values())
        
        # If the original ID is still used, we need to update its data
        if track_id in new_ids:
            new_ids.remove(track_id)
            
            # Update the frames for the original ID
            original_frames = [f for f, id_ in frame_to_new_id.items() if id_ == track_id]
            if original_frames:
                summary_data[entity_type][track_id]["first_appearance"] = min(original_frames)
                summary_data[entity_type][track_id]["last_frame_lost"] = max(original_frames)
                
                # Update disappear frames
                disappear_frames = []
                for i in range(min(original_frames), max(original_frames) + 1):
                    if i not in original_frames:
                        disappear_frames.append(i)
                summary_data[entity_type][track_id]["disappear_frame"] = disappear_frames
        
        # Create new entries for new IDs
        for new_id in new_ids:
            # Get frames for this new ID
            new_id_frames = [f for f, id_ in frame_to_new_id.items() if id_ == new_id]
            
            if not new_id_frames:
                continue
                
            # Create a new summary entry
            summary_data[entity_type][new_id] = {
                "first_appearance": min(new_id_frames),
                "last_frame_lost": max(new_id_frames),
                "disappear_frame": [],
                "crop_valid_before": [],
                "crop_valid_after": [],
                "classification_frames": []
            }
            
            # Copy any additional fields from the original track
            for key, value in summary_data[entity_type][track_id].items():
                if key not in summary_data[entity_type][new_id]:
                    summary_data[entity_type][new_id][key] = value
            
            # Update disappear frames
            disappear_frames = []
            for i in range(min(new_id_frames), max(new_id_frames) + 1):
                if i not in new_id_frames:
                    disappear_frames.append(i)
            summary_data[entity_type][new_id]["disappear_frame"] = disappear_frames


class PostProcessor:
    def __init__(
        self,
        tracking_data,
        reid_model,
        team_classifier,
        video_loader,
        similarity_threshold,
        projection_threshold: float = 10.0,
    ):
        self.updated_tracking_path = config.UPDATED_TRACKING_DATA_PATH
        self.updated_summary_path = config.UPDATED_SUMMARY_DATA_PATH

        self.device = config.DEVICE
        self.batch_size = config.BATCH_SIZE

        self.reid_module = reid_model
        self.team_classifier = team_classifier
        self.similarity_threshold = similarity_threshold
        self.projection_threshold = projection_threshold


        self.tracking_data = tracking_data
        self.video_loader = video_loader
        self.frame_cache = LRUCache(maxsize=200)
        self.preload_executor = ThreadPoolExecutor(max_workers=1)

    def get_cached_frame(self, frame_num):
        if frame_num in self.frame_cache:
            return self.frame_cache[frame_num]
        frame = self.video_loader.get_frame(frame_num)
        self.frame_cache[frame_num] = frame
        return frame

    def _preload_frames(self, frame_nums, batch_size=64):
        """Preload frames into the LRU cache using batch decoding."""
        missing = [f for f in frame_nums if f not in self.frame_cache]
        if not missing:
            return
        for idx, img in self.video_loader.get_batch_safe(missing, batch_size):
            self.frame_cache[idx] = img

    def _preload_frames_async(self, frame_nums, batch_size=64):
        """Asynchronously preload frames into the cache."""
        if not frame_nums:
            return None
        return self.preload_executor.submit(self._preload_frames, frame_nums, batch_size)


    #TODO CHANGE THIS FOR TIMEZONE BASED
    def extract_entity_crops(self, entity_type: str, track_id: str, frames: List[int]) -> List[np.ndarray]:
        # self._preload_frames(frames, batch_size=64)
        crops = []
        for frame_num in frames:
            if not (0 <= frame_num < len(self.tracking_data)):
                continue
            frame_data = self.tracking_data[frame_num]
            entity_data = frame_data.get(entity_type, {}).get(str(track_id), {})
            bbox = entity_data.get("bbox")
            if bbox and len(bbox) == 4:
                try:
                    x_min, y_min, x_max, y_max = map(int, bbox)
                except ValueError:
                    continue


                frame_image = self.get_cached_frame(frame_num)
                if frame_image is None:
                    continue
                h, w, _ = frame_image.shape
                x_min, x_max = max(0, x_min), min(w, x_max)
                y_min, y_max = max(0, y_min), min(h, y_max)
                if x_min < x_max and y_min < y_max:
                    crops.append(frame_image[y_min:y_max, x_min:x_max])
        return crops





    def _calculate_average_projection(self, projections: list, track_id: str, context: str) -> Optional[List[float]]:
        """
        Safely compute average (x,y) from a list of projection points.
        """
        if not projections:
            logger.debug(f"No valid projections {context} for {track_id}")
            return None
        try:
            avg_x = sum(p[0] for p in projections) / len(projections)
            avg_y = sum(p[1] for p in projections) / len(projections)
            return [avg_x, avg_y]
        except Exception as e:
            logger.error(f"Error calculating {context} projection for {track_id}: {str(e)}")
            return None
    def compute_average_projection_after_appearance(
        self, entity_type: str, track_id: str, first_frame_present: int, window_size: int = 2
    ) -> Optional[List[float]]:
        """
        Averages the 'projection' field in up to 'window_size' frames after 'first_frame_present'.
        """
        projections = []
        frame = first_frame_present
        collected = 0
        while frame < len(self.tracking_data) and collected < window_size:
            if frame not in self.summary_data[entity_type][track_id].get("disappear_frame", []):
                proj = self.tracking_data[frame].get(entity_type, {}).get(str(track_id), {}).get("projection")
                if proj and len(proj) == 2:
                    projections.append(proj)
                    collected += 1
            frame += 1
        if not projections:
            logger.debug(f"No valid projections after appearance for {track_id}")
            return None
        return [sum(p[0] for p in projections) / len(projections),
                sum(p[1] for p in projections) / len(projections)]

    def compute_average_projection_before_lost(
        self, entity_type: str, track_id: str, last_frame_present: Optional[int], window_size: int = 2
    ) -> Optional[List[float]]:
        """
        Averages the 'projection' field for up to 'window_size' frames before 'last_frame_present'.
        """
        try:
            if last_frame_present is None:
                if not self.tracking_data:
                    return None
                last_frame_present = len(self.tracking_data) - 1

            projections = []
            collected = 0
            frame = last_frame_present
            while frame >= 0 and collected < window_size:
                if frame not in self.summary_data[entity_type][track_id].get("disappear_frame", []):
                    proj = self.tracking_data[frame].get(entity_type, {}).get(str(track_id), {}).get("projection")
                    if proj and len(proj) == 2:
                        projections.append(proj)
                        collected += 1
                frame -= 1
            return self._calculate_average_projection(projections, track_id, "before_lost")
        except Exception as e:
            logger.error(f"Error computing before-lost projection for {track_id}: {str(e)}")
            return None


    def get_lost_stable_ids(self,entity_type , stable_ids, new_first_appearance, age_max , temporal_window=0, time_zone=None ):
        lost_ids = []
        for sid in stable_ids:
            s_meta = self.summary_data[entity_type].get(sid, {})
            s_last_frame_lost = s_meta.get('last_frame_lost')
            time_zone_lost = s_meta.get('time_zone') 
            if s_last_frame_lost is None:
                s_last_frame_lost = len(self.tracking_data) - 1
            if (s_last_frame_lost is not None) and (new_first_appearance > s_last_frame_lost)   and (abs(s_last_frame_lost - new_first_appearance) < age_max) and (time_zone_lost == time_zone):
                lost_ids.append(sid)
        return lost_ids

    ##############################################################################
    # --------------------- Track Similarity ------------------
    ##############################################################################

    def refresh_projection_data(self, entity_type, track_id, meta , window_size):
        """
        If avg projections are missing, compute them with a larger window_size=5.
        """
        
        meta['avg_projection_after_appearance'] = self.compute_average_projection_after_appearance(
            entity_type, track_id, meta['first_appearance'], window_size=window_size
        )
    
        meta['avg_projection_before_lost'] = self.compute_average_projection_before_lost(
            entity_type, track_id, meta.get('last_frame_lost'), window_size=window_size
        )


    ##############################################################################
    # -------------------------- Metadata Creation -------------------------------
    ##############################################################################

    def _create_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Create metadata for each track: first_appearance, last_frame_lost, disappear_frame, etc.
        """
        entity_types = ['goalkeeper', 'player', 'referee', 'ball']
        metadata = {etype: defaultdict(dict) for etype in entity_types}
        presence = {etype: defaultdict(set) for etype in entity_types}

        # Collect frames for each track_id and get time zone
        for frame_num, frame in enumerate(self.tracking_data):

            for etype in entity_types:
                
                for eid in frame.get(etype, {}):
                    zone = frame.get('time_zone') 
                    presence[etype][eid].add(frame_num)
                    # Store time zone in metadata if not already set
                    if eid not in metadata[etype] or 'time_zone' not in metadata[etype][eid]:
                        metadata[etype][eid]['time_zone'] =zone

        # Build summary
        for etype in entity_types:
            for eid, frameset in presence[etype].items():
                sorted_frames = sorted(frameset)
                if not sorted_frames:
                    continue
                first_appearance = sorted_frames[0]
                last_frame_present = sorted_frames[-1]
                disappear_frames = [
                    f for f in range(first_appearance, last_frame_present + 1)
                    if f not in frameset
                ]
                # Get the time zone we stored earlier
                time_zone = metadata[etype][eid].get('time_zone')
                
                metadata[etype][eid] = {
                    "first_appearance": first_appearance,
                    "last_frame_lost": None if last_frame_present == len(self.tracking_data) - 1 else last_frame_present,
                    "disappear_frame": disappear_frames,
                    "crop_valid_before": [],
                    "crop_valid_after": [],
                    "classification_frames": [],
                    "time_zone": time_zone  # Include the time zone in final metadata
                }
        return metadata
    def _select_frames(
        self,
        track_id: str,
        entity_type: str,
        track_frames: List[int],
        disappear_frames: List[int],
        isolated_ther: float,
        max_select: int
    ) -> List[int]:
        """
        Picks up to max_select frames from track_frames randomly, without considering isolation.
        """
        valid_frames = [f for f in track_frames if f not in disappear_frames]
        # Shuffle the valid frames randomly
        random.shuffle(valid_frames)
        # Select up to max_select frames
        selected = valid_frames[:max_select]
        # Remove duplicates while preserving order
        return list(dict.fromkeys(selected))[:max_select]
    def _create_crop_valid_attributes(self, 
        isolated_ther: int = 10,
        isolated_ther_classifciation: int = 10,
        max_select: int = 3,
        max_select_classification: int = 5,
        window_size: int = 3,
        ):
        """
        For each track, fill in the frames for crop_valid_before/after + classification_frames.
        """
        for entity_type, tracks in self.summary_data.items():
            for track_id, meta in tracks.items():

                present_frames = sorted(
                    fnum for fnum, frame in enumerate(self.tracking_data)
                    if track_id in frame.get(entity_type, {})
                )
                # present_frames = sorted(
                #         frame.get("frame_num") for frame in self.tracking_data
                #         if track_id in frame.get(entity_type, {})
                #     )

                if not present_frames:
                    continue
                # Last few -> crop_valid_before
                before_frames = present_frames[-window_size:]
                meta['crop_valid_before'] = self._select_frames(
                    track_id, entity_type, before_frames, [], isolated_ther=isolated_ther, max_select=max_select
                )
                # First few -> crop_valid_after
                after_frames = present_frames[:window_size]
                meta['crop_valid_after'] = self._select_frames(
                    track_id, entity_type, after_frames, [], isolated_ther=isolated_ther, max_select=max_select
                )
                # classification_frames (larger sample)
                meta['classification_frames'] = self._select_frames(
                    track_id, entity_type, present_frames, [], isolated_ther=isolated_ther_classifciation, max_select=max_select_classification
                )

    ##############################################################################
        # ----------------------- Updating Clubs & Summaries -------------------------
    ##############################################################################
    def update_club_info(self, track_id: str, assigned_team: str,conf: float, entity_type: str) -> None:
        """
        Sets 'club' and 'club_color' for the track_id across all frames + summary.
        """
        team_colors = {
            "Club1": [255, 0, 0],
            "Club2": [0, 0, 255],
            "Unknown": [0, 0, 0],
        }
        for frame_data in self.tracking_data:
            entity_data = frame_data.get(entity_type, {})
            if track_id in entity_data:
                entity_data[track_id]["club"] = assigned_team
                entity_data[track_id]["club_color"] = team_colors.get(assigned_team, [0, 0, 0])
        self.summary_data[entity_type][track_id]["club"] = assigned_team
        self.summary_data[entity_type][track_id]["club_conf"] = conf
        # logger.info(f"for this track id {track_id} we got this team classficition {assigned_team}")



    #--------------***------------
    def reassign_track(self, new_tid: str, existing_tid: str, entity_type: str):
        """
        Moves references of new_tid -> existing_tid in tracking_data, merges summary metadata.
        """
        # 1) Update tracking_data
        for frame_data in self.tracking_data:
            edict = frame_data.get(entity_type, {})
            if new_tid in edict:
                edict[existing_tid] = edict.pop(new_tid)

        # 2) Merge & remove new_tid from summary
        lost_meta = self.summary_data[entity_type].pop(new_tid, None)
        if not lost_meta:
            return
        active_meta = self.summary_data[entity_type].get(existing_tid, {})

        # earliest start
        active_meta["first_appearance"] = min(
            active_meta.get("first_appearance", float("inf")),
            lost_meta["first_appearance"]
        )
        # last frame lost
        c_last = active_meta.get("last_frame_lost")
        n_last = lost_meta.get("last_frame_lost")
        if None in (c_last, n_last):
            active_meta["last_frame_lost"] = None
        else:
            active_meta["last_frame_lost"] = max(c_last, n_last)
        # unify disappear frames
        dis = set(active_meta.get("disappear_frame", [])) | set(lost_meta.get("disappear_frame", []))
        active_meta["disappear_frame"] = sorted(dis)
        if lost_meta.get("crop_valid_before"):
            active_meta["crop_valid_before"] = lost_meta["crop_valid_before"]
        # refresh new projections
        active_meta["avg_projection_after_appearance"] = self.compute_average_projection_after_appearance(
            entity_type, existing_tid, active_meta["first_appearance"]
        )
        active_meta["avg_projection_before_lost"] = self.compute_average_projection_before_lost(
            entity_type, existing_tid, active_meta["last_frame_lost"]
        )
        self.summary_data[entity_type][existing_tid] = active_meta

    ##############################################################################
    # ------------------------- Merge Criteria & Scores --------------------------
    ##############################################################################
    def verify_merge_criteria(self,similarity ,projection_dist, use_sim) :
        """Enhanced verification with new team classification"""
        if projection_dist > self.projection_threshold :
            return False
        if use_sim:
            if similarity < self.similarity_threshold:   
                return False
        return True
    def calculate_combined_score(self, entity_type: str, track_id: str, candidate_id: str, projection_dist: float,wp, similarity: float,ws: float , team=False,wt=0.1) -> float:
        """
        Weighted cost for step_2, includes team match factor.
        """
        norm_proj = min(projection_dist / self.projection_threshold, 1.0)
        norm_sim = 1 - similarity

        cost = (wp * norm_proj + ws * norm_sim)
        if team:
            track_team = self.summary_data[entity_type][track_id]["club"]
            cand_team = self.summary_data[entity_type][candidate_id]["club"]
            if track_team == cand_team:
                team_score = 1.0
            elif "Unknown" in [track_team, cand_team]:
                team_score = 0.5
            else:
                team_score = 0.0
            norm_team = 1 - team_score
            cost += (wt * norm_team)
            logger.info(f"the cost {cost} for track id {track_id} and this candidate id for reassign {candidate_id}")
        return cost
    def get_track_projection_distance(self, entity_type: str, track_id1: str, track_id2: str) -> float:
        """
        Distance between track1's avg_projection_after_appearance and
        track2's avg_projection_before_lost. Returns self.max_projection_distance if missing.
        """
        proj1 = self.summary_data[entity_type][track_id1].get("avg_projection_after_appearance")
        proj2 = self.summary_data[entity_type][track_id2].get("avg_projection_before_lost")
        if not proj1 or not proj2:
            logger.debug(f"Missing projection for {track_id1} or {track_id2}")
            return self.projection_threshold
        dx, dy = proj2[0] - proj1[0], proj2[1] - proj1[1]
        return math.hypot(dx, dy)
    
    def get_track_similarity(self, entity_type: str, track_id1: str, track_id2: str) -> float:
        """
        Computes the similarity between two track IDs based on their ReID features.

        Args:
            entity_type (str): The type of entity (e.g., "player").
            track_id1 (str): The first track ID.
            track_id2 (str): The second track ID.

        Returns:
            float: Cosine similarity score between the two track IDs.
        """
        def get_all_features(tid: str , SET:str) -> Optional[np.ndarray]:
            track_meta = self.summary_data.get(entity_type, {}).get(tid, {})
            if not track_meta:
                return None
            frames_before = track_meta.get('crop_valid_before', [])
            frames_after = track_meta.get('crop_valid_after', [])
            if SET == "BEFORE":
                all_frames = frames_before 
            else:
                all_frames =  frames_after
            if not all_frames:
                return None

            crops = []
            for f_num in all_frames:
                if 0 <= f_num < len(self.tracking_data):
                    frame_data = self.tracking_data[f_num]
                    obj_data = frame_data.get(entity_type, {}).get(tid, {})
                    bbox = obj_data.get('bbox')
                    if bbox and len(bbox) == 4:
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        # Use cached frame if available
                        # if f_num in self.frame_cache:
                        #     frame_img = self.frame_cache[f_num]
                        # else:
                        #     frame_img = self.video_loader.get_frame(f_num)
                        #     self.frame_cache[f_num] = frame_img

                        frame_img = self.get_cached_frame(f_num)

                        if frame_img is not None:
                            h, w, _ = frame_img.shape
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)
                            x_max = min(w, x_max)
                            y_max = min(h, y_max)
                            if x_min < x_max and y_min < y_max:
                                crop = frame_img[y_min:y_max, x_min:x_max]
                                if crop.size > 0:
                                    crops.append(crop)
            if not crops:
                return None
            # Extract features for all crops
            features = self.reid_module.extract_features(crops)
            if features is None or len(features) == 0:
                return None
            return features

        feat_set_1 = get_all_features(track_id1 , "AFTER")
        feat_set_2 = get_all_features(track_id2 , 'BEFORE')
        if feat_set_1 is None or feat_set_2 is None:
            return 0.0
        sim_matrix = cosine_similarity(feat_set_1, feat_set_2)
        #weighted average favoring higher similarities
        weights = np.exp(sim_matrix) / np.sum(np.exp(sim_matrix))
        final_similarity = float(np.sum(sim_matrix * weights))
        return final_similarity

    ##############################################################################
    # ----------------- Single Function Handling Step 1 or Step 2 ---------------
    ##############################################################################
    def reassign_tracks(self, entity_types: List[str] = ['player'], temporal_window: int = 0, projection_threshold: int = 0, 
                        team: bool = False, use_sim_verify: bool = False, wp_cost: float = 0.5, ws_cost: float = 0.5, 
                        wt_cost: float = 0.5, age_max: int = 50, amg: bool = False, similarity_threshold: float = 0.5, 
                        window_size: int = 2, iou_threshold: float = 0.4) -> None:
        """
        The reassignment process now includes an IoU threshold check for bounding boxes during track reassignment.
        """
        logger.info("Performing Step 1 Reassign Logic with IoU check (iou_threshold=0.4).")
        self.projection_threshold = projection_threshold
        self.max_projection_distance = projection_threshold
        self.similarity_threshold = similarity_threshold

        for entity_type in entity_types:
            stable_ids, processed_ids = set(), set()
            tracks = sorted(
                self.summary_data[entity_type].keys(),
                key=lambda tid: self.summary_data[entity_type][tid].get('first_appearance', 0),
            )
            for track_id in tracks:
                meta = self.summary_data[entity_type][track_id]
                first_appearance = meta['first_appearance']

                if team:
                    new_team = self.classify_team_track_id(entity_type, track_id)
                    self.update_club_info(track_id, new_team, entity_type)
                # Make sure projections are up to date
                self.refresh_projection_data(entity_type, track_id, meta, window_size)
                # Gather possible lost IDs to merge with
                time_zone = self.summary_data[entity_type][track_id].get('time_zone')
                lost_ids = self.get_lost_stable_ids(entity_type, stable_ids, first_appearance, age_max, temporal_window, time_zone)
                best_match, min_score, best_proj_dist = None, float('inf'), None
                iou_best = 0
                ids_candidate = []
                for lost_id in lost_ids:
                    if not self.summary_data[entity_type].get(lost_id):
                        logger.debug(f"No metadata found for lost track ID '{lost_id}' in '{entity_type}'. Skipping.")
                        continue
                    proj_dist = self.get_track_projection_distance(entity_type, track_id, lost_id)
                    if proj_dist > self.projection_threshold:
                        continue
                    similarity = 0
                    if use_sim_verify:
                        similarity = self.get_track_similarity(entity_type, track_id, lost_id)
                    # IoU check: Get first appearance frame bbox for new track and last appearance for lost track
                    first_frame_new = self.tracking_data[first_appearance].get(entity_type, {}).get(str(track_id), {}).get('bbox')
                    last_frame_lost = self.summary_data[entity_type][lost_id].get('last_frame_lost')
                    if last_frame_lost is None:
                        last_frame_lost = len(self.tracking_data) - 1

                    if last_frame_lost is not None:
                        last_frame_bbox = self.tracking_data[last_frame_lost].get(entity_type, {}).get(str(lost_id), {}).get('bbox')

                    if not self.verify_merge_criteria(similarity, proj_dist, use_sim_verify):
                        continue
                    if first_frame_new and last_frame_bbox:
                        iou = calculate_iou(first_frame_new, last_frame_bbox)
                        if iou < iou_threshold:
                            continue  # Skip if IoU is below the threshold
                    ids_candidate.append(lost_id)
                    score = self.calculate_combined_score(entity_type=entity_type, track_id=track_id, 
                                                        candidate_id=lost_id, projection_dist=proj_dist, wp=wp_cost, 
                                                        similarity=similarity, ws=ws_cost, team=team, wt=wt_cost)
                    if score < min_score:
                        min_score = score
                        best_match = lost_id
                        best_proj_dist = proj_dist
                        iou_best = iou
                if amg and len(ids_candidate) > 1:
                    stable_ids.add(track_id)
                    processed_ids.add(track_id)
                    logger.info(f"AMG: No suitable lost track for track ID '{track_id}' ({entity_type}).")
                    continue
                if best_match and iou_best > iou_threshold :
                    self.reassign_track(track_id, best_match, entity_type)
                    # logger.info(f"Merged lost track ID '{best_match}' -> active track ID '{track_id}' ({entity_type}), Combined Score={min_score:.4f}, Dist={best_proj_dist}")
                else:
                    stable_ids.add(track_id)
                    processed_ids.add(track_id)
                    # logger.info(f"No suitable lost track for track ID '{track_id}' ({entity_type}). Marked stable.")

            logger.info(f"Final stable '{entity_type}' tracks count: {len(stable_ids)}")
    ##############################################################################
    # -------------------- Remove Short Tracks & Main Process --------------------
    ##############################################################################
    def _remove_short_tracks(self, min_frames: int = 10) -> None:
        """
        Remove tracks that appear fewer than `min_frames` frames from both data & summary.
        Now calculates duration based on actual present frames rather than first/last appearance.
        """
        logger.info(f"Removing short tracks (duration < {min_frames} frames)")
        tracks_to_remove = defaultdict(list)
        total_frames = len(self.tracking_data)
        for entity_type, tracks in self.summary_data.items():
            if entity_type != "player":
                continue
            for track_id, meta in tracks.items():
                try:
                    # Calculate duration based on actual present frames
                    duration = sum(1 for frame in self.tracking_data 
                                if entity_type in frame 
                                and track_id in frame[entity_type])
                    if duration < min_frames:
                        tracks_to_remove[entity_type].append(track_id)
                        logger.debug(f"Marking short track: {entity_type}-{track_id} ({duration} frames)")
                except KeyError as e:
                    logger.error(f"Missing metadata key {e} for {entity_type}-{track_id}")

        # Rest of the removal logic remains the same
        for entity_type, track_ids in tracks_to_remove.items():
            for track_id in track_ids:
                if track_id in self.summary_data[entity_type]:
                    del self.summary_data[entity_type][track_id]
                    logger.info(f"Removed {entity_type}-{track_id} from summary data")

            removal_count = 0
            for frame_data in self.tracking_data:
                entities = frame_data.get(entity_type, {})
                for track_id in track_ids:
                    if track_id in entities:
                        del entities[track_id]
                        removal_count += 1
            logger.info(f"Removed {removal_count} entries for {len(track_ids)} {entity_type} tracks from tracking data")

        for entity_type in list(self.summary_data.keys()):
            if not self.summary_data[entity_type]:
                del self.summary_data[entity_type]
                logger.debug(f"Removed empty entity type: {entity_type}")

        logger.info(f"Short track removal completed. Removed {sum(len(v) for v in tracks_to_remove.values())} tracks")
    def classify_team_track_id(self, entity_type: str, track_id: str) -> str:
        """
        Classifies the team affiliation for the given track_id using frames in 'classification_frames'.
        """
        logger.info(f"Starting classification for track ID {track_id} in '{entity_type}'.")
        entity_summary = self.summary_data.get(entity_type, {})
        track_summary = entity_summary.get(str(track_id), {})
        if not track_summary:
            logger.info(f"No summary data for track ID {track_id} in '{entity_type}'.")
            return "Unknown", 0.0

        class_frames = track_summary.get('classification_frames', [])
        if not class_frames:
            logger.info(f"No valid frames for classification in track ID {track_id} / {entity_type}.")
            return "Unknown", 0.0

        crops = self.extract_entity_crops(entity_type, track_id, class_frames)
        if len(crops) % 2 == 0:
            crops.pop()

        logger.info(f"Extracted {len(crops)} crops for track ID {track_id}. Predicting team...")
        predicted_teams = self.team_classifier.predict(crops)
        if not predicted_teams:
            # logger.info(f"No predictions returned for {track_id}. Assigning 'Unknown'.")
            return "Unknown", 0.0
        from collections import defaultdict
        team_counts = defaultdict(int)
        for t in predicted_teams:
            team_counts[t] += 1
        total_predictions = len(predicted_teams)
        max_count = max(team_counts.values())
        confidence = max_count / total_predictions if total_predictions > 0 else 0.0
        assigned_team = max(team_counts.items(), key=lambda x: x[1])[0]
        # logger.info(f"Assigned team '{assigned_team}' to track ID {track_id} with confidence {confidence:.2f}.")
        return assigned_team, confidence
    def identify_mismatched_tracks(self):
        """Identify tracks that appear in multiple categories."""
        track_category_counts = {}

        for frame in self.tracking_data:
            for category in ['player', 'goalkeeper', 'referee']:
                if category in frame:
                    for track_id_str in frame[category].keys():
                        trackid = track_id_str
                        if trackid not in track_category_counts:
                            track_category_counts[trackid] = {'player': 0, 'goalkeeper': 0, 'referee': 0}
                        track_category_counts[trackid][category] += 1

        mismatched_tracks = {}
        for track_id, counts in track_category_counts.items():
            categories_with_detections = [cat for cat, count in counts.items() if count > 0]
            if len(categories_with_detections) > 1:
                correct_category = max(counts, key=lambda k: (counts[k], k))
                mismatched_tracks[track_id] = {'correct_category': correct_category, 'counts': counts}
        return mismatched_tracks
    def update_tracking_data(self, mismatched_tracks):
        """Update tracking data to correct mismatched tracks."""
        for track_id_str, info in mismatched_tracks.items():
            correct_category = info['correct_category']
            for frame in self.tracking_data:
                for category in ['player', 'goalkeeper', 'referee']:
                    if category != correct_category and track_id_str in frame.get(category, {}):
                        attributes = frame[category][track_id_str]
                        if correct_category not in frame:
                            frame[correct_category] = {}
                        
                        # Ensure required attributes for 'player' category
                        if correct_category == 'player' or  correct_category == 'goalkeeper':
                            if 'club' not in attributes:
                                attributes['club'] = None
                            if 'club_color' not in attributes:
                                attributes['club_color'] = [None, None, None]

                        frame[correct_category][track_id_str] = attributes
                        del frame[category][track_id_str]

 
    def classify_team_track_id_batch(self, entity_type: str, track_ids: List[str]) -> Dict[str, str]:
        """Batch classify team for multiple track IDs with optimized frame loading."""
        # Collect all required (frame_num, track_id, bbox)

        frame_track_bboxes = defaultdict(list)  # {frame_num: [(track_id, bbox)]}
        for track_id in track_ids:
            class_frames = self.summary_data[entity_type][track_id].get('classification_frames', [])
            for frame_num in class_frames:
                if 0 <= frame_num < len(self.tracking_data):
                    frame_data = self.tracking_data[frame_num]
                    bbox = frame_data.get(entity_type, {}).get(track_id, {}).get('bbox')
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        frame_track_bboxes[frame_num].append((track_id, bbox))

        # Process frames in batches to manage memory
        BATCH_SIZE = self.batch_size
        frame_nums = sorted(frame_track_bboxes.keys())


        results = defaultdict(list)
        
        prefetch_future = None
        for i in range(0, len(frame_nums), BATCH_SIZE):
            batch_frames = frame_nums[i:i + BATCH_SIZE]
            if prefetch_future is None:
                # Initial preload
                self._preload_frames(batch_frames, batch_size=BATCH_SIZE)
            else:
                # Wait for async preload of current batch
                prefetch_future.result()
            # Start preloading next batch in background
            next_frames = frame_nums[i + BATCH_SIZE:i + 2 * BATCH_SIZE]
            prefetch_future = self._preload_frames_async(next_frames, batch_size=BATCH_SIZE)

            batch_crops = []
            batch_track_ids = []
            
            for frame_num in batch_frames:
                try:
                    # Get frame and resize to reduce memory usage
                    frame_img = self.get_cached_frame(frame_num)
                    # frame_img = self.frame_cache.get(frame_num)

                    if frame_img is None:
                        continue
                        

                    h, w, _ = frame_img.shape
                    for track_id, bbox in frame_track_bboxes[frame_num]:
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(w, x_max)
                        y_max = min(h, y_max)
                        
                        if x_min < x_max and y_min < y_max:
                            crop = frame_img[y_min:y_max, x_min:x_max]
                            batch_crops.append(crop)
                            batch_track_ids.append(track_id)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_num}: {str(e)}")
                    continue

            # Process batch if we have crops
            if batch_crops:
                try:
                    predicted_teams = self.team_classifier.predict(batch_crops)
                    for tid, team in zip(batch_track_ids, predicted_teams):
                        results[tid].append(team)
                except Exception as e:
                    logger.error(f"Error in batch prediction: {str(e)}")
                    continue

            # Clear memory after each batch
            del batch_crops
            del batch_track_ids
            gc.collect()  # Force garbage collection

        if prefetch_future is not None:
            prefetch_future.result()

        # Aggregate final results
        final_results = {}
        for tid in track_ids:
            teams = results.get(tid, [])
            if teams:
                final_results[tid] = max(set(teams), key=teams.count)
            else:
                final_results[tid] = "Unknown"

        return final_results
    def classify_players(self, entity_types: List[str] = ["player"]) -> None:
        """Optimized batch classification."""
        for entity_type in entity_types:
            if entity_type not in self.summary_data:
                continue                
            track_ids = list(self.summary_data[entity_type].keys())
            if not track_ids:
                continue
            # Batch classify all tracks
            results = self.classify_team_track_id_batch(entity_type, track_ids)
            # Update club info in bulk
            for track_id, team in results.items():
                confidence = 1.0  # Adjust if confidence is tracked
                self.update_club_info(track_id, team, confidence, entity_type)
                # logger.info(f"for this track id {track_id} we got this team classficition {team}")

    def enhanced_hierarchical_matching(
        self,
        entity_types: List[str] = ["player"],
        temporal_window: int = 0,
        projection_threshold: float = 100.0,
        wp: float = 0.2,  # Projection weight 
        ws: float = 0.8,  # ReID similarity weight
        similarity_threshold: float = 0.70,
        age_max: int = 50,
        window_size: int = 2,
        use_sim_verify: bool = False,
        iou_threshold: float = 0.4
    ) -> None:
        """
        Revised hierarchical matching:
        
        - Process each track one by one (ordered by first appearance).
        - Assign each track to its corresponding team set (only 'Club1' and 'Club2' are valid).
        - For each new track, check if there are any lost stable tracks in the same team.
        - If candidates exist, evaluate projection distance and ReID similarity to compute a combined cost.
        - If the best candidate's cost is below a threshold, merge the tracks using a unidirectional merge 
        (merging the new track into the lost track) to prevent excessive ID generation.
        """
        logger.info("Starting revised enhanced hierarchical matching...")
        logger.info(f"Weights: wp={wp}, ws={ws}")
        self.projection_threshold = projection_threshold
        self.max_projection_distance = projection_threshold
        self.similarity_threshold = similarity_threshold

        # We only process the 'player' entity type
        entity_type = "player"

        # Initialize stable track sets for each team
        stable_tracks_by_team = {"Club1": set(), "Club2": set()}

        # Process tracks in order of first appearance
        all_tracks = sorted(
            self.summary_data[entity_type].keys(),
            key=lambda tid: self.summary_data[entity_type][tid].get("first_appearance", 0)
        )

        for track_id in all_tracks:
            meta = self.summary_data[entity_type][track_id]
            team = meta.get("club")



            if team not in ["Club1", "Club2"]:
                logger.warning(f"Track {track_id} has invalid team '{team}'. Skipping merge.")
                continue
            first_appearance = meta["first_appearance"]
            first_frame_new = self.tracking_data[first_appearance].get(entity_type, {}).get(str(track_id), {}).get('bbox')
            # Refresh projection data for the current track
            self.refresh_projection_data(entity_type, track_id, meta, window_size)
            # Retrieve lost (stable) IDs already in this team that are eligible for merge
            time_zone = meta.get('time_zone')
            lost_ids = self.get_lost_stable_ids(entity_type, stable_tracks_by_team[team], first_appearance, age_max, temporal_window, time_zone)
            # logger.info(f"for this track {track_id} we got {lost_ids} lost")
            if not lost_ids:
                # No candidates available; add this track as stable
                stable_tracks_by_team[team].add(track_id)
                continue
            # Evaluate each candidate using projection distance and similarity
            candidates = []
            for lost_id in lost_ids:
                proj_dist = self.get_track_projection_distance(entity_type, track_id, lost_id)
                similarity = self.get_track_similarity(entity_type, track_id, lost_id)
                # Skip candidate if basic criteria are not met
                last_frame_lost = self.summary_data[entity_type][lost_id].get('last_frame_lost')
                if last_frame_lost is None:
                    last_frame_lost = len(self.tracking_data) - 1
                if last_frame_lost is not None:
                    last_frame_bbox = self.tracking_data[last_frame_lost].get(entity_type, {}).get(str(lost_id), {}).get('bbox')
                if first_frame_new and last_frame_bbox:
                        iou = calculate_iou(first_frame_new, last_frame_bbox)
                        cost_iou = 1 - iou
                if not self.verify_merge_criteria(similarity, proj_dist, use_sim_verify) or iou < 0.001:
                    continue
                norm_proj = min(proj_dist / projection_threshold, 1.0)
                norm_sim = 1.0 - similarity
                total_cost = (wp * norm_proj) + (ws * norm_sim) + (cost_iou * 0.5)
                candidates.append((lost_id, total_cost, proj_dist, similarity , iou))
            if candidates:
                candidates.sort(key=lambda x: x[1])  # sort by combined cost
                best_match, best_cost, best_proj, best_sim , iou_best = candidates[0]
                self.reassign_track(track_id, best_match, entity_type)
                # logger.info(f"Merged track {track_id} -> {best_match} (cost={best_cost:.3f}, proj={best_proj:.3f}, sim={best_sim:.3f})  ,  iou={iou_best:.3f}) ")
            else:
                stable_tracks_by_team[team].add(track_id)

        for team, stable_ids in stable_tracks_by_team.items():
            logger.info(f"Final stable tracks for {team}: {len(stable_ids)}")



    def assign_goalkeepers_to_team_by_proximity(self) -> None:
        """
        Assign each goalkeeper's team based on proximity to the two team centroids (Club1, Club2).
        All logic is contained here in one function.
        """
        logger.info("Assigning goalkeepers to teams based on proximity to team centroids...")
        # ----------------------------
        # 1) Compute team centroids for "player" tracks
        # ----------------------------
        club1_positions = []
        club2_positions = []
        # We'll use the 'avg_projection_after_appearance' or 'avg_projection_before_lost'
        # from summary_data to represent the on-field (x, y).
        for track_id, meta in self.summary_data.get("player", {}).items():
            club = meta.get("club")
            if club not in ("Club1", "Club2"):
                continue  # Skip unknown or unassigned clubs
            # Try 'avg_projection_after_appearance'
            avg_after = meta.get("avg_projection_after_appearance")
            avg_before = meta.get("avg_projection_before_lost")
            # Gather whichever valid positions we have
            points = []
            if avg_after and len(avg_after) == 2:
                points.append(avg_after)
            if avg_before and len(avg_before) == 2:
                points.append(avg_before)

            if not points:
                continue
            # Average them for a single representative point
            x_mean = sum(p[0] for p in points) / len(points)
            y_mean = sum(p[1] for p in points) / len(points)
            # Add to the right list
            if club == "Club1":
                club1_positions.append((x_mean, y_mean))
            else:  # club == "Club2"
                club2_positions.append((x_mean, y_mean))

        # Now compute each team's centroid
        def _mean_xy(coords):
            if not coords:
                return None
            x_sum = sum(c[0] for c in coords)
            y_sum = sum(c[1] for c in coords)
            return [x_sum / len(coords), y_sum / len(coords)]
        club1_centroid = _mean_xy(club1_positions)
        club2_centroid = _mean_xy(club2_positions)
        if not club1_centroid or not club2_centroid:
            logger.warning("Could not compute both team centroids. Skipping GK assignment by proximity.")
            return
        # logger.info(f"Club1 centroid = {club1_centroid}, Club2 centroid = {club2_centroid}")
        # ----------------------------
        # 2) For each goalkeeper, compute average position + assign the closest team
        # ----------------------------
        goalkeepers = self.summary_data.get("goalkeeper", {})
        for gk_id, gk_meta in goalkeepers.items():

            gk_points = []
            after_app = gk_meta.get("avg_projection_after_appearance")
            before_lost = gk_meta.get("avg_projection_before_lost")

            if after_app and len(after_app) == 2:
                gk_points.append(after_app)
            if before_lost and len(before_lost) == 2:
                gk_points.append(before_lost)

            if not gk_points:
                # If there's no valid on-field projection, skip or default to Unknown
                logger.debug(f"Goalkeeper {gk_id} has no valid projection. Skipping team assignment.")
                continue
            # Compute a single average (x, y) for this goalkeeper
            x_mean = sum(p[0] for p in gk_points) / len(gk_points)
            y_mean = sum(p[1] for p in gk_points) / len(gk_points)
            # Compute distances
            dist_to_club1 = math.hypot(x_mean - club1_centroid[0], y_mean - club1_centroid[1])
            dist_to_club2 = math.hypot(x_mean - club2_centroid[0], y_mean - club2_centroid[1])
            # Decide which team is closer
            if dist_to_club1 < dist_to_club2:
                assigned_team = "Club1"
            else:
                assigned_team = "Club2"
            # logger.info(
            #     f"Goalkeeper {gk_id}: dist(Club1)={dist_to_club1:.2f}, "
            #     f"dist(Club2)={dist_to_club2:.2f} -> assigned to {assigned_team}"
            # )
            # ----------------------------
            # 3) Update the summary & tracking data for this goalkeeper
            # ----------------------------
            self.update_club_info(gk_id, assigned_team, 1.0, "goalkeeper")

    def reassign_goalkeeper_referee_based_on_projection(
        self,
        entity_types: List[str] = ['goalkeeper', 'referee'],
        projection_threshold: float = 150.0
    ) -> None:

        self.projection_threshold = 100
        self.max_projection_distance = 100
        
        """
        Reassign goalkeepers and referees based on projection distance.
        The function matches tracks that are within a threshold projection distance
        and merges them.

        Args:
            entity_types: The types of entities to process (goalkeeper, referee).
            projection_threshold: The maximum allowable projection distance to merge tracks.
        """
        logger.info("Starting reassignment for goalkeepers and referees based on projection...")
        
        for entity_type in entity_types:
            stable_ids, processed_ids = set(), set()
            # Sort tracks by appearance time (first appearance)
            tracks = sorted(
                self.summary_data[entity_type].keys(),
                key=lambda tid: (
                    self.summary_data[entity_type][tid].get('first_appearance', 0),
  
                )
            )
            # logger.info(f"Found {len(tracks)} total {entity_type} tracks to process")

            for track_id in tracks:
                meta = self.summary_data[entity_type][track_id]
                first_appearance = meta['first_appearance']

                self.refresh_projection_data(entity_type, track_id, meta , 2)
                
                # Gather lost IDs to merge with
                time_zone = meta.get('time_zone')
                lost_ids = self.get_lost_stable_ids(entity_type , stable_ids, first_appearance,temporal_window= 0 ,  age_max=10000, time_zone=time_zone)
                
                if not lost_ids:
                    stable_ids.add(track_id)
                    processed_ids.add(track_id)
                    # logger.info(f"No candidates found. Adding {track_id} to stable tracks.")
                    continue

                best_match = None
                min_projection_dist = float('inf')

                for lost_id in lost_ids:
                    proj_dist = self.get_track_projection_distance(entity_type, track_id, lost_id)

                    if proj_dist < projection_threshold and proj_dist < min_projection_dist:
                        best_match = lost_id
                        min_projection_dist = proj_dist

                if best_match:
                    self.reassign_track(track_id, best_match, entity_type)
                    # logger.info(f"Merged {best_match} -> {track_id} based on projection (dist={min_projection_dist:.4f})")
                else:
                    stable_ids.add(track_id)
                    # logger.info(f"No suitable lost track for {track_id} based on projection. Marked stable.")

                processed_ids.add(track_id)

            # logger.info(f"Final stable {entity_type} tracks: {len(stable_ids)}")
    def renumber_track_ids(self, entity_order: List[str] = None) -> None:
        """
        Renumbers track IDs sequentially within each entity type based on their first appearance.
        Updates both tracking data and summary data.
        
        Args:
            entity_order: Order of entity types to process. Default: ['ball', 'referee', 'goalkeeper', 'player']
        """
        if entity_order is None:
            entity_order = ['referee' , 'player' , 'goalkeeper' ,'ball']
        
        # Create mapping from old IDs to new sequential IDs for each entity type
        new_id = 1
        id_mapping = defaultdict(dict)
        
        # Process summary data to create mappings
        for entity_type in entity_order:
            if entity_type not in self.summary_data:
                continue
                
            # Sort tracks by first appearance time
            sorted_ids = sorted(
                self.summary_data[entity_type].keys(),
                key=lambda tid: self.summary_data[entity_type][tid]['first_appearance']
            )
            
            # Assign new sequential IDs starting from 1
            
            for old_id in sorted_ids:
                id_mapping[entity_type][old_id] = str(new_id)
                new_id += 1
        
        # Update tracking data with new IDs
        for frame in self.tracking_data:
            for entity_type in entity_order:
                if entity_type not in frame:
                    continue
                    
                entities = frame[entity_type]
                updated_entities = {}
                
                for old_id, data in entities.items():
                    if old_id in id_mapping[entity_type]:
                        new_id = id_mapping[entity_type][old_id]
                        updated_entities[new_id] = data
                        
                frame[entity_type] = updated_entities
        


        # logger.info("Successfully renumbered track IDs for entity types: %s", entity_order)   
    def _compute_team_centroids(self) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Computes the average projection (centroid) for each team (Club1, Club2) based on player positions.
        """
        club_projections = defaultdict(list)
        for frame in self.tracking_data:
            players = frame.get('player', {})
            for pid, pdata in players.items():
                club = pdata.get('club')
                projection = pdata.get('projection')
                if club in ['Club1', 'Club2'] and projection and len(projection) == 2:
                    club_projections[club].append(projection)
        
        team_centroids = {}
        for club, projs in club_projections.items():
            if not projs:
                team_centroids[club] = None
                continue
            avg_x = sum(p[0] for p in projs) / len(projs)
            avg_y = sum(p[1] for p in projs) / len(projs)
            team_centroids[club] = (avg_x, avg_y)
        return team_centroids
    def _get_goalkeeper_projection(self, gk_track_id: str) -> Optional[Tuple[float, float]]:
        """
        Computes the average projection for a goalkeeper track.
        """
        projections = []
        for frame in self.tracking_data:
            gk_data = frame.get('goalkeeper', {}).get(gk_track_id, {})
            projection = gk_data.get('projection')
            if projection and len(projection) == 2:
                projections.append(projection)
        if not projections:
            return None
        avg_x = sum(p[0] for p in projections) / len(projections)
        avg_y = sum(p[1] for p in projections) / len(projections)
        return (avg_x, avg_y)
    def classify_goalkeepers(self) -> None:
        """
        Assigns each goalkeeper to the team (Club1/Club2) whose centroid is closest to their average position.
        """
        logger.info("Classifying goalkeepers based on team proximity...")
        team_centroids = self._compute_team_centroids()
        
        # Check if centroids are available
        if not team_centroids.get('Club1') or not team_centroids.get('Club2'):
            logger.warning("Cannot classify goalkeepers: Insufficient data for team centroids.")
            return

        for gk_id in self.summary_data.get('goalkeeper', {}).keys():
            gk_proj = self._get_goalkeeper_projection(gk_id)
            if not gk_proj:
                logger.info(f"Goalkeeper {gk_id} has no valid projections. Assigning 'Unknown'.")
                self.update_club_info(gk_id, "Unknown", 0.0, 'goalkeeper')
                continue

            # Calculate distances to each team's centroid
            dist_club1 = math.dist(gk_proj, team_centroids['Club1'])
            dist_club2 = math.dist(gk_proj, team_centroids['Club2'])
            
            # Determine confidence based on relative distances
            total_dist = dist_club1 + dist_club2
            confidence = 1.0 - (min(dist_club1, dist_club2) / total_dist) if total_dist > 0 else 1.0
            
            # Assign team
            if dist_club1 < dist_club2:
                assigned_team = 'Club1'
            else:
                assigned_team = 'Club2'
            
            self.update_club_info(gk_id, assigned_team, confidence, 'goalkeeper')
            # logger.info(f"Assigned goalkeeper {gk_id} to {assigned_team} (confidence: {confidence:.2f})")

    def process(self):
        """
        Orchestrates the entire post-processing pipeline.

        Steps:
        1. Reassign track IDs by merging newly appeared tracks into lost existing tracks
           based on team classification and ReID similarity.
        2. Assign club information for new stable IDs.
        3. Save the updated tracking and summary data if desired.

        Add or remove steps as needed for your pipeline.
        """
        logger.info("Starting post-processing pipeline...")

        splitter = TrackletSplitter(
            reid_model=self.reid_module,
            eps=0.4,
            min_samples=3,
            min_cluster_size=3
        )
    
    # Apply the splitter to each entity type
    
        mismatched_tracks = self.identify_mismatched_tracks()
        if mismatched_tracks:
            logger.info(f"Found {len(mismatched_tracks)} mismatched tracks. Updating tracking data...")
            self.update_tracking_data(mismatched_tracks)
        # Initial metadata generation
        self.summary_data = self._create_metadata()
        self._create_crop_valid_attributes()
        self.reassign_goalkeeper_referee_based_on_projection()
        # Initial metadata generation
        self.summary_data = self._create_metadata()
        self._create_crop_valid_attributes(
                                            isolated_ther = 0,
                                            isolated_ther_classifciation = 0,
                                            max_select= 2,
                                            max_select_classification= 5,
                                            window_size = 10,
                                            )
        logger.info(f"first try reassigning tracks")
        self.reassign_tracks(       entity_types = ['player'],
                                    temporal_window= 0,
                                    projection_threshold = 10 ,
                                    team= False,
                                    use_sim_verify= True,
                                    similarity_threshold = 0.90,
                                    wp_cost= 0.6,
                                    ws_cost= 0.4,
                                    wt_cost= 0.5,
                                    age_max = 5,
                                    window_size = 1,
                                    amg = True,
                                    iou_threshold=0.4)

        logger.info(f"second try reassigning tracks")


        self.reassign_tracks(       entity_types = ['player'],
                                    temporal_window= 0,
                                    projection_threshold = 15 ,
                                    team= False,
                                    use_sim_verify= True,
                                    similarity_threshold = 0.85,
                                    wp_cost= 0.6,
                                    ws_cost= 0.4,
                                    wt_cost= 0.5,
                                    age_max = 5,
                                    window_size = 2,
                                    amg = True,
                                    iou_threshold=0.25)

        self._remove_short_tracks(3)
        self.summary_data = self._create_metadata()
        
        
        
        
        for entity_type in ['player']:
            self.tracking_data, self.summary_data, split_info = splitter.apply_splits(
                entity_type, 
                self.tracking_data, 
                self.video_loader, 
                self.summary_data
            )
            
        logger.info(f"Split info for {entity_type}: {split_info}")
        self._create_crop_valid_attributes(
                                            isolated_ther = 0,
                                            isolated_ther_classifciation = 0,
                                            max_select= 2,
                                            max_select_classification= 9,
                                            window_size = 10,
                                            )
        logger.info(f"classify players")


        self.classify_players()
        logger.info(f"enhanced hierarchical matching")
        self.enhanced_hierarchical_matching( entity_types = ['player'])


        self.classify_goalkeepers()                               

        self.summary_data = self._create_metadata()
        self._create_crop_valid_attributes()
        self.renumber_track_ids()
        self.summary_data = self._create_metadata()
        self._create_crop_valid_attributes()
        save_json(self.tracking_data, self.updated_tracking_path)
        save_json(self.summary_data, self.updated_summary_path)

        logger.info("Post-processing pipeline completed successfully.")
        self.preload_executor.shutdown(wait=True)
        











def calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: List of coordinates [x_min, y_min, x_max, y_max] for the first bbox.
        bbox2: List of coordinates [x_min, y_min, x_max, y_max] for the second bbox.
    
    Returns:
        float: The IoU value between 0 and 1.
    """
    x_min1, y_min1, x_max1, y_max1 = bbox1
    x_min2, y_min2, x_max2, y_max2 = bbox2

    # Calculate the area of intersection
    x_intersection_min = max(x_min1, x_min2)
    y_intersection_min = max(y_min1, y_min2)
    x_intersection_max = min(x_max1, x_max2)
    y_intersection_max = min(y_max1, y_max2)

    intersection_area = max(0, x_intersection_max - x_intersection_min) * max(0, y_intersection_max - y_intersection_min)

    # Calculate the area of both bounding boxes
    bbox1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    bbox2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    # Calculate the area of union
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

















