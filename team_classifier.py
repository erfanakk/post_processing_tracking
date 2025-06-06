#team_classifier.py

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.cluster import KMeans
import torch
from typing import List, Tuple, Dict
import logging
import cv2
import umap
from concurrent.futures import ThreadPoolExecutor, as_completed
from .feature_extractor import ReIDFeatureExtractor
logger = logging.getLogger(__name__)

class TeamClassifier:
    def __init__(self, extractor: ReIDFeatureExtractor, device: str = 'cuda', batch_size: int = 4, target_size: Tuple[int, int] = (256, 128)):
        self.extractor = extractor
        self.device = device
        self.batch_size = batch_size
        self.target_size = target_size
        self.reducer = umap.UMAP(n_components=5, random_state=42, n_jobs=-1)
        self.cluster_model = KMeans(n_clusters=2, random_state=42)
        self.club_names: Dict[int, str] = {}
        self.club_colors: Dict[str, Tuple[int, int, int]] = {}
        self.use_mask = False
    
    def apply_mask(self, image: np.ndarray, green_threshold: float = 0.08) -> np.ndarray:
        """ Apply a mask to remove green areas based on HSV color space. """
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
        total_pixels = image.shape[0] * image.shape[1]
        masked_pixels = cv2.countNonZero(cv2.bitwise_not(mask))
        mask_percentage = masked_pixels / total_pixels
        if mask_percentage < green_threshold:
            return image.copy()  # Prevent modifying original
        else:
            return cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))


    def calculate_average_color(self, crop: np.ndarray) -> Tuple[float, float, float]:
        """ Calculate the average color of an image crop. """
        mean_color = crop.astype(np.float32).mean(axis=(0, 1))
        return tuple(mean_color)

    def calculate_average_color_batch(self, crops: List[np.ndarray]) -> List[Tuple[float, float, float]]:
        """Calculate the average color for each crop in a batch."""
        return [self.calculate_average_color(crop) for crop in crops]

    def extract_features(self, crops: List[np.ndarray], green_threshold: float = 0.08) -> np.ndarray:
        """ Extract features from image crops using the feature extractor. """
        features = self.extractor.extract_features(crops)
        return features

    def fit(self, crops: List[np.ndarray], green_threshold: float = 0.08) -> None:
        """
        Train the team classifier on provided image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
            green_threshold (float): Threshold for green mask application.
        """
        logger.info("Starting training of TeamClassifier...")

        # Process crops in batches
        batch_size = 64  # Adjust batch size for performance
        batches = [crops[i:i + batch_size] for i in range(0, len(crops), batch_size)]

        all_avg_colors = []

        # Using ThreadPoolExecutor for parallel processing of batches
        with ThreadPoolExecutor() as executor:
            # Submit all batches to the executor
            futures = [executor.submit(self.calculate_average_color_batch, batch) for batch in batches]

            # Collect the results as they are completed
            for future in as_completed(futures):
                avg_colors_batch = future.result()
                all_avg_colors.extend(avg_colors_batch)

        logger.info(f"Average color calculation completed. Total clusters: {len(all_avg_colors)}")

        # Extract features from crops
        logger.info("Extracting features from crops...")
        data = self.extract_features(crops, green_threshold)

        logger.info("Applying UMAP for dimensionality reduction...")
        # Reduce dimensionality using UMAP
        projections = self.reducer.fit_transform(data)

        logger.info("Fitting KMeans clustering...")
        # Perform clustering
        self.cluster_model.fit(projections)
        labels = self.cluster_model.labels_
        logger.info("Clustering complete. Calculating average colors for each cluster...")
        # Calculate average colors for each cluster
        cluster_colors = defaultdict(list)
        for label, avg_color in zip(labels, all_avg_colors):
            cluster_colors[label].append(avg_color)
        # Assign cluster names and calculate mean colors
        for cluster_id in range(self.cluster_model.n_clusters):
            self.club_names[cluster_id] = f'Club{cluster_id + 1}'
            mean_color = np.mean(cluster_colors[cluster_id], axis=0).astype(int)
            mean_color_bgr = (int(mean_color[2]), int(mean_color[1]), int(mean_color[0]))  # Convert to BGR
            self.club_colors[self.club_names[cluster_id]] = tuple(mean_color_bgr)

        logger.info("Training complete. Clubs assigned with corresponding colors:")
        for club, color in self.club_colors.items():
            logger.info(f"{club}: {color}")
    def predict(self, crops: List[np.ndarray]) -> List[str]:
        """Optimized batch prediction."""
        if not crops:
            return []        
        # Use larger batch size for feature extraction
        features = self.extractor.extract_features(crops, batch_size=64)  # Adjust batch size
        projections = self.reducer.transform(features)
        predictions = self.cluster_model.predict(projections)
        return [self.club_names[p] for p in predictions]
