



# post_processing_pipeline.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
from post_processing.data_loader import load_json, save_json, extract_isolated_object_crops, load_json2
from post_processing.video_processor import VideoFrameLoader
from post_processing.post_processor import PostProcessor
import cv2
from post_processing import config
from post_processing.feature_extractor import ReIDFeatureExtractor
from post_processing.team_classifier import TeamClassifier
from post_processing.utils_post import extract_isolated_object_crops
from sklearn.metrics.pairwise import cosine_similarity

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logfile.txt', mode='w')
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    config.VIDEO_PATH = r"input_videos\clip_7.mp4"


    tracking_data = load_json2(config.TRACKING_DATA_PATH)
    video_loader = VideoFrameLoader(config.VIDEO_PATH)
    video_props = video_loader.get_video_properties()
    
    logger.info("Loading ReID model...")
    reid_extractor = ReIDFeatureExtractor(
        device=config.DEVICE,
    )
    
    logger.info("Initializing team classifier...")
    team_classifier = TeamClassifier(reid_extractor, device=config.DEVICE)


    logger.info("Training team classifier on isolated players...")

    # metadata_video = {
    #     "start_first_half": '00:00',
    #     "end_first_half": '00:33',
    #     "start_second_half": '00:33',
    #     "end_second_half": '01:03',
    #     "start_extra_time": None,
    #     "end_extra_time": None,
    #     "start_penalty_shootout": None,
    #     "end_penalty_shootout": None,
    
    # }
    metadata_video = None


    isolated_crops = extract_isolated_object_crops(
        video_path=config.VIDEO_PATH,
        tracking_data=tracking_data,
        entity_types=["player"],
        max_frames=20,
        frame_step=2,
        start_frame=0,
        distance_threshold=0
        ,metadata_video=metadata_video
    )
    team_classifier.fit(isolated_crops["player"])

    post_processor = PostProcessor(
        tracking_data=tracking_data,
        reid_model=reid_extractor,
        team_classifier=team_classifier,
        video_loader=video_loader,
        similarity_threshold=config.SIMILARITY_THRESHOLD,
    )

    # Execute post-processing
    post_processor.process()

    logger.info("Post-processing pipeline completed successfully.")

if __name__ == "__main__":
    main()

