# config.py
import torch
VIDEO_PATH = r'input_videos\clip_7.mp4'
TRACKING_DATA_PATH = r".\output\debug\final\object_tracks.jsonl"
SIMILARITY_THRESHOLD = 0.65
BATCH_SIZE = 8
UPDATED_TRACKING_DATA_PATH = r".\output\debug\post_process\object_tracks.jsonl"
UPDATED_SUMMARY_DATA_PATH = r".\\output\debugt\post_process\metadata.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



