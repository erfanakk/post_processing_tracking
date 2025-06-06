# # video_processor.py

# import cv2
# import logging
# from typing import Optional
# import numpy as np

# logger = logging.getLogger(__name__)

# class VideoFrameLoader:
#     def __init__(self, video_path: str):
#         """
#         Initialize the VideoFrameLoader with the given video file path.

#         Args:
#             video_path (str): Path to the video file.
#         """
#         self.cap = cv2.VideoCapture(video_path)
#         if not self.cap.isOpened():
#             logger.error(f"Failed to open video: {video_path}")
#             raise IOError(f"Cannot open video file: {video_path}")
#         self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
#         self.current_frame = 0
#         logger.info(f"Video {video_path} opened with {self.total_frames} frames at {self.frame_rate} FPS.")

#     def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
#         """
#         Retrieve a specific frame from the video.

#         Args:
#             frame_idx (int): Index of the frame to retrieve.

#         Returns:
#             np.ndarray or None: The requested frame as an image, or None if frame is invalid.
#         """
#         if frame_idx < 0 or frame_idx >= self.total_frames:
#             logger.warning(f"Requested frame {frame_idx} is out of bounds.")
#             return None
        
#         # Seek to the desired frame if it's not already the current one
#         if frame_idx != self.current_frame:
#             self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
#             self.current_frame = frame_idx

#         # Read the frame
#         ret, frame = self.cap.read()
#         if not ret:
#             logger.warning(f"Failed to read frame {frame_idx}.")
#             return None

#         self.current_frame += 1
#         return frame

#     def get_frame_by_time(self, timestamp: float) -> Optional[np.ndarray]:
#         """
#         Retrieve a frame by timestamp in seconds.

#         Args:
#             timestamp (float): Timestamp in seconds.

#         Returns:
#             np.ndarray or None: The frame at the given timestamp, or None if frame is invalid.
#         """
#         frame_idx = int(timestamp * self.frame_rate)
#         return self.get_frame(frame_idx)

#     def release(self):
#         """
#         Release the video capture object and close the video file.
#         """
#         self.cap.release()
#         logger.info("Video capture released.")

#     def get_video_properties(self) -> dict:
#         """
#         Get the basic properties of the video file (frame count, frame rate, etc.).

#         Returns:
#             dict: A dictionary containing the video properties.
#         """
#         return {
#             'total_frames': self.total_frames,
#             'frame_rate': self.frame_rate,
#             'frame_width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
#             'frame_height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
#             "fps": self.frame_rate
#         }




import time
import random
import logging
from typing import Optional, Sequence, Union

import numpy as np
from decord import VideoReader, cpu, gpu

logger = logging.getLogger(__name__)


class VideoFrameLoader:
    """
    Random-access video loader backed by Decord.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    ctx : decord.ndarray.ctx, optional
        Decoding context.  Use ``cpu(0)`` (default) or ``gpu(0)`` if
        your hardware + driver stack supports NVDEC / VA-API / QSV.
    """

    def __init__(self, video_path: str, ctx: Union[cpu, gpu] = cpu(0)):
        self.vr = VideoReader(video_path, ctx=ctx)
        self.total_frames = len(self.vr)
        self.frame_rate = float(self.vr.get_avg_fps())
        self.ctx = ctx

        # size info from the first frame
        h, w, _ = self.vr[0].shape
        self.frame_width, self.frame_height = w, h
        logger.info(
            f"Opened {video_path}: {self.total_frames} frames "
            f"@ {self.frame_rate:.3f} FPS • {w}×{h} • ctx={ctx}"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Return a single frame as ``np.ndarray`` (BGR)."""
        if 0 <= frame_idx < self.total_frames:
            try:
                return self.vr[frame_idx].asnumpy()
            except Exception as e:
                logger.warning(f"Decoding error at frame {frame_idx}: {e}")
                return None
        logger.warning(f"Requested frame {frame_idx} out of bounds")
        return None

    def get_batch(self, indices: Sequence[int]) -> np.ndarray:
        """
        Fetch *many* frames in one shot.  Decord accepts any order:

        >>> frames = loader.get_batch([10, 500, 3])
        >>> frames.shape   # (3, H, W, 3)
        """
        return self.vr.get_batch(list(indices)).asnumpy()

    def get_frame_by_time(self, timestamp: float) -> Optional[np.ndarray]:
        """Nearest frame at ``timestamp`` seconds."""
        idx = int(timestamp * self.frame_rate)
        return self.get_frame(idx)

    def get_video_properties(self) -> dict:             # unchanged
        return {
            "total_frames": self.total_frames,
            "frame_rate": self.frame_rate,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.frame_rate,
        }

    def release(self) -> None:
        """Drop the underlying VideoReader."""
        del self.vr
        logger.info("VideoReader released.")

    def get_batch_safe(self, indices, batch_size=64):
        """
        Yield lists of (idx, frame ndarray) in chunks so you never load
        thousands of Full-HD frames at once.

        Parameters
        ----------
        indices : Sequence[int]
            Any iterable of frame indices (can be unsorted / repeated).
        batch_size : int, optional
            Max number of frames to hold in memory at once.
        """
        # Decord wants ascending order
        sorted_unique = sorted(set(i for i in indices if 0 <= i < self.total_frames))
        for i in range(0, len(sorted_unique), batch_size):
            chunk = sorted_unique[i:i + batch_size]
            imgs  = self.vr.get_batch(chunk).asnumpy()          # one call
            for idx, img in zip(chunk, imgs):
                yield idx, img
