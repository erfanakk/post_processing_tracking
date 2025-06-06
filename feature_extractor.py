# # # #feature_extractor.py
# # import torch
# # import cv2
# # import numpy as np
# # from typing import List
# # import logging
# # from typing import List, Tuple

# # logger = logging.getLogger(__name__)

# # from transformers import ViTImageProcessor, ViTModel

# # class ReIDFeatureExtractor:
# #     def __init__(self, device: str = 'cuda', target_size: Tuple[int, int] = (224, 224)):
# #         self.device = device
# #         self.model = ViTModel.from_pretrained(r".\post_processing\dino-vitb16").to(device)
# #         self.processor = ViTImageProcessor.from_pretrained(r".\post_processing\dino-vitb16")
# #         self.target_size = target_size
# #         self.model.eval()
# #         logger.info("Team classifier model initialized.")

# #     def extract_features(self, images: List[np.ndarray], batch_size: int = 32) -> np.ndarray:
# #         """
# #         Extract features from a list of images in batches for better memory efficiency.
        
# #         Args:
# #             images: List of numpy arrays representing images
# #             batch_size: Number of images to process in each batch (default: 32)
            
# #         Returns:
# #             numpy array of extracted features
# #         """
# #         with torch.no_grad():
# #             features = []
# #             for i in range(0, len(images), batch_size):
# #                 batch_images = images[i:i+batch_size]
# #                 inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
# #                 outputs = self.model(**inputs)
# #                 batch_features =  outputs.last_hidden_state[:, 0, :].cpu().numpy()   #hiera-small-224-hf
# #                 features.append(batch_features)
            
# #             return np.concatenate(features, axis=0)

# #!/usr/bin/env python3
# """
# Light-weight Re-ID feature extractor built on **Google MobileNet V3‑Small**
# (weights hosted on the Hugging Face Hub; model‑id:
# `google/mobilenet_v3_small_100_224`).

# Public API:
#     • `extract_features(images: List[np.ndarray], batch_size=64)`
#       → returns an L2‑normalised float32 NumPy array of shape (N, 576).

# The block under `if __name__ == "__main__":` performs **one dummy
# forward pass** with random data (`torch.rand`) so you can instantly verify
# that the backbone loads and runs on your machine.
# """

# from __future__ import annotations

# import logging
# from typing import List, Tuple

# import cv2
# import numpy as np
# import torch
# from transformers import AutoImageProcessor, AutoModel

# logger = logging.getLogger("feature_extractor")
# logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)


# class ReIDFeatureExtractor:
#     """Extract 576‑D global embeddings using MobileNet V3‑Small."""

#     def __init__(
#         self,
#         model_id: str = "google/mobilenet_v3_small_100_224",
#         device: str | None = None,
#         target_size: Tuple[int, int] = (224, 224),
#     ) -> None:
#         self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
#         self.target_size = target_size
#         self.model_id = model_id

#         # ── Load backbone + processor ────────────────────────────────────
#         self.processor = AutoImageProcessor.from_pretrained(model_id)
#         self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
#         logger.info("Loaded %s on %s", model_id, self.device)

#     # ────────────────────────────────────────────────────────────────────
#     @torch.inference_mode()
#     def extract_features(
#         self, images: List[np.ndarray], batch_size: int = 64
#     ) -> np.ndarray:
#         """Return an (N, 576) float32 NumPy array of unit‑norm vectors."""

#         feats: list[np.ndarray] = []
#         for i in range(0, len(images), batch_size):
#             batch = images[i : i + batch_size]
#             # OpenCV delivers BGR → convert to RGB first
#             batch_rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in batch]

#             inputs = self.processor(
#                 images=batch_rgb,
#                 size=self.target_size,
#                 return_tensors="pt",
#             ).to(self.device)

#             outputs = self.model(**inputs)
#             featmap = outputs.last_hidden_state  # (B, 576, H, W)
#             vec = torch.nn.functional.normalize(featmap.mean(dim=(-2, -1)), dim=1)
#             feats.append(vec.cpu().numpy())

#         return np.concatenate(feats, axis=0)


# # ────────────────────────────── Quick sanity check ───────────────────────
# if __name__ == "__main__":
#     logger.info("Running a dummy forward pass to verify the model …")

#     extractor = ReIDFeatureExtractor()

#     # Generate a single dummy 224×224 RGB image in CHW format
#     dummy_input = torch.rand(1, 3, 224, 224, device=extractor.device)

#     with torch.inference_mode():
#         featmap = extractor.model(dummy_input).last_hidden_state  # (1, 576, H, W)
#         pooled = torch.nn.functional.normalize(featmap.mean(dim=(-2, -1)), dim=1)

#     print(
#         "Dummy forward OK – feature vector shape:", pooled.shape,
#         "| L2‑norm:", pooled.norm(dim=1).cpu().tolist()
#     )



#!/usr/bin/env python3
"""
Light‑weight Re‑ID feature extractor built on **MobileNet V3‑Small** with
weights from the Hugging Face Hub.  Default backbone:
`timm/mobilenetv3_small_100.lamb_in1k` (2.5 M params, 0.08 GFLOPs @ 224²).

Public API
----------
    extract_features(images: List[np.ndarray], batch_size=64) → np.ndarray
        Returns an L2‑normalised float32 matrix of shape (N, 576).

The block under `if __name__ == "__main__":` performs a *single* dummy
forward pass (random tensor) so you can verify that the model downloads
and runs without errors.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger("feature_extractor")
logging.basicConfig(format="%(levelname)s | %(message)s", level=logging.INFO)


class ReIDFeatureExtractor:
    """Extract 576‑D global embeddings using MobileNet V3‑Small."""

    def __init__(
        self,
        model_id: str = "timm/mobilenetv3_small_100.lamb_in1k",
        device: str | None = None,
        target_size: Tuple[int, int] = (224, 224),  # kept for manual resize, if needed
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_size = target_size
        self.model_id = model_id

        # ── Load backbone + image processor ──────────────────────────────
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        # timm backbones need `trust_remote_code=True`
        self.model = (
            AutoModel.from_pretrained(model_id, trust_remote_code=True)
            .to(self.device)
            .eval()
        )
        logger.info("Loaded %s on %s", model_id, self.device)

    # ────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def extract_features(
        self, images: List[np.ndarray], batch_size: int = 64
    ) -> np.ndarray:
        """Return an (N, 576) float32 NumPy array of unit‑norm vectors."""

        feats: list[np.ndarray] = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # OpenCV delivers BGR → convert to RGB first
            batch_rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in batch]

            # Let the processor handle resize & normalisation (default 224×224)
            inputs = self.processor(images=batch_rgb, return_tensors="pt").to(self.device)

            outputs = self.model(**inputs)  # (B, 576, H, W)
            featmap = outputs.last_hidden_state
            vec = torch.nn.functional.normalize(featmap.mean(dim=(-2, -1)), dim=1)
            feats.append(vec.cpu().numpy())

        return np.concatenate(feats, axis=0)


# ────────────────────────────── Quick sanity check ───────────────────────
if __name__ == "__main__":
    logger.info("Running a dummy forward pass to verify the model …")

    extractor = ReIDFeatureExtractor()

    # Random RGB tensor shaped like a real image
    dummy_input = torch.rand(1, 3, 224, 224, device=extractor.device)

    with torch.inference_mode():
        featmap = extractor.model(pixel_values=dummy_input).last_hidden_state
        pooled = torch.nn.functional.normalize(featmap.mean(dim=(-2, -1)), dim=1)

    print(
        "Dummy forward OK – feature vector shape:", pooled.shape,
        "| L2‑norm:", pooled.norm(dim=1).cpu().tolist()
    )
