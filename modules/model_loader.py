from ultralytics import YOLO, SAM, FastSAM
from ultralytics.engine.results import Results

import torch
import numpy as np
from typing import List


class ModelLoader:
    def __init__(
        self,
        weights_path: str,
        image_size: int = 640,
        confidence: float = 0.5,
        class_filter: List[int] = None,
        batch_size: int = 1,
        tracking: bool = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.model = YOLO(weights_path)
        self.image_size = image_size
        self.confidence = confidence
        self.class_filter = class_filter
        self.class_names = self.model.names
        self.batch_size = batch_size
        self.tracking = tracking
        self.device = device

    def _common_parameters(self) -> dict:
        return {
            'imgsz': self.image_size,
            'conf': self.confidence,
            'classes': self.class_filter,
            'device': self.device,
            'verbose': False,
            'stream': True,
            'agnostic_nms': True,
        }
    
    def _tracking_parameters(self) -> dict:
        """Return additional parameters used for tracking"""
        return {
            'persist': True,
            'tracker': "bytetrack.yaml"
        }
    
    def _segmentation_parameters(self) -> dict:
        return {
            'retina_masks': True
        }

    def inference(self, image: np.ndarray) -> Results:
        parameters = {
            'source': image,
            **self._common_parameters()
        }
        if self.tracking:
            parameters.update(self._tracking_parameters())
            return self.model.track(**parameters)

        return self.model(**parameters)
    

    def batch_inference(self, source: str) -> Results:
        parameters = {
            'source': source,
            'batch': self.batch_size,
            **self._common_parameters()
        }
        if self.tracking:
            parameters.update(self._tracking_parameters())
            return self.model.track(**parameters)

        return self.model(**parameters)
    