from typing import Tuple

import numpy as np
import torch
import logging

from .interface import VietOCRBase


class VietOCRLegacy(VietOCRBase):
    def __init__(self,
                 ocr_vocab_path: str,
                 ocr_weight_path: str,
                 ocr_confidence_threshold:float=0.4,
                 debug: bool = False,
                 **kwargs):
        
        super(VietOCRLegacy, self).__init__(
                                # ocr_weight_path=ocr_weight_path,
                                ocr_vocab_path=ocr_vocab_path, 
                                debug=debug, 
                                ocr_confidence_threshold=ocr_confidence_threshold)
        # Init model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(ocr_weight_path)
        self.model = torch.jit.load(ocr_weight_path)
        self.model.to(self.device)

        logging.info("Init VietOCRLegacy successfully!")

    @torch.no_grad()
    def forward(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        images = torch.from_numpy(images).to(self.device)
        preds, scores = self.model(images)

        scores = scores.cpu().numpy()
        preds = preds.cpu().numpy()
        return preds, scores
