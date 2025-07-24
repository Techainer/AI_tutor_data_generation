from typing import Tuple

import numpy as np
import torch
import logging

from .interface import ONMTModelBase


class ONMTModelLegacy(ONMTModelBase):
    def __init__(self,
                 ocr_vocab_path: str,
                 ocr_weight_path: str,
                 ocr_confidence_threshold:float=0.6,
                 debug: bool = False):
        
        super(ONMTModelLegacy, self).__init__(ocr_vocab_path=ocr_vocab_path, 
                                debug=debug, 
                                ocr_confidence_threshold=ocr_confidence_threshold)
        # Init model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = torch.jit.load(ocr_weight_path)
        self.model.to(self.device)

        logging.info("Init ONMTModelLegacy successfully!")

    @torch.no_grad()
    def forward(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        images = torch.from_numpy(images).to(self.device)
        scores, preds, indices = self.model(images)

        scores = scores.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        indices = indices.detach().cpu().numpy()

        preds = np.split(preds, indices)
        return scores, preds
