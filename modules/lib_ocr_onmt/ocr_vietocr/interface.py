import re
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from .config import Cfg
from .vocab import Vocab
from .utils import process_input

class VietOCRBase:
    def __init__(self,
                 ocr_vocab_path: str,
                #  ocr_weight_path: str,
                 debug: bool = False,
                 ocr_confidence_threshold:float=0.6,
                 **kwargs):
        self.debug = debug
        self.config = Cfg.load_config_from_file(ocr_vocab_path)
        self.vocab = Vocab(self.config['vocab'])
        self.ocr_confidence_threshold = ocr_confidence_threshold
        # Init model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, images: np.ndarray):
        raise NotImplementedError()

    @staticmethod
    def post_process(raw_result, threshold: float = 0.3):
        # Fix repeat last word
        raw_result['prediction_raw'] = raw_result['prediction']

        text = raw_result['prediction'].strip()
        words = text.split(' ')
        last_word = words[-1]
        regex = f"( {last_word}){{3,}}"
        try:
            raw_result['prediction'] = re.sub(regex, ' ' + last_word, text)
        except re.error:
            pass

        # Fix long trailing ...
        text = raw_result['prediction'].strip()
        regex = r"(\.){4,}"
        try:
            raw_result['prediction'] = re.sub(regex, '', text)
        except re.error:
            pass
        
        if raw_result['confidence'] < threshold:
            raw_result['prediction'] = ""

        return raw_result

    def predict(self, images: np.ndarray, post_process: bool = True, threshold=None):
        try:
            if threshold is None:
                threshold = self.ocr_confidence_threshold
            else:
                threshold = float(threshold)
        except:
            threshold = 0.2
        if not isinstance(images, list):
            images = [images]

        images = process_input(images, self.config['dataset']['image_height'], 
                                self.config['dataset']['image_min_width'], 
                                self.config['dataset']['image_max_width'],
                                )
        pred, score = self.forward(images)
        pred = pred.tolist()
        pred = self.vocab.batch_decode(pred)
        results = []
        for i, (p, s) in enumerate(zip(pred, score)):
            res_dict = {
                "confidence": float(s),
                "prediction": p,
                "idx": i
            }
            res_dict = self.post_process(res_dict, threshold=threshold)
            results.append(res_dict)
        if len(results) == 1:
            return results[0]
        
        return results
