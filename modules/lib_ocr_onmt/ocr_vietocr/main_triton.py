from typing import Tuple

import numpy as np
# import tritonclient.http as input_client
import tritonclient.grpc as input_client

from .interface import VietOCRBase
import logging

def parse_model_path(model_path):
    _, model_name, model_version = model_path.split("/")[:3]
    return model_name, model_version

class VietOCRTriton(VietOCRBase):
    def __init__(self,
                 ocr_vocab_path: str,
                 ocr_weight_path: str,
                 ocr_confidence_threshold:float=0.6,
                 debug: bool = False,
                 **kwargs):
    
        super(VietOCRTriton, self).__init__(ocr_vocab_path=ocr_vocab_path, 
                                        ocr_confidence_threshold=ocr_confidence_threshold,
                                        debug=debug)
        
        self.model_name, self.model_version = parse_model_path(ocr_weight_path)

        triton_client = kwargs.get('triton_client', None)
        if triton_client is not None:
            self.triton_client = triton_client
        else:
            raise ValueError("Triton client is not provided!")
        logging.info("Init VietOCRTriton successfully!")

    def forward(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inputs = [input_client.InferInput('INPUT__0', images.shape, 'FP32')]
        inputs[0].set_data_from_numpy(images)
        outputs = [input_client.InferRequestedOutput('OUTPUT__0'),
                    input_client.InferRequestedOutput('OUTPUT__1')]
        responses = self.triton_client.infer(self.model_name,
                                             inputs,
                                             model_version=self.model_version,
                                             outputs=outputs)
        preds = responses.as_numpy('OUTPUT__0')
        scores = responses.as_numpy('OUTPUT__1')
        return preds, scores
