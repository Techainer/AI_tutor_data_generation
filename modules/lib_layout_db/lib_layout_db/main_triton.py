import os

from .interface import DBTextDetectorBase

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import logging
import time
from typing import List

import numpy as np
import tritonclient.grpc as input_client

logger = logging.getLogger()

def parse_model_path(model_path):
    _, model_name, model_version = model_path.split("/")[:3]
    return model_name, model_version

class DBTextDetectorTriton(DBTextDetectorBase):
    def __init__(self, layout_model_path: str, **kwargs):
        debug = kwargs.get("debug", False)
        super(DBTextDetectorTriton, self).__init__(debug=debug)
        self.model_name, self.model_version = parse_model_path(layout_model_path)
        triton_client = kwargs.get('triton_client', None)
        if triton_client is not None:
            self.triton_client = triton_client
        else:
            raise ValueError("Triton client is not provided")

    def forward(self, input_batch: np.ndarray) -> List[np.ndarray]:
        start_time = time.time()

        input_batch = input_batch.copy() # NOTE: Not sure why we need this line but the forward pass result will be incorrect if we remove this line

        inputs = [input_client.InferInput('x', input_batch.shape, 'FP32')]
        inputs[0].set_data_from_numpy(input_batch)
        outputs = [input_client.InferRequestedOutput('sigmoid_121.tmp_0')]
        responses = self.triton_client.infer(self.model_name,
                                             inputs,
                                             model_version=self.model_version,
                                             outputs=outputs)
        preds = responses.as_numpy('sigmoid_121.tmp_0')

        outputs = [preds]
        logger.info(f"Forward pass for {self.debug_id} tooks {time.time()-start_time} seconds")
        return outputs
