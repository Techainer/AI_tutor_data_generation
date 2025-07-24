import os

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import logging
import time
from typing import List

import numpy as np

from .interface import DBTextDetectorBase

logger = logging.getLogger()


class DBTextDetectorLegacy(DBTextDetectorBase):
    def __init__(self,
                 layout_model_path: str,
                 use_gpu: bool = False,
                 limit_gpu_mem: int = 500,
                 use_tensorrt: bool = False,
                 use_fp16: bool = False,
                 max_batch_size: int = 32,
                 enable_mkldnn: bool = False,
                 debug: bool = False, **kwargs):
        super(DBTextDetectorLegacy, self).__init__(debug=debug)
        # init onnx model
        import onnxruntime
        device = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.predictor = onnxruntime.InferenceSession(layout_model_path, providers=device)
        self.debug = debug in ['True', True, 'true']

    def forward(self, input_batch: np.ndarray) -> List[np.ndarray]:
        start_time = time.time()
        input_batch = input_batch.copy() # NOTE: Not sure why we need this line but the forward pass result will be incorrect if we remove this line
        
        ort_inputs = {self.predictor.get_inputs()[0].name: input_batch.astype(np.float32)}
        ort_outputs = self.predictor.run(None, ort_inputs)
        outputs = []
        for output_tensor in ort_outputs:
            outputs.append(output_tensor)
        logger.info(f"Forward pass for {self.debug_id} tooks {time.time()-start_time} seconds")
        return outputs