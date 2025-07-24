import os

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import logging
import time
from typing import List, Tuple

import cv2
import numpy as np

from .postprocess import DBPostProcess, filter_tag_det_res
from .preprocess import resize_image_to_multiple_of_32, NormalizeImage
from .utils import draw_text_det_res, mask_to_heatmap, overlay_image
import uuid


logger = logging.getLogger()


class DBTextDetectorBase:
    def __init__(self,
                 limit_gpu_mem: int = 500,
                 use_tensorrt: bool = False,
                 use_fp16: bool = False,
                 max_batch_size: int = 32,
                 enable_mkldnn: bool = False,
                 debug: bool = False, *args, **kwargs):
        self.normalizer = NormalizeImage(std=[0.229, 0.224, 0.225],
                                        mean=[0.485, 0.456, 0.406],
                                        scale='1./255.',
                                        order='hwc')
        self.postprocess_op = DBPostProcess(max_candidates=1000,
                                            use_dilation=True)
        
        self.debug = debug in ['True', True, 'true']
        self.debug_dir = './debugs'
        if self.debug:
            os.makedirs(self.debug_dir, exist_ok=True)
        logger.info("Init DBTextDetector sucessfully!")

    def preprocess(self, 
                   img: np.ndarray,
                   canvas_size: int,
                   limit_type: str = 'max') -> Tuple[np.ndarray, np.ndarray]:
        start_time = time.time()

        # First resize the image
        resized_image, shape_list = resize_image_to_multiple_of_32(img, limit_side_len=canvas_size, limit_type=limit_type)
        if self.debug:
            debug_path = os.path.join(self.debug_dir, self.debug_id)
            cv2.imwrite(os.path.join(debug_path, 'preprocessed.png'), resized_image)
 
        # Then normalize it
        input_batch = self.normalizer(resized_image)

        # Then convert hwc image to chw image
        input_batch = input_batch.transpose((2, 0, 1))
        input_batch = np.expand_dims(input_batch, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        logger.info(f"Preprocess for {self.debug_id} tooks {time.time()-start_time} seconds")
        logger.info(f"Batch shape after preprocessing for {self.debug_id} is {input_batch.shape}")
        return input_batch, shape_list, resized_image

    def forward(self, input_batch: np.ndarray) -> List[np.ndarray]:
        raise NotImplementedError()

    def postprocess(self,
                    outputs: np.ndarray, 
                    shape_list: np.ndarray,
                    thresh: float,
                    box_thresh: float,
                    unclip_ratio: float,
                    resized_image: np.ndarray,
                    return_raw_polygon: bool = False) -> np.ndarray:
        start_time = time.time()

        preds = {'maps': outputs[0]}
        orignal_shape = shape_list[0][:2]
        post_result, raw_mask, thresholded_mask = self.postprocess_op(preds,
                                                                     shape_list,
                                                                     thresh,
                                                                     box_thresh,
                                                                     unclip_ratio,
                                                                     return_raw_polygon)
        if self.debug:
            debug_path = os.path.join(self.debug_dir, self.debug_id)
            raw_mask = mask_to_heatmap(raw_mask[0])
            cv2.imwrite(os.path.join(debug_path, 'mask.png'), raw_mask)
            cv2.imwrite(os.path.join(debug_path, 'thresholded_mask.png'), thresholded_mask[0]*255)
            # And don't forget to overlay the mask on to the resized image to clearly see the output mask
            out = overlay_image(resized_image, raw_mask, alpha=0.5)
            cv2.imwrite(os.path.join(debug_path, 'output.png'), out)

        dt_boxes = list(zip(post_result[0]['points'], post_result[0]['scores']))
        dt_boxes = filter_tag_det_res(dt_boxes, orignal_shape)
        logger.info(f"Postprocessing for {self.debug_id} tooks {time.time()-start_time} seconds")
        return dt_boxes

    def predict(self, 
                img: np.ndarray,
                canvas_size: int = 960,
                limit_type: str = 'max',
                thresh: float = 0.3,
                box_thresh: float = 0.4,
                unclip_ratio: float = 1.6,
                return_raw_polygon: bool = True,
                debug_id: str = None) -> List[List[Tuple[int, int]]]:
        """
        Predict layout using Differentiable Binarization (DB) model for a single image.
        ### Parameters
            - `img` (np.ndarray): The input image (Should be BGR)
            - `limit_type` (str): 'max' or 'min', denote which side of the image to use to calculate new image size before inference.
            - `canvas_size` (int): Target size for the limited side above of the input image to be resized to before inference.
            - `thresh` (float): The threshold for binarization of the segmentation map in DBPostProcess. It must be set between 0 and 1.
            - `box_thresh` (float): The threshold for filtering output boxes in DBPostProcess. Boxes below this threshold will not be output.
            - `unclip_ratio` (float): The unclip ratio of the text box in DBPostProcess.
            - `return_raw_polygon` (bool): Whether or not to return raw polygon from mask

        ### Returns
            - `dt_boxes`: List of list of tuple (polygon) of all the textline detected in the input image
        """
        start_time = time.time()
        if debug_id is None:
            debug_id = str(uuid.uuid4())
        self.debug_id = debug_id
        logger.info(f"Received input {debug_id} with shape {img.shape}")
        if self.debug:
            debug_path = os.path.join(self.debug_dir, debug_id)
            os.makedirs(debug_path, exist_ok=True)
            cv2.imwrite(os.path.join(debug_path, 'raw.png'), img)

        # Preprocessing
        input_batch, shape_list, resized_image = self.preprocess(img,
                                                                 canvas_size,
                                                                 limit_type)

        # Forward pass
        outputs = self.forward(input_batch)

        # Postprocessing
        dt_boxes = self.postprocess(outputs,
                                    shape_list,
                                    thresh,
                                    box_thresh,
                                    unclip_ratio,
                                    resized_image,
                                    return_raw_polygon)
        if self.debug:
            debug_path = os.path.join(self.debug_dir, debug_id)
            final_visualization = draw_text_det_res(img, dt_boxes)
            cv2.imwrite(os.path.join(debug_path, 'final.png'), final_visualization)

        dt_boxes = [each['polygon'] for each in dt_boxes]
        elapse = time.time() - start_time
        logger.info(f"Prediction for {debug_id} tooks {elapse} seconds")
        return dt_boxes
