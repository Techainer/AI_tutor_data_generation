import cv2
import os
import numpy as np
from loguru import logger
import tritonclient.grpc as input_client
from .utils import *
from .lib_layout_db.lib_layout_db import load_layout_db_model
from .lib_ocr_onmt.ocr_vietocr import load_ocr_model

class RetryInferenceServerClient(input_client.InferenceServerClient):
    def __init__ (self, *args, **kwargs):
        self.reconnect_time = kwargs.pop('reconnect_time', 100)
        super().__init__(*args, **kwargs)

        self.current_infer_time = 0
        self.args = args
        self.kwargs = kwargs

    def infer(self, *args, **kwargs):
        try:
            self.current_infer_time += 1
            if self.current_infer_time == self.reconnect_time:
                logger.info('Reinit client')
                super().__init__(*self.args, **self.kwargs)
                self.current_infer_time = 0
            return super().infer(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Inference failed with error: {e}. Retrying...")
            return super().infer(*args, **kwargs)
class CropExerciseDBModel:
    def __init__(self, debug=False):
                
        self.debug = debug
        triton_client = RetryInferenceServerClient(
                    url="localhost:11001", verbose=False, reconnect_time=100
                )

        self.layout_model = load_layout_db_model(
            flow_mode="triton",
            triton_client=triton_client,
            use_gpu=False,
            limit_gpu_mem=500,
            use_tensorrt=False,
            use_fp16=False,
            max_batch_size=32,
            enable_mkldnn=False,
            layout_model_path="models/card_textline_detection/1/optimized_model_20240624.onnx"
        )

        self.ocr_model = load_ocr_model(
                    flow_mode="triton",
                    triton_client=triton_client,
                    ocr_weight_path="models/card_ocr_vietocr/1/scripted_resnet50_transformer_vn.pt",
                    ocr_vocab_path="/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/modules/lib_ocr_onmt/models/card_ocr_vietocr/1/config_resnet_trans.yml",
                    ocr_confidence_threshold=0.6
                )

    def crop_and_ocr(self, image, line):
        line_temp = [tuple(np.array(each).astype(np.int32)) for each in line]
        line_cropped, bbox = cut_polygon(image, line_temp, text_line=True, fx=0.01, fy=0.08)
        if line_cropped.shape[0] <= 0 or line_cropped.shape[1] <= 0:
            return None
        
        line = sort_polys(
                [
                    (int(e[0]), int(e[1]))
                    for e in cv2.boxPoints(cv2.minAreaRect(np.array(line)))
                ]
            )

        if line_cropped is None:
            return None
        if (
            min(line_cropped.shape[0], line_cropped.shape[1]) > 10
            and line_cropped.shape[0] > line_cropped.shape[1] * 4
        ):
            line_cropped = np.rot90(line_cropped, 1)

        ocr_output = self.ocr_model.predict(line_cropped)
        return ocr_output, bbox
            

    def process(self, image: np.ndarray, page_idx: int=0):
        image_resized, target_ratio = resize_aspect_ratio(
                    image, square_size=1024, interpolation=cv2.INTER_CUBIC, mag_ratio=1.49
                )
        ratio_h = 1 / target_ratio
        doc = self.layout_model.predict(img=image_resized) 

        doc = [
                {
                    "id": idx,
                    "polys": [(int(x[0] * ratio_h), int(x[1] * ratio_h)) for x in line],
                }
                for idx, line in enumerate(doc)
            ]

        if self.debug:
            viz_img = image.copy()
            for line in doc:
                polygon = line["polys"]
                normed_polygon = np.array(polygon).astype(np.int32).reshape(-1, 2)
                cv2.polylines(
                    viz_img, [normed_polygon], True, color=(0, 0, 255), thickness=2
                )
            output_layout_path = os.path.join(f"debug/page_{page_idx}/layout.png")
        cv2.imwrite(output_layout_path, viz_img)
        
        potential_line = []
        for line in doc:
            text, bbox = self.crop_and_ocr(image, line['polys'])
            if text['prediction'].startswith("Câu") \
                or text['prediction'].startswith("Cấu") \
                or text['prediction'].startswith("Cầu") \
                or text['prediction'].startswith("Cẩu") \
                or text['prediction'].startswith("Côu"):
                potential_line.append(bbox[1])
                
        potential_line.sort()
        return potential_line
        
        
        
                