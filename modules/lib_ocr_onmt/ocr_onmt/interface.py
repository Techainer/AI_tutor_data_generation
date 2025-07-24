import re
from typing import List

import numpy as np
import torch
from PIL import Image


class ONMTModelBase:
    def __init__(self,
                 ocr_vocab_path: str,
                 debug: bool = False,
                 ocr_confidence_threshold:float=0.6,
                 **kwargs):
        self.debug = debug in ['True', True, 'true']
        self.vocab = torch.load(ocr_vocab_path)["vocab"] 
        self.eos_token = "</s>"
        self.ocr_confidence_threshold = ocr_confidence_threshold

    def forward(self, image: np.ndarray):
        raise NotImplementedError

    def predict(self, image: np.ndarray, post_process: bool = True, threshold=None):
        try:
            if threshold is None:
                threshold = self.ocr_confidence_threshold
            else:
                threshold = float(threshold)
        except:
            threshold = 0.2

        image = ONMTModelBase.preprocess(image)
        score, pred = self.forward(image)
        score = score[0]
        pred = pred[0]

        pred = self._build_target_tokens(pred)
        pred = [" ".join("".join(x.replace(r"\,", "").split()).split("\;")) for x in pred]
        pred = "".join(pred)
        raw_result = {
            "confidence": np.exp(score).item(),
            "prediction": pred 
        }
        if post_process:
            return self.post_process(raw_result, threshold)
        else:
            return raw_result

    def predict_batch(self, images: List[np.ndarray], ratio_diff: float, max_batch_size: int = 32, post_process: bool = True, threshold=None):
        try:
            threshold = float(threshold)
        except:
            threshold = 0.2

        # identity function if not using post_process
        post_process_fn = lambda x, t: x
        if post_process:
            post_process_fn = self.post_process

        results = []
        images_with_idx = list(enumerate(images))
        micro_batches, micro_batches_idx = ONMTModelBase.preprocess_batch(images_with_idx, ratio_diff, max_batch_size)
        for batch_idx, batch in zip(micro_batches_idx, micro_batches):
            scores, preds = self.forward(batch)
            for idx, score, pred in zip(batch_idx, scores, preds):
                pred = self._build_target_tokens(pred)
                pred = [" ".join("".join(x.replace(r"\,", "").split()).split("\;")) for x in pred]
                pred = "".join(pred)
                result = {
                    "confidence": np.exp(score).item(),
                    "prediction": pred,
                    "idx": idx
                }
                results.append(post_process_fn(result, threshold))
        return sorted(results, key=lambda res: res["idx"])

    def _build_target_tokens(self, pred):
        tokens = []
        for tok in pred:
            tokens.append(self.vocab[int(tok)])
            if tokens[-1] == self.eos_token:
                tokens = tokens[:-1]
                break
        return tokens

    @staticmethod
    def preprocess_batch(images: List[np.ndarray], ratio_diff: float = 0.4, max_batch_size: int = 32) -> List[np.ndarray]:
        target_h = 64
        mean = np.array([[123.675, 116.28 , 103.53]])
        std = np.array([[58.395, 57.12 , 57.375]])

        images = sorted(images, key=lambda img: img[1].shape[1] / img[1].shape[0])
        images_ratio = [(idx, image.shape[1] / image.shape[0]) for idx, image in images]

        pivot = images_ratio[0][1]
        batching_indices = [0]  # Init first index
        for idx, (_, ratio) in enumerate(images_ratio[1:]):
            # check whether excess max_batch_size
            if ((idx - batching_indices[-1] + 1 == max_batch_size)):
                batching_indices.append(idx+1)
                pivot = ratio
                continue
            if (ratio - pivot >= ratio_diff):
                batching_indices.append(idx+1)
                pivot = ratio
        batching_indices.append(len(images_ratio))    # Append end index

        micro_batches = []
        micro_batches_idx = []
        for start_idx, end_idx in zip(batching_indices[0:], batching_indices[1:]):
            # pivot_image = images[start_idx][1]
            # new_shape = (max(int(pivot_image.shape[1] * target_h / pivot_image.shape[0] / 4) * 4, 4), target_h)
            # micro_batch = []
            # micro_batch_idx = []
            # for idx, image in images[start_idx:end_idx]:
            #     image = Image.fromarray(image).resize(new_shape, Image.LANCZOS)
            #     micro_batch.append(np.array(image))
            #     micro_batch_idx.append(idx)
            pivot_image = images[end_idx-1][1]
            padded_new_shape = (max(int(pivot_image.shape[1] * target_h / pivot_image.shape[0] / 4) * 4, 4), target_h)
            micro_batch = []
            micro_batch_idx = []
            for idx, image in images[start_idx:end_idx]:
                new_shape = (max(int(image.shape[1] * target_h / image.shape[0] / 4) * 4, 4), target_h)
                image = Image.fromarray(image).resize(new_shape, Image.LANCZOS)
                image = np.array(image)
                image = np.pad(image, pad_width=((0,0), (0,padded_new_shape[0]-new_shape[0]), (0,0)), mode="constant", constant_values=0)
                micro_batch.append(image)
                micro_batch_idx.append(idx)
                # 
            micro_batches_idx.append(micro_batch_idx)
            micro_batch = np.array(micro_batch)
            micro_batch = (micro_batch - mean) / std
            micro_batch = micro_batch.transpose(0, 3, 1, 2).astype(np.float32)
            micro_batches.append(micro_batch)
        return micro_batches, micro_batches_idx
    
    @staticmethod
    def preprocess(image: np.ndarray):
        min_h = 64
        max_h = 64
        mean = np.array([123.675, 116.28 , 103.53])
        std = np.array([58.395, 57.12 , 57.375])
        if (image.shape[0] <= min_h):
            new_width = max(int(image.shape[1] * min_h / image.shape[0] / 4) * 4, 4)
            image = Image.fromarray(image).resize((new_width, min_h), Image.LANCZOS)
        elif (image.shape[0] > max_h):
            new_width = max(int(image.shape[1] * max_h / image.shape[0] / 4) * 4, 4)
            image = Image.fromarray(image).resize((new_width, max_h), Image.LANCZOS)
        image = (image - mean) / std
        image = np.array(image).transpose(2, 0, 1).astype(np.float32)
        image = np.expand_dims(image, axis=0)
        return image

    @staticmethod
    def post_process(raw_result, threshold: float = 0.6):
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


if __name__ == "__main__":
    import glob
    import time

    import cv2

    all_image_paths = glob.glob("/mnt/ssd/santapo/revamp_idcard/lib_ocr_onmt/tests/sample_dataset/images/*.jpg")
    all_images = [cv2.imread(image_path) for image_path in all_image_paths]

    # model = ONMTModel
    start_time = time.time()
    micro_batches = ONMTModelBase.predict_batch(all_images)
    print("--- %s seconds ---" % (time.time() - start_time))

    for batch in micro_batches:
        print(batch.shape)