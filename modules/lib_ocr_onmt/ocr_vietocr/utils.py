import logging
from multiprocessing import Lock
import numpy as np
import math
import cv2

logger = logging.getLogger()

def resize(w, h, expected_height, image_min_width, image_max_width):
    new_w = int(expected_height * float(w) / float(h))
    round_to = 10
    new_w = math.ceil(new_w/round_to)*round_to
    new_w = max(new_w, image_min_width)
    new_w = min(new_w, image_max_width)

    return new_w, expected_height

def find_max_width(images):
    max_width = 0
    for img in images:
        if img.shape[2] > max_width:
            max_width = img.shape[2]
    return max_width

def process_image_infer(imgs, image_height, image_min_width, image_max_width):
    def process_image(img, image_height, image_min_width, image_max_width):
        h, w = img.shape[0:2]
        new_w, image_height = resize(w, h, image_height, image_min_width, image_max_width)

        img = cv2.resize(img, (new_w, image_height))
        img = img.transpose(2, 0, 1)
        img = img/255
        return img.astype('float32')
    if len(imgs) == 1:
        return process_image(imgs[0], image_height, image_min_width, image_max_width)
    else:
        output = []
        for img in imgs:
            img = process_image(img, image_height, image_min_width, image_max_width)
            output.append(img)
        max_width = find_max_width(output)
        processed_output = []
        for img in output:
            if img.shape[2] == max_width:
                processed_output.append(img)
                continue
            pad_img = np.zeros((img.shape[0], img.shape[1], max_width), dtype=np.float32)
            pad_img[:, :, 0:img.shape[2]] = img
            processed_output.append(pad_img)
        return np.array(processed_output, dtype="float32")

def process_input(image, image_height, image_min_width, image_max_width):
    img = process_image_infer(image, image_height, image_min_width, image_max_width)
    if len(img.shape) == 3:
        img = img[np.newaxis, ...]
    return img
