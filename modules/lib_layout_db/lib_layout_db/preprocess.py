import cv2
import numpy as np


def resize_image_to_multiple_of_32(img: np.ndarray, 
                                    limit_side_len: int,
                                    limit_type: str = 'max'):
    """
    resize image to a size multiple of 32 which is required by the network
    args:
        img(array): array with shape [h, w, c]
    return(tuple):
        img, (h, w, ratio_h, ratio_w)
    """
    if limit_type not in ['min', 'max']:
        raise ValueError(f"limit_type must be either 'min' or 'max', not {limit_type}")

    h, w, _ = img.shape

    # limit the max side
    if limit_type == 'max':
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            # ratio = 1.
            # NOTE: The original implementation from Paddle OCR does not scale the input image if
            # it is smaller than the limit_side_len, for the default configuration of the online
            # augmentation, I find that scaling it up as well achive massive accuracy gain so that why 
            # I did it here for now temporarily. The best solution should be training with smaller scale
            # to achive better performance as well as better accuracy.
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
    else:
        if min(h, w) < limit_side_len:
            if h < w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            # ratio = 1.
            if h < w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
    resize_h = int(h * ratio)
    resize_w = int(w * ratio)

    resize_h = int(round(resize_h / 32) * 32)
    resize_w = int(round(resize_w / 32) * 32)

    try:
        if int(resize_w) <= 0 or int(resize_h) <= 0:
            return None, (None, None)
        resized_img = cv2.resize(img, (int(resize_w), int(resize_h)))
    except:
        raise ValueError(f"Failed to resize image with shape: {img.shape}. Target width: {resize_w}. Target height: {resize_h}")
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return resized_img, np.array([h, w, ratio_h, ratio_w])


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self, scale=None, mean=None, std=None, order='chw', **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img: np.ndarray):
        return (img.astype('float32') * self.scale - self.mean) / self.std