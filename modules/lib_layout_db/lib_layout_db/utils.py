import logging
import cv2
import numpy as np

logger = logging.getLogger()

def draw_text_det_res(src_img: np.ndarray, dt_boxes, draw_conf: bool = True):
    viz_img = src_img.copy()
    for each in dt_boxes:
        polygon = each['polygon']
        normed_polygon = np.array(polygon).astype(np.int32).reshape(-1, 2)
        cv2.polylines(viz_img, [normed_polygon], True, color=(0, 0, 255), thickness=2)

    if draw_conf: # To make sure all text stay infront of the polygon border
        for each in dt_boxes:
            polygon = each['polygon']
            score = each['score']
            all_x = [e[0] for e in polygon]
            all_y = [e[1] for e in polygon]
            min_x, max_x, min_y, max_y = min(all_x), max(all_x), min(all_y), max(all_y)
            width, height = max_x-min_x, max_y-min_y
            x = max_x - int(width*0.15)
            y = min_y + int(height*0.15)
            viz_img = cv2.putText(viz_img, "{:.2f}".format(score), (x, y), cv2.FONT_HERSHEY_SIMPLEX ,
                0.25, (0, 0, 255), 1, cv2.LINE_AA)
    return viz_img

def mask_to_heatmap(img: np.ndarray) -> np.ndarray:
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

def overlay_image(img1: np.ndarray, img2: np.ndarray, alpha:float = 0.5) -> np.ndarray:
    assert 0 <= alpha <= 1
    out = img1.copy().astype(np.uint8)
    img2 = img2.astype(np.uint8)
    out = cv2.addWeighted(img2, alpha, out, 1-alpha, 0, out)
    return out
