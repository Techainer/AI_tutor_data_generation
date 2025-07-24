from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper


class DBPostProcess():
    """
    The post process for Differentiable Binarization (DB).
    """

    def __init__(self,
                 max_candidates=1000,
                 use_dilation=False,
                 **kwargs):
        self.max_candidates = max_candidates
        self.min_size = 3
        self.dilation_kernel = None if not use_dilation else np.array(
            [[1, 1], [1, 1]])

    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height, box_thresh, unclip_ratio, return_raw_polygon):
        '''
        _bitmap: single map with shape (1, H, W),
                whose values are binarized as {0, 1}
        '''

        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        # boxes = []
        polygons = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]

            # NOTE: Kept this just to remove small textline
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            # score = self.box_score_fast(pred, points.reshape(-1, 2))
            score = self.countour_score_fast(pred, contour)
            if box_thresh > score:
                continue

            # NOTE: Trying to get the points from contour
            if return_raw_polygon:
                points = [each.tolist() for each in np.squeeze(contour)]

            polygon = self.unclip(points, unclip_ratio, is_polygon=return_raw_polygon).reshape(-1, 1, 2)

            # NOTE: Kept this just to remove small textline
            box, sside = self.get_mini_boxes(polygon)
            if sside < self.min_size + 2:
                continue
            if not return_raw_polygon:
                box = np.array(box)

                box[:, 0] = np.clip(
                    np.round(box[:, 0] / width * dest_width), 0, dest_width)
                box[:, 1] = np.clip(
                    np.round(box[:, 1] / height * dest_height), 0, dest_height)
                polygons.append(box.astype(np.int16))
            else:
                # NOTE: Just trying to mimic format of polygon to the same as box.
                # This is unnecessary and should be clean up in the future.
                polygon = np.array([each[0].astype(np.float32) for each in polygon])

                # NOTE: Try to return polygon instead of box
                polygon[:, 0] = np.clip(
                    np.round(polygon[:, 0] / width * dest_width), 0, dest_width)
                polygon[:, 1] = np.clip(
                    np.round(polygon[:, 1] / height * dest_height), 0, dest_height)
                polygons.append(polygon.astype(np.int16))
            scores.append(score)
        # return np.array(boxes, dtype=np.int16), scores
        return polygons, scores # polygons cannot be stack to a np.array since they had different shape lol

    def unclip(self, box, unclip_ratio, is_polygon=False):
        poly = Polygon(box)
        distance = (poly.area * unclip_ratio / poly.length) #* 2.5
        if is_polygon:
            distance *= 1.45 # Increase the distance by 15%
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(distance)
        if len(expanded) > 1:
            pc = pyclipper.Pyclipper()
            pc.AddPaths(expanded, pyclipper.PT_SUBJECT, True)
            res = pc.Execute(pyclipper.CT_UNION, pyclipper.PFT_EVENODD,
                            pyclipper.PFT_EVENODD)
            # NOTE: This might return multiple polygons that are not even able to be join with pyclipper.CT_UNION
            if len(res) > 1:
                res = max(expanded, key=lambda x: len(x))
            expanded = [res]
        return np.array(expanded)

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int64), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int64), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int64), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int64), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def countour_score_fast(self, bitmap, _contour):
        h, w = bitmap.shape[:2]
        contour = _contour.copy().squeeze()
        xmin = np.clip(np.floor(contour[:, 0].min()).astype(np.int64), 0, w - 1)
        xmax = np.clip(np.ceil(contour[:, 0].max()).astype(np.int64), 0, w - 1)
        ymin = np.clip(np.floor(contour[:, 1].min()).astype(np.int64), 0, h - 1)
        ymax = np.clip(np.ceil(contour[:, 1].max()).astype(np.int64), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin
        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

    def __call__(self,
                 outs_dict, 
                 shape_list,
                 thresh: float = 0.3,
                 box_thresh: float = 0.5,
                 unclip_ratio: float = 1.6,
                 return_raw_polygon: bool = False):
        pred = outs_dict['maps']
        pred = pred[:, 0, :, :]
        segmentation = pred > thresh

        thresholded_mask = []
        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, ratio_h, ratio_w = shape_list[batch_index]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(
                    np.array(segmentation[batch_index]).astype(np.uint8),
                    self.dilation_kernel)
            else:
                mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask,
                                                   src_w, src_h, box_thresh, unclip_ratio, return_raw_polygon)
            thresholded_mask.append(mask)
            boxes_batch.append({'points': boxes, 'scores': scores})
        thresholded_mask = np.stack(thresholded_mask)
        raw_mask = pred
        return boxes_batch, raw_mask, thresholded_mask

def order_points_clockwise(pts):
    """
    reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
    # sort the points based on their x-coordinates
    """
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    rect = np.array([tl, tr, br, bl], dtype="float32")
    return rect

def clip_det_res(points, img_height, img_width):
    for pno in range(points.shape[0]):
        points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
        points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
    return points

def filter_tag_det_res(dt_boxes, image_shape):
    img_height, img_width = image_shape[0:2]
    dt_boxes_new = []
    for box, score in dt_boxes:
        # box = order_points_clockwise(box)
        box = clip_det_res(box, img_height, img_width)
        # rect_width = int(np.linalg.norm(box[0] - box[1]))
        # rect_height = int(np.linalg.norm(box[0] - box[3]))
        # if rect_width <= 3 or rect_height <= 3:
        #     continue
        polygon = [(int(each[0]), int(each[1])) for each in box]
        dt_boxes_new.append({
            'polygon': polygon,
            'score': score
        })
    return dt_boxes_new
