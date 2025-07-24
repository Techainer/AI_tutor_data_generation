import cv2
import numpy as np
import imutils
from shapely.geometry import Polygon
from shapely import affinity


def rotate_polygon(polygon, angle, raw_img_shape, offset_x=0, offset_y=0):
    try:
        p = Polygon(polygon)
    except ValueError:
        # Use bb instead
        all_x = [each[0] for each in polygon]
        all_y = [each[1] for each in polygon]
        x1, y1, x2, y2 = min(all_x), min(all_y), max(all_x), max(all_y)
        bb = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        p = Polygon(bb)
    p = affinity.rotate(p, angle, (raw_img_shape[1]//2, raw_img_shape[0]//2))
    new_p = p.exterior.coords[:]
    new_p = [(int(e[0]-offset_x), int(e[1]-offset_y)) for e in new_p][:-1]
    return new_p


def cut_polygon(image, org_pts, text_line=False, fx=0.0005, fy=0.1, additional_orientation=0):
    pts = np.array(org_pts)
    if not text_line:  # Cut bouding box of polygon
        pts = np.squeeze(pts)
        X = [each[0] for each in pts]
        Y = [each[1] for each in pts]
        x1, x2, y1, y2 = min(X), max(X), min(Y), max(Y)
        cropped = image[y1:y2, x1:x2]
        if additional_orientation != 0:
            cropped = imutils.rotate_bound(
                cropped, additional_orientation, flags=cv2.INTER_CUBIC)
    else:  # Only cut the polygon to avoid overlap other textline
       # Let's try to rotate the org page
        if additional_orientation != 0:
            before_shape = image.shape
            image = imutils.rotate_bound(
                image, angle=additional_orientation, cval=(255, 255, 255))
            full_page = [(0, 0), (before_shape[1], 0),
                         (before_shape[1], before_shape[0]), (0, before_shape[0])]
            rotated_full_page = rotate_polygon(full_page, additional_orientation, image.shape)
            # Calculate offset of rotation
            top_left_x = min([each[0] for each in rotated_full_page])
            top_left_y = min([each[1] for each in rotated_full_page])
            org_pts = rotate_polygon(
                org_pts, additional_orientation, image.shape, top_left_x, top_left_y)
            pts = np.array(pts)

        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        if angle < -45:
            angle += 90
        angle = -angle
        before_shape = image.shape
        image = imutils.rotate_bound(image, angle=angle, cval=(255, 255, 255))
        # And also rotate the polygon of text line
        full_page = [(0, 0), (before_shape[1], 0),
                     (before_shape[1], before_shape[0]), (0, before_shape[0])]
        rotated_full_page = rotate_polygon(full_page, angle, image.shape)
        # Calculate offset of rotation
        top_left_x = min([each[0] for each in rotated_full_page])
        top_left_y = min([each[1] for each in rotated_full_page])
        rotated_pts = rotate_polygon(
            org_pts, angle, image.shape, top_left_x, top_left_y)
        all_x = [each[0] for each in rotated_pts]
        all_y = [each[1] for each in rotated_pts]
        x1, y1, x2, y2 = min(all_x), min(all_y), max(all_x), max(all_y)
        # And extend the box a little bit
        gh, gw, _ = image.shape
        ch, cw = y2-y1, x2-x1

        if ch < 25:
            fy_down = 0.28
        elif ch > 100:
            fy_down = 0.15
        else:
            fy_down = 0.17

        x1 = max(0, int(x1-fx*cw))
        x2 = min(gw, int(x2+2*fx*cw))
        y1 = max(0, int(y1-fy*ch))
        y2 = min(gh, int(y2+fy_down*ch))
        cropped = image[y1:y2, x1:x2]
    return cropped
