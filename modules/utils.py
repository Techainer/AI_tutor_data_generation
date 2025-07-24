import base64
import fitz
from PIL import Image
from typing import List
import io

def image_to_base64(byte_image):
    image_base64 = base64.b64encode(byte_image).decode('utf-8')
    img_str = f"data:image/jpeg;base64,{image_base64}"
    return img_str

def pdf_pages_to_images(pdf_path, dpi=150) -> List[bytes]:  # Default to 300 DPI, which is print quality
    pdf_document = fitz.open(pdf_path)
    pages_as_images = []

    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        
        # Calculate zoom factor based on desired DPI (default is 72 DPI)
        zoom_factor = dpi / 72
        
        # Create a higher resolution pixmap with the zoom factor
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        pix = page.get_pixmap(matrix=matrix)
        
        # Convert the pixmap to a Pillow image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Save the Pillow image in byte format
        img_byte_arr = io.BytesIO()
        # For better quality when saving as JPEG, you can specify quality
        img.save(img_byte_arr, format='JPEG', quality=95)
        pages_as_images.append(img_byte_arr.getvalue())

    pdf_document.close()
    return pages_as_images

import base64
import logging
import operator
import os
import tempfile
import traceback
from math import pi

import cv2
import uuid
import fitz
import numpy as np
import unidecode
from imutils.video import FileVideoStream
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import distance as dist
from shapely import affinity
from shapely.geometry import Polygon

logger = logging.getLogger()
LINE_THRESHOLD_RATIO = 3


class MlChainError(Exception):
    """Base class for all exceptions."""

    def __init__(self, msg, code='exception', status_code=500):
        super(MlChainError, self).__init__(msg)
        self.msg = msg
        self.message = msg
        self.code = code
        self.status_code = status_code

def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im
    
def pdf_to_list_ndarray(filename, data_bytes):
    doc = fitz.Document(stream=data_bytes, filetype='pdf')
    # doc = fitz.open(pdf_path)
    scale= 3.0
    mat = fitz.Matrix(scale, scale)

    output = []
    for index, page in enumerate(doc):
        pix = page.getPixmap(alpha=False, matrix=mat)
        image_page = pix2np(pix)

        output.append(image_page)

    return output

def bytes_video_to_ndarray_imutils(input: bytes = None, fps=5, max_process_frame=10): 
    if input is None: 
        return []

    output = []
    with tempfile.NamedTemporaryFile() as tfile:
        tfile.write(input)

        fvs = FileVideoStream(tfile.name).start()
        # start the FPS timer
        # fps = FPS().start()

        # loop over frames from the video file stream
        while fvs.more():
            # grab the frame from the threaded video file stream, resize
            # it, and convert it to grayscale (while still retaining 3
            # channels)
            frame = fvs.read()
            if frame is not None:
                output.append(frame)
            # fps.update()

        # stop the timer and display FPS information
        # fps.stop()

    return [output[i] for i in np.round(np.linspace(0, len(output) - 1, max_process_frame)).astype(int)]

def load_video_from_storage(input, fps=5, max_process_frame=10, max_iter = 500): 
    """
    Load video bytes and ndarray from flask storage
    """
    if type(input).__name__ != 'FileStorage': 
        raise MlChainError(
                    msg="Input video can not be None",
                    code="E812",
                    status_code=200
                )

    file_name = input.filename
    file_ext = unidecode.unidecode(file_name.split(".")[-1].lower())
    try:
        bytes_data = input.read()
        # vr = VideoReader(BytesIO(bytes_data))
        # output_images = [cv2.cvtColor(vr[i].asnumpy(), cv2.COLOR_BGR2RGB) for i in np.round(np.linspace(0, len(vr) - 1, max_process_frame)).astype(int)]
        output_images = bytes_video_to_ndarray_imutils(bytes_data, fps=fps, max_process_frame=max_process_frame)
    except Exception as ex: 
        logging.info("READ VIDEO ERROR: {0}".format(traceback.format_exc()))
        raise MlChainError(
            msg="The input video is not in true format of {0}".format(file_ext),
            code="E814",
            status_code=200
        )

    return file_name, file_ext, bytes_data, output_images

def order_points(pts):
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return tl, tr, br, bl


PADDING_THRESHOLD = 7


def box_compare_function(u, v):
    u_x1, u_x2, u_y1, u_y2 = u['box']
    v_x1, v_x2, v_y1, v_y2 = v['box']
    middle_u = (u_y1 + u_y2)/2
    middle_v = (v_y1 + v_y2)/2

    if u_y1 > v_y2 - PADDING_THRESHOLD or (middle_u > v_y2 and middle_v < u_y1):
        return 1
    elif u_y2 < v_y1 + PADDING_THRESHOLD or (middle_u < v_y1 and middle_v > u_y2):
        return -1
    else:
        if u_x1 > v_x2 - PADDING_THRESHOLD:
            return 1
        elif u_x2 < v_x1 + PADDING_THRESHOLD:
            return -1
        else:
            if (min(u_x2, v_x2) - max(u_x1, v_x1)) * min(u_y2 - u_y1, v_y2 - v_y1) > min(u_x2 - u_x1, v_x2 - v_x1) * (min(u_y2, v_y2) - max(u_y1, v_y1)):
                if u_y1 < v_y1:
                    return -1
                else:
                    return 1
            else:
                return -1


def get_box_from_polys(x):
    x_x1 = min([u[0] for u in x['polys']])
    x_x2 = max([u[0] for u in x['polys']])

    x_y1 = min([u[1] for u in x['polys']])
    x_y2 = max([u[1] for u in x['polys']])
    return x_x1, x_x2, x_y1, x_y2


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


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1.5, square=True):
    height, width, channel = img.shape

    # magnify image size
    target_size = int(mag_ratio * max(height, width))

    # set original image size
    if int(target_size) > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    if square:
        # make canvas and paste image
        target_h32, target_w32 = target_h, target_w
        if target_h % 32 != 0:
            target_h32 = target_h + (32 - target_h % 32)
        if target_w % 32 != 0:
            target_w32 = target_w + (32 - target_w % 32)
        resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
        resized[0:target_h, 0:target_w, :] = proc
    else:
        resized = proc

    return resized, ratio


def rotate_bound(image, angle, flags=cv2.INTER_CUBIC, cval=(0, 0, 0)):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=flags, borderValue=cval)


def cut_polygon(image, org_pts, text_line=False, fx=0.03, fy=0.05):
    pts = np.array(org_pts)
    if not text_line:  # Cut bouding box of polygon
        pts = np.squeeze(pts)
        X = [each[0] for each in pts]
        Y = [each[1] for each in pts]
        x1, x2, y1, y2 = min(X), max(X), min(Y), max(Y)
        cropped = image[y1:y2, x1:x2]
    else:  # Only cut the polygon to avoid overlap other textline
       # Let's try to rotate the org page
        rect = cv2.minAreaRect(pts)
        angle = rect[2]
        if abs(angle) > 45:
            if angle > 0:
                angle = 90 - angle
            else:
                angle = -90 + angle
        else:
            angle = -angle
        before_shape = image.shape
        image = rotate_bound(image, angle=angle, cval=(255, 255, 255))
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
            fy_down = 0.3
        elif ch > 100:
            fy_down = 0.12
        else:
            fy_down = 0.2
            
        x1 = max(0, int(x1-fx*cw))
        x2 = min(gw, int(x2+4*fx*cw))
        y1 = max(0, int(y1-fy*ch*0.1))
        y2 = min(gh, int(y2+fy_down*ch))
        cropped = image[y1:y2, x1:x2]
    return cropped, [x1,y1,x2,y2]


def calculate_roundness(polygon) -> float:
    # Polsby-Popper method
    shapely_polygon = create_shapely_polygon(polygon)
    p = shapely_polygon.length
    a = shapely_polygon.area
    # NOTE: Somehow sentry at VNM production env report that this polygon:
    # [[265, 125], [265, 124], [263, 124], [263, 124], [260, 124], [259, 124], [259, 123], [259, 122], [263, 122], [263, 122]]
    # has length of 0.0 and area of 0.0. Which is stange ..
    # I'm not sure why that is the case because when I use shapely 1.7.1 at my local machine,
    # the length and area are correct, and not 0
    # Anw, I left a check here to avoid having ZeroDivisionError in the future
    if p == 0:
        return 0
    return (4 * pi * a)/ p ** 2

def calculate_polygon_area(polygon) -> float:
    shapely_polygon = create_shapely_polygon(polygon)
    return shapely_polygon.area

def get_fine_rotation_angle(doc, raw_img):
    # Calculate angle vote from each line
    raw_h, raw_w = raw_img.shape[:-1]
    angle_dict = []
    for index, line in enumerate(doc):
        line_area = calculate_polygon_area(line['polys'])
        roundness = calculate_roundness(line['polys'])
        if line_area < 25 or roundness >= 0.8:
            # Don't bother rotating too round object
            angle = 0
        else:
            mask = np.zeros((raw_h, raw_w), dtype=np.uint8)
            mask = cv2.fillPoly(
                mask, pts=[np.array(line['polys'])], color=255)
            coords = np.column_stack(np.where(mask > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if abs(angle) > 45:
                if angle > 0:
                    angle = 90 - angle
                else:
                    angle = -90 + angle
            else:
                angle = -angle
        angle_dict.append(angle)
    angle_range = list(range(-90, 90, 10))
    angle_freq = {}
    for i in range(len(angle_range)-1):
        angle_freq[i] = 0
        for angle in angle_dict:
            if angle >= angle_range[i] and angle < angle_range[i+1]:
                angle_freq[i] += 1

    # Get the most populated cluster and calculate avg angle
    most_freq_index = max(angle_freq.items(),
                          key=operator.itemgetter(1))[0]
    angle_candidates = []
    for angle in angle_dict:
        if angle >= angle_range[most_freq_index] and angle < angle_range[most_freq_index+1]:
            angle_candidates.append(angle)
    angle = np.mean(angle_candidates)
    return -angle if not np.isnan(angle) else 0


def rotate_anno(layout, angle, raw_img_shape, before_shape=None):
    new_layout = []
    # Create full page mask
    old_shape = before_shape if before_shape is not None else raw_img_shape[::-1]
    full_page = [(0, 0), (old_shape[1], 0),
                 (old_shape[1], old_shape[0]), (0, old_shape[0])]
    rotated_full_page = rotate_polygon(full_page, angle, raw_img_shape)
    # Calculate offset of rotation
    top_left_x = min([each[0] for each in rotated_full_page])
    top_left_y = min([each[1] for each in rotated_full_page])
    for each in layout:
        each['polys'] = rotate_polygon(
            each['polys'], angle, raw_img_shape, top_left_x, top_left_y)
        new_layout.append(each)
    return new_layout


def rotate_image_and_layout(raw_layout_res, raw_img, angle):
    before_shape = raw_img.shape[:-1]
    raw_img = rotate_bound(raw_img, angle=angle)
    raw_layout_res = rotate_anno(
        raw_layout_res, angle, raw_img.shape[:-1], before_shape)
    return raw_layout_res, raw_img


def sort_polys(line):
    line = order_points(np.array(
        [[line[0][0], line[0][1]], [line[1][0], line[1][1]], [line[2][0], line[2][1]], [line[3][0], line[3][1]]]))
    line = [(line[0][0], line[0][1]), (line[1][0], line[1][1]), (line[2][0], line[2][1]), (line[3][0], line[3][1])]

    return line 


def resize_image(image: np.ndarray, target_size=1024):
    height, width = image.shape[:2]
    if max(height, width) > target_size:
        ratio = target_size / max(height, width)

        target_h, target_w = int(height * ratio), int(width * ratio)
        image = cv2.resize(image, (target_w, target_h),
                           interpolation=cv2.INTER_CUBIC)

    return image


def remove_small_line(doc):
    if len(doc) <= 3:
        return doc

    m1 = m2 = m3 = float('-inf')
    for x in doc:
        if x['box'][3] - x['box'][2] > m1:
            m2 = m1
            m1 = x['box'][3] - x['box'][2]
        elif x['box'][3] - x['box'][2] > m2:
            m3 = m2
            m2 = x['box'][3] - x['box'][2]
        elif x['box'][3] - x['box'][2] > m3:
            m3 = x['box'][3] - x['box'][2]

    return [x for x in doc if (x['box'][3] - x['box'][2]) * LINE_THRESHOLD_RATIO >= m3]

def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im

def pdf_to_list_ndarray(filename, data_bytes):
    doc = fitz.Document(stream=data_bytes, filetype='pdf')
    # doc = fitz.open(pdf_path)
    scale= 3.0
    mat = fitz.Matrix(scale, scale)

    output = []
    for index, page in enumerate(doc):
        pix = page.getPixmap(alpha=False, matrix=mat)
        image_page = pix2np(pix)

        output.append(image_page)

    return output


def create_shapely_polygon(polygon):
    try:
        obj_polygon = Polygon(polygon)
        if not obj_polygon.is_valid:
            obj_polygon = obj_polygon.buffer(0)
    except ValueError:
        # Use bb instead
        all_x = [each[0] for each in polygon]
        all_y = [each[1] for each in polygon]
        x1, y1, x2, y2 = min(all_x), min(all_y), max(all_x), max(all_y)
        org_bb = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        obj_polygon = Polygon(org_bb)
        if not obj_polygon.is_valid:
            obj_polygon = obj_polygon.buffer(0)
    return obj_polygon


def polygon_area(polygon):
    polygon = create_shapely_polygon(polygon)
    return polygon.area

def get_color(n):
    return '\x1b[3{0}m'.format(n)
class MultiLine(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%', newline=None):
        logging.Formatter.__init__(self, fmt, datefmt, style)
        self.newline = newline
        BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

        self.color = {
            logging.DEBUG: get_color(WHITE),
            logging.INFO: get_color(BLUE),
            logging.WARNING: get_color(YELLOW),
            logging.ERROR: get_color(RED),
            logging.CRITICAL: get_color(RED),
        }

    def format(self, record):
        result = super().format(record)
        if self.newline:
            result = result.replace("\n", "\n{0}".format(self.newline % (record.__dict__)))
        levelname = record.levelno
        if levelname in self.color:
            result = self.color[levelname] + result + '\x1b[39m'
        return result.strip()


def set_output_file_for_logger(logger, output_dir: str):
    fh = logging.FileHandler(output_dir)
    fh.setFormatter(
        MultiLine('[%(name)s]:[%(process)d] [%(asctime)s]-[%(levelname)s]-[%(filename)s:%(lineno)d]-%(message)s',
              "%Y-%m-%d %H:%M:%S %z",
              newline='[%(name)s]:[%(process)d] [%(asctime)s]-[%(levelname)s] '))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger, fh

def draw_ser_results(image,
                     ocr_results,
                     font_size=10):
    np.random.seed(2021)
    color = (np.random.permutation(range(255)),
             np.random.permutation(range(255)),
             np.random.permutation(range(255)))
    color_map = {
        idx: (color[0][idx], color[1][idx], color[2][idx])
        for idx in range(0, 255)
    }
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)

    font = ImageFont.load_default()
    for ocr_info in ocr_results.textlines:
        if ocr_info.label not in color_map:
            continue
        if ocr_info.class_name.startswith('key'):
            continue
        color = color_map[ocr_info.label]
        text = ocr_info.class_name
        x_coord = [item[0] for item in ocr_info.location.raw_polygon]
        y_coord = [item[1] for item in ocr_info.location.raw_polygon]
        bbox = [min(x_coord), min(y_coord), max(x_coord), max(y_coord)]
        draw_box_txt(bbox, text, draw, font, font_size, color)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


def draw_box_txt(bbox, text, draw, font, font_size, color):
    # draw ocr results outline
    bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
    draw.rectangle(bbox, fill=color)

    # draw ocr results
    start_y = max(0, bbox[0][1] - font_size)
    tw = font.getlength(text)
    draw.rectangle(
        [(bbox[0][0] + 1, start_y), (bbox[0][0] + tw + 1, start_y + font_size)],
        fill=(0, 0, 255))
    draw.text((bbox[0][0] + 1, start_y), text, fill=(255, 255, 255), font=font)

def encode_jpg_base64_image(image, encode_quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), encode_quality]
    output = base64.b64encode(cv2.imencode('.jpg', image, encode_param)[1]).decode()

    return output

def convert_numpy_array_return_api(data, level: int = 1):
    if level == 1 and isinstance(data, tuple):
        data = list(data)

        return tuple(convert_numpy_array_return_api(data, level + 1))
        
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if len(value.shape) == 3:
                    data[key] = encode_jpg_base64_image(value, 80)
                else:
                    data[key] = value.tolist()
            else:
                data[key] = convert_numpy_array_return_api(value, level+1)

    if isinstance(data, list):
        for idx, value in enumerate(data):
            if isinstance(value, np.ndarray):
                if len(value.shape) == 3:
                    data[idx] = encode_jpg_base64_image(value, 80)
                else:
                    data[idx] = value.tolist()
            else:
                data[idx] = convert_numpy_array_return_api(value, level+1)

    return data

def check_blur_image(image, score_threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score > score_threshold:
        return False
    else:
        return True