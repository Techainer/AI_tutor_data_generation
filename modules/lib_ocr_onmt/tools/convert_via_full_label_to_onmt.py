import os
import argparse
import multiprocessing
import glob
import json
from tqdm import tqdm
import cv2
from utils.crop_utils import cut_polygon
import logging

logger = logging.getLogger()

def convert(args):
    # Read VIA json file
    via_raw = json.loads(open(args.via_json_file, 'r').read())
    
    # Create output folder
    os.makedirs(args.output_path, exist_ok=True)
    project_name = '{}_ocr_onmt'.format(via_raw['_via_settings']['project']['name'])
    output_path = os.path.join(args.output_path, project_name)
    os.makedirs(output_path, exist_ok=True)
    output_images_path = os.path.join(output_path, 'images')
    os.makedirs(output_images_path, exist_ok=True)

    # Multi process all samples
    onmt_output = []
    all_samples = via_raw["_via_img_metadata"]
    for sample in tqdm(all_samples.values()):
        sample_path = os.path.join(args.images_path, sample['filename'])
        img = cv2.imread(sample_path)
        if img is None:
            return None
        all_regions = sample['regions']
        for region_index, region in enumerate(all_regions):
            if ('text' in region['region_attributes'].keys() and region['region_attributes']['text'].strip() == ''):
                continue
            if region['region_attributes']['key'] in ['page', 'logo', 'profile_image', 'qr_code', 'van_tay']:
                continue
            region_orientation = int(region['region_attributes']['orientation'])
            region_file_name = '{}_{}.png'.format(sample['filename'], region_index)
            region_file_path = os.path.join(output_images_path, region_file_name)
            if region['shape_attributes']['name'] == 'polygon':
                try:
                    all_x = region['shape_attributes']['all_points_x']
                    all_y = region['shape_attributes']['all_points_y']
                    region_polygon = [(x, y) for x, y in zip(all_x, all_y)]
                    region_image = cut_polygon(img.copy(), region_polygon, text_line=True, additional_orientation=region_orientation)
                    cv2.imwrite(region_file_path, region_image)
                    onmt_output.append((region_file_name, region['region_attributes']['text']))
                except Exception as ex:
                    logger.error(ex)
                    continue
            elif region['shape_attributes']['name'] == 'rect':
                try:
                    x = region['shape_attributes']['x']
                    y = region['shape_attributes']['y']
                    w = region['shape_attributes']['width']
                    h = region['shape_attributes']['height']
                    region_polygon = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                    region_image = cut_polygon(img.copy(), region_polygon, text_line=True, additional_orientation=region_orientation)
                    cv2.imwrite(region_file_path, region_image)
                    onmt_output.append((region_file_name, region['region_attributes']['text']))
                except Exception as ex:
                    logger.error(ex)
                    continue

    # Save as ONMT
    all_src = [e[0] for e in onmt_output]
    all_tgt = [e[1] for e in onmt_output]
    with open(os.path.join(output_path, 'src.txt'), 'w') as f:
        for each in all_src:
            f.write('{}\n'.format(each))

    with open(os.path.join(output_path, 'tgt.txt'), 'w') as f:
        for text in all_tgt:
            word = [' '.join(list(word)) for word in text.split()]
            text = ' \; '.join(word)
            f.write('{}\n'.format(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert VIA Full Label data to ONMT OCR only format')
    parser.add_argument('--via_json_file', default='/Users/linus/Downloads/demo_seabank_full_label_2.json',
                        type=str, required=False, help='Path to exported VIA json file')
    parser.add_argument('--images_path', default='/Users/linus/techainer/real_data/demo_seabank_full_label', type=str, required=False, help='Path to folder contain all images of the VIA project')
    parser.add_argument('--output_path', default='./result/', type=str, required=False, help='Path to folder to save exported ONMT dataset')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='Numbers of worker for multiprocessing')
    global args
    args = parser.parse_args()
    
    if not os.path.exists(args.images_path):
        logger.error(args.images_path, 'doses not exist!')
    else:
        convert(args)
