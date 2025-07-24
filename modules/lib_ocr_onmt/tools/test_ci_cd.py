import os
import argparse
import multiprocessing
import glob
import json
from tqdm import tqdm
import cv2
from mlchain.client import Client
from mlchain.workflows import Parallel, Task
from datetime import datetime
import xlsxwriter
import difflib
from tools.xlxs_utils import set_all_column_autowidth

from PIL import Image
import numpy as np

import logging

logger = logging.getLogger()

def insert_image_worksheet(worksheet, img_path, index, col_index, scale=3, positioning=2):
    img = np.array(Image.open(img_path))
    height = img.shape[0]
    worksheet.insert_image(scale*index-scale+1, col_index, img_path, {
        'x_offset': 2, 'y_offset': 2,
        # 'x_scale': 20.0/height, 'y_scale': 20.0/height,
        'x_scale': scale*20.0/height, 'y_scale': scale*20.0/height,
        'positioning': positioning
    })

def get_all_data(test_data_path):
    src_path = os.path.join(test_data_path, 'src.txt')
    tgt_path = os.path.join(test_data_path, 'tgt.txt')
    # src_path = os.path.join(test_data_path, 'src-val.txt')
    # tgt_path = os.path.join(test_data_path, 'tgt-val.txt')
    all_src = [each for each in open(src_path, 'r').read().split('\n')]
    all_tgt = [each.replace(' ', '').replace('\;', ' ') for each in open(tgt_path, 'r').read().split('\n')]
    all_labels = [(src, tgt) for src, tgt in zip(all_src, all_tgt) if src.strip() != ""]
    return all_labels


def diff_rows(s1, s2):
    diff = difflib.SequenceMatcher(None, s1, s2)
    result = ''
    format_pairs = ''
    if s2 == '' and len(s1) > 0:
        result = 'OCR no value field'
    else:
        for tag, i1, i2, j1, j2 in diff.get_opcodes():
            print_s1 = 'SPACE' if s1[i1:i2] == ' ' else s1[i1:i2]
            print_s2 = 'SPACE' if s2[j1:j2] == ' ' else s2[j1:j2]
            print_s1 = 'ENTER' if s1[i1:i2] == '\n' else print_s1
            print_s2 = 'ENTER' if s2[j1:j2] == '\n' else print_s2
            if tag == 'replace':
                result = result + \
                    'Mistake {} -> {}\n'.format(print_s2, print_s1)
                # format_pairs.append(('r', i1, i2))
                format_pairs = format_pairs + " ('r',{},{})".format(i1, i2)
            elif tag == 'delete':
                result = result + \
                    'Excess {}(position {})\n'.format(print_s1, str(i1))
                # format_pairs.append(('r', i1, i2))
                format_pairs = format_pairs + " ('r',{},{})".format(i1, i2)
            elif tag == 'insert':
                if len(s1) == 0:
                    result = result + 'Can not OCR'
                else:
                    result = result + \
                        'Lost {}(position {})\n'.format(print_s2, str(j1))
    match_n = 0
    for block in diff.get_matching_blocks():
        match_n = match_n + block[2]
    return match_n, result


def predict_single_sample(img, model, expected_result):
    result = model.predict(img)['prediction']
    match_n, explanation = diff_rows(expected_result, result)
    match_n_1, explanation_1 = diff_rows(result, expected_result)
    if explanation_1 != "OCR no value field":
        try:
            return result, (match_n/len(expected_result) + match_n_1/len(result))/2, explanation_1
        except ZeroDivisionError:
            return result, 1, explanation_1
    else:
        return result, 0, explanation_1

def prediction_task(args, model, file_name, gt_tgt):
    sample_path = os.path.join(args.test_dataset_path, 'images', file_name)
    sample_img = cv2.imread(sample_path)
    if sample_img is None:
        return None
    return {
        'res': predict_single_sample(sample_img, model, gt_tgt),
        'file_name': file_name,
        'gt_tgt': gt_tgt
    }

def test(args):
    model = Client(api_address=args.onmt_api_address, serializer='msgpack').model(check_status=True)
    os.makedirs(args.output_report_path, exist_ok=True)
    report_file_name = 'ocr_onmt_{}_{}.xlsx'.format(os.path.basename(args.test_dataset_path), datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))
    report_path = os.path.join(args.output_report_path, report_file_name)
    
    report_workbooks = xlsxwriter.Workbook(report_path)
    report_worksheets = report_workbooks.add_worksheet("AccuracyReport")
    column_name_format = report_workbooks.add_format({'bold': True, 'font_size': 14})
    cell_format = report_workbooks.add_format()
    cell_format.set_font_size(14)
    report_worksheets.write(0, 0, 'Index', column_name_format)
    report_worksheets.write(0, 1, 'File Name', column_name_format)
    report_worksheets.write(0, 2, 'Input', column_name_format)
    report_worksheets.write(0, 3, 'Predicted', column_name_format)
    report_worksheets.write(0, 4, 'Ground Truth', column_name_format)
    report_worksheets.write(0, 5, 'Accuracy by char', column_name_format)
    report_worksheets.write(0, 6, 'Explanation', column_name_format)
    
    scale = 2
    index = 1
    all_acc_by_char = []
    total_correct_by_field = 0

    all_labels = get_all_data(args.test_dataset_path)
    all_tasks = [Task(prediction_task, args, model, file_name, gt_tgt) for (file_name, gt_tgt) in all_labels]
    all_predictions = Parallel(all_tasks, max_threads=8, pass_fail_job=True).run(progress_bar=True)
    all_predictions = [each for each in all_predictions if each is not None]
    for each_sample in all_predictions:
        output, acc_by_char, explanation = each_sample['res']
        file_name = each_sample['file_name']
        gt_tgt = each_sample['gt_tgt']
        all_acc_by_char.append(acc_by_char)
        curr_index = scale*index-scale+1
        if acc_by_char == 1:
            total_correct_by_field += 1
        report_worksheets.write(curr_index, 5, acc_by_char, cell_format)
        report_worksheets.write(curr_index, 0, index, cell_format)
        report_worksheets.write(curr_index, 1, file_name, cell_format)
        # insert_image_worksheet(report_worksheets, sample_path, index, 2, scale)
        report_worksheets.write(curr_index, 3, output, cell_format)
        report_worksheets.write(curr_index, 4, gt_tgt, cell_format)
        report_worksheets.write(curr_index, 6, explanation, cell_format)
        index += 1

    avg_acc_by_char = np.mean(all_acc_by_char)
    acc_by_field = total_correct_by_field / len(all_labels)
    report_worksheets.write(scale * index - scale + 1, 4, "Average Accuracy by char: ", column_name_format)
    report_worksheets.write(scale * index - scale + 1, 5, avg_acc_by_char, cell_format)
    index += 1
    report_worksheets.write(scale * index - scale + 1, 4, "Accuracy by field: ", column_name_format)
    report_worksheets.write(scale * index - scale + 1, 5, acc_by_field, cell_format)
    set_all_column_autowidth(report_worksheets)
    report_workbooks.close()
    logger.info('Exproted excel report to {}'.format(report_path))

    if avg_acc_by_char < args.acc_by_char_threshold:
        logger.warning('Average accuracy by char: {}. Smaller than {} threshold ==> Test FAILED!'.format(avg_acc_by_char, args.acc_by_char_threshold))
    else:
        logger.info('Average accuracy by char: {}. Larger than {} threshold ==> Test PASSED!'.format(avg_acc_by_char, args.acc_by_char_threshold))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Auto test, calculate OCR ONMT metric and export report')
    parser.add_argument('--onmt_api_address', default='0.0.0.0:8002',
                        type=str, required=False, help='API address of OCR ONMT services')
    parser.add_argument('--test_dataset_path', default='/Users/linus/techainer/opennmt-py/data/myanmar/dataset_myanmar_ocr',
                        type=str, required=False, help='Path to ONMT test dataset')
    parser.add_argument('--output_report_path', default='./results/',
                        type=str, required=False, help='Path to save excel report')
    parser.add_argument('--acc_by_char_threshold', default=0.7,
                        type=float, required=False, help='Path to save excel report')
    global args
    args = parser.parse_args()

    if not os.path.exists(args.test_dataset_path):
        logger.error(args.test_dataset_path, 'doses not exist!')
    else:
        test(args)
