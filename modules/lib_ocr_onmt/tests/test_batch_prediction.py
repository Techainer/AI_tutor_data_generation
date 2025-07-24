import difflib
import glob
import json
import logging
import os
import unittest
from datetime import datetime

import cv2
import numpy as np
import xlsxwriter
from mlchain import mlconfig
from ocr_onmt import load_onmt_model
from PIL import Image
from tqdm import tqdm
from tools.xlxs_utils import set_all_column_autowidth

mlconfig.load_config('mlconfig.yaml')

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
    all_src = [each for each in open(src_path, 'r').read().split('\n')]
    all_tgt = [each.replace(' ', '').replace('\;', ' ')
               for each in open(tgt_path, 'r').read().split('\n')]
    all_labels = [(src, tgt)
                  for src, tgt in zip(all_src, all_tgt) if src.strip() != ""]
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


def evaluate_single_sample(predicted_res, expected_result):
    result = predicted_res["prediction"]
    match_n, explanation = diff_rows(expected_result, result)
    match_n_1, explanation_1 = diff_rows(result, expected_result)
    if explanation_1 != "OCR no value field":
        try:
            acc_by_char = (match_n/len(expected_result) + match_n_1/len(result))/2
            # 
            return result, acc_by_char, explanation_1
        except ZeroDivisionError:
            if expected_result == "":
                return result, 1, explanation_1
            return result, 0, explanation_1
    else:
        return result, 0, explanation_1


class TestAndReportOCRONMT(unittest.TestCase):
    def __init__(self,
                 *args,
                 weights_path=mlconfig.weight,
                 test_dataset_path='tests/idcard_ocr_testset',
                 output_report_path='tests/results',
                 acc_by_char_threshold=0.7,
                 **kwargs):
        self.acc_by_char_threshold = acc_by_char_threshold
        self.test_dataset_path = test_dataset_path
        self.model = load_onmt_model(flow_mode="legacy", vocab_path=mlconfig.ocr_vocab_path, device="cuda")
        os.makedirs(output_report_path, exist_ok=True)
        hardware_type = os.getenv("HARDWARE_TYPE", "")
        report_file_name = '{}_ocr_onmt_{}_{}.xlsx'.format(hardware_type, os.path.basename(
            test_dataset_path), datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.report_path = os.path.join(output_report_path, report_file_name)
        self.report_workbooks = xlsxwriter.Workbook(self.report_path)
        self.report_worksheets = self.report_workbooks.add_worksheet(
            "AccuracyReport")
        self.column_name_format = self.report_workbooks.add_format(
            {'bold': True, 'font_size': 14})
        self.cell_format = self.report_workbooks.add_format()
        self.cell_format.set_font_size(14)
        self.report_worksheets.write(0, 0, 'Index', self.column_name_format)
        self.report_worksheets.write(
            0, 1, 'File Name', self.column_name_format)
        self.report_worksheets.write(0, 2, 'Input', self.column_name_format)
        self.report_worksheets.write(
            0, 3, 'Predicted', self.column_name_format)
        self.report_worksheets.write(
            0, 4, 'Ground Truth', self.column_name_format)
        self.report_worksheets.write(
            0, 5, 'Accuracy by char', self.column_name_format)
        self.report_worksheets.write(
            0, 6, 'Explanation', self.column_name_format)
        self.all_labels = get_all_data(test_dataset_path)
        unittest.TestCase.__init__(self, *args, **kwargs)

    def test_and_generate_report(self):
        scale = 2
        index = 1
        all_acc_by_char = []
        total_correct_by_field = 0

        all_samples = []
        all_tgt = []
        for sample_index, (file_name, gt_tgt) in tqdm(enumerate(self.all_labels), total=len(self.all_labels)):
            sample_path = os.path.join(
                self.test_dataset_path, 'images', file_name)
            sample_img = cv2.imread(sample_path)
            if sample_img is None:
                continue
            all_samples.append(sample_img)
            all_tgt.append(gt_tgt)
        all_predictions = self.model.predict_batch(images=all_samples, ratio_diff=1)

        for sample_index, (predicted_res, gt_tgt) in tqdm(enumerate(zip(all_predictions, all_tgt)), total=len(all_predictions)):
            with self.subTest(i=os.path.basename(self.all_labels[sample_index][0])):
                # 
                sample_path = os.path.join(
                    self.test_dataset_path, 'images', self.all_labels[sample_index][0])
                output, acc_by_char, explanation = evaluate_single_sample(predicted_res, gt_tgt)
                all_acc_by_char.append(acc_by_char)
                curr_index = scale*index-scale+1
                if acc_by_char == 1:
                    total_correct_by_field += 1
                self.report_worksheets.write(
                    curr_index, 5, acc_by_char, self.cell_format)
                self.report_worksheets.write(
                    curr_index, 0, index, self.cell_format)
                self.report_worksheets.write(
                    curr_index, 1, file_name, self.cell_format)
                insert_image_worksheet(
                    self.report_worksheets, sample_path, index, 2, scale)
                self.report_worksheets.write(
                    curr_index, 3, output, self.cell_format)
                self.report_worksheets.write(
                    curr_index, 4, gt_tgt, self.cell_format)
                self.report_worksheets.write(
                    curr_index, 6, explanation, self.cell_format)
                index += 1

        avg_acc_by_char = np.mean(all_acc_by_char)
        acc_by_field = total_correct_by_field / len(self.all_labels)
        self.report_worksheets.write(scale * index - scale + 1, 4,
                                     "Average Accuracy by char: ", self.column_name_format)
        self.report_worksheets.write(scale * index - scale + 1,
                                     5, avg_acc_by_char, self.cell_format)
        index += 1
        self.report_worksheets.write(scale * index - scale + 1,
                                     4, "Accuracy by field: ", self.column_name_format)
        self.report_worksheets.write(scale * index - scale + 1,
                                     5, acc_by_field, self.cell_format)
        set_all_column_autowidth(self.report_worksheets)
        self.report_workbooks.close()
        logger.info('Exported excel report to {}'.format(self.report_path))

        if avg_acc_by_char < self.acc_by_char_threshold:
            raise Exception('Average accuracy by char: {}. Smaller than {} threshold ==> Test FAILED!'.format(
                avg_acc_by_char, self.acc_by_char_threshold))
        else:
            logger.info('Average accuracy by char: {}. Larger than {} threshold ==> Test PASSED!'.format(
                avg_acc_by_char, self.acc_by_char_threshold))

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestAndReportOCRONMT('test_and_generate_report'))
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    if result.wasSuccessful():
        exit(0)
    else:
        exit(1)
