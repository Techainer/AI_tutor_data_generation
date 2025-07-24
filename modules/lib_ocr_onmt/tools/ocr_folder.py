import glob
import json
import multiprocessing
import os

import cv2
from tqdm import tqdm

from ocr_onmt import Img2LatexModel

model = Img2LatexModel(
    weight_path='/Users/linus/techainer/opennmt-py/models/general_card_11/card_only_11-ocr_step_1820000.pt',
    multiprocesing=True
)
num_workers = 8
input_path = '/Users/linus/techainer/vietnamese-identity-card/prelabel/textlines/'
output_path = '/Users/linus/techainer/vietnamese-identity-card/prelabel/ocr.json'

all_samples = glob.glob(os.path.join(input_path, '*'))


def handle_single_sample(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    res = model.predict(img)
    res['path'] = os.path.basename(img_path)
    return res


pool = multiprocessing.Pool(num_workers)
output = list(tqdm(
    pool.imap(handle_single_sample, all_samples), total=len(all_samples), desc="Inferencing"))
pool.terminate()

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
