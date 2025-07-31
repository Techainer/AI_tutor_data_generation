from main import MainDataGeneration
import os
import glob

data_generator = MainDataGeneration("data/data.json", require_step_by_step_solution=True)
# data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/cac-dang-toan-gtln-gtnn-cua-ham-so-thuong-gap-trong-ky-thi-thptqg.pdf", start_page=0, end_page=19)
# data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/chuyen-de-trac-nghiem-tinh-don-dieu-cua-ham-so.pdf", start_page=69, end_page=101)
# data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/chuyen-de-cac-so-dac-trung-do-muc-do-phan-tan-cho-mau-so-lieu-ghep-nhom-toan-12.pdf", start_page=35, end_page=46)
# data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/su-dong-bien-va-nghich-bien-cua-ham-so-le-hai-trung.pdf", start_page=13, end_page=-1)

data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/tai-lieu-chuyen-de-duong-tiem-can-cua-do-thi-ham-so.pdf", start_page=0, end_page=13)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/tai-lieu-chuyen-de-duong-tiem-can-cua-do-thi-ham-so.pdf", start_page=32, end_page=37)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/tai-lieu-chuyen-de-duong-tiem-can-cua-do-thi-ham-so.pdf", start_page=52, end_page=61)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/tai-lieu-chuyen-de-duong-tiem-can-cua-do-thi-ham-so.pdf", start_page=90, end_page=96)

data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/toan-thuc-te-nguyen-ham-va-tich-phan-toan-12.pdf", start_page=1, end_page=8)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/toan-thuc-te-nguyen-ham-va-tich-phan-toan-12.pdf", start_page=32, end_page=46)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/toan-thuc-te-nguyen-ham-va-tich-phan-toan-12.pdf", start_page=83, end_page=91)

data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/toan-thuc-te-phuong-phap-toa-do-trong-khong-gian-toan-12.pdf", start_page=3, end_page=6)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/toan-thuc-te-phuong-phap-toa-do-trong-khong-gian-toan-12.pdf", start_page=19, end_page=23)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/toan-thuc-te-phuong-phap-toa-do-trong-khong-gian-toan-12.pdf", start_page=39, end_page=53)

data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/Ungdungdaoham.pdf", start_page=0, end_page=6)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/Ungdungdaoham.pdf", start_page=22, end_page=28)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/Ungdungdaoham.pdf", start_page=44, end_page=48)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/Ungdungdaoham.pdf", start_page=58, end_page=63)
data_generator.process_pdf("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/tailieu/Ungdungdaoham.pdf", start_page=73, end_page=77)

