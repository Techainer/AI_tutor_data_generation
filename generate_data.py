from main import MainDataGeneration

D = ["/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/Chuyên đề SỐ PHỨC đầy đủ - Bùi Trần.pdf",
     "/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/DẠNG-2-CẤP-SỐ-CỘNG-CẤP-SỐ-NHÂN.pdf",
     "/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/DẠNG-22-XÁC-ĐỊNH-HÌNH-CHIẾU-CỦA-ĐIỂM-LÊN-MẶT-PHẲNG.pdf",
     "/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/DS_C3_UNG DUNG TICH PHAN.pdf",
     "/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/khám phá tư duy chứng minh bất đẳng thức.pdf",
     "/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/TOAN CANH DE MINH HOA - DE CHINH THUC 2017-2018-2019-BGD.pdf"]

for pdf_path in D:
    print(f"Processing {pdf_path}")
    data_generator = MainDataGeneration("data.json", require_step_by_step_solution=True)
    data_generator.process_pdf(pdf_path, start_page=0, end_page=-1)
