import os 
import io
import cv2
import copy
from typing import Union, List
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from tqdm import tqdm
from modules.exercise_extractor import ExerciseExtractor
from modules.step_by_step_solver_model import StepByStepSolver
from modules.crop_exercise_model import CropExerciseModel
from modules.utils import pdf_pages_to_images
from loguru import logger
load_dotenv()

debug = True
class MainDataGeneration:
    def __init__(self, save_path: str = "output.json", require_step_by_step_solution: bool = False):
        openai_api_key = os.getenv("LITELLM_API_KEY")
        openai_api_url = os.getenv("LITELLM_API_URL")
        google_api_key = os.getenv("GOOGLE_API_KEY")
        self_host_api_key = os.getenv("SELF_HOST_API_KEY")
        self_host_api_url = os.getenv("SELF_HOST_API_URL")
        
        self.exercise_extractor = ExerciseExtractor(
            api_key=openai_api_key,
            llm_model="gpt-4o",
            provider="openai",
            llm_host=openai_api_url
        )
        self.save_path = save_path
        self.require_step_by_step_solution = require_step_by_step_solution
        if self.require_step_by_step_solution:
            self.step_by_step_solver = StepByStepSolver(
                api_key=openai_api_key,
                llm_model="gpt-4o",
                provider="openai",
                llm_host=openai_api_url
            )
        else:
            self.step_by_step_solver = None
            
        self.crop_exercise_model = CropExerciseModel(
                # api_key=self_host_api_key,
                # llm_model="qwen2.5-vl-7b",
                # provider="openai",
                # llm_host=self_host_api_url,
                api_key=openai_api_key,
                llm_model="gpt-4o",
                provider="openai",
                llm_host=openai_api_url
                )

        self.standard_height = 720 # height
        if debug:
            os.makedirs("debug",exist_ok=True)
        
    def chunking_image(self, input_image: bytes, num_chunk: int=2) -> List[bytes]:
        """Split image into smaller patch to reduce (words/area) rate """
        image = np.array(Image.open(io.BytesIO(input_image)))
        
        height, width = image.shape[:2]
        chunk_height = int(height / num_chunk)
        
        chunks = []
        for i in range(num_chunk):
            if i == num_chunk-1:
                chunk = image[i*chunk_height: , 0: width]
                ratio = self.standard_height / (height - i*chunk_height)
            else:
                chunk = image[i*chunk_height: (i+1)*chunk_height, 0: width]
                ratio = self.standard_height / chunk_height
                
            chunk = cv2.resize(chunk, None, fx=ratio, fy=ratio)
            chunk = Image.fromarray(chunk)
            buffer = io.BytesIO()
            chunk.save(buffer, format="JPEG", quality=95)
            chunk = buffer.getvalue()
            chunks.append(chunk)
        
        return chunks
            
    def process_image(self, input_image: Union[str, bytes, np.ndarray], page_idx: int, chunk_idx: int) -> None:
        """Run the exercise extraction process."""
        logger.info("Starting exercise extraction in image...")
        if debug:
            os.makedirs(f"debug/page_{page_idx}/chunk_{chunk_idx}", exist_ok=True)
        try: 
            if isinstance(input_image, str):
                # If input is a file path, convert to bytes
                input_image = Image.open(input_image)
                buffer = io.BytesIO()
                input_image.save(buffer, format="JPEG", quality=95)
                input_image = buffer.getvalue()
                
            elif isinstance(input, np.ndarray):
                # If input is a numpy image, convert to bytes
                input_image = Image.fromarray(input_image)
                input_image.save(buffer, format="JPEG", quality=95)
                input_image = buffer.getvalue()
                
            cropped_exercises = self.crop_image(input_image, page_idx=page_idx, chunk_idx=chunk_idx)
            for idx, cropped_exercise in enumerate(cropped_exercises):
                # Extract exercise list exercise from the image
                if debug:
                    debug_image = Image.open(io.BytesIO(cropped_exercise))
                    debug_image.save(f"debug/page_{page_idx}/chunk_{chunk_idx}/exercise_title_image_{idx}.png")
                    logger.info(f"Save image to debug/page_{page_idx}/chunk_{chunk_idx}/exercise_title_image_{idx}.png")
                    
                exercise_list = self.exercise_extractor.process(cropped_exercise)
                
                if self.require_step_by_step_solution:
                    logger.info("Starting give step by step solution for each exercise...")
                    for exercise in exercise_list.exercise_list:
                        solution = self.step_by_step_solver.process(exercise.question)
                        exercise.answer = solution

                for exercise in exercise_list.exercise_list:
                    with open(self.save_path, 'a', encoding='utf-8') as f:
                        f.write(f"{exercise.model_dump_json()},\n")

                logger.success("Extracted exercises successfully.")
        except Exception as e:
            logger.error(f"Error during exercise extraction: {e} in line {e.__traceback__.tb_lineno}, code: {e.__traceback__.tb_frame.f_code.co_name}")

    def crop_image(self, input_image: bytes, page_idx: int, chunk_idx: int) -> List[bytes]:
        """Crop all exercise in image"""
        logger.info("Start crop exercise from image")
        if debug:
            debug_image = Image.open(io.BytesIO(input_image))
            debug_image.save(f"debug/page_{page_idx}/chunk_{chunk_idx}/input_crop_image.png")
        exercise_title_bb_list = self.crop_exercise_model.process(input_image)
        image = np.array(Image.open(io.BytesIO(input_image)))
        w = image.shape[1]
        cropped_images = []
        
        for idx in range(len(exercise_title_bb_list.exercise_title_list)):
            x_min1,y_min1,x_max1,y_max1 = exercise_title_bb_list.exercise_title_list[idx].bounding_box
            
            if idx == len(exercise_title_bb_list.exercise_title_list)-1:
                cropped_image = image[y_min1: ,0: w]
            else:
                x_min2,y_min2,x_max2,y_max2 = exercise_title_bb_list.exercise_title_list[idx+1].bounding_box
                cropped_image = image[y_min1 - 3: y_min2 + 3, 0: w]
                
            buffer = io.BytesIO()
            cropped_image = Image.fromarray(cropped_image)
            if cropped_image.mode == 'RGB':
                cropped_image.save(buffer, format="JPEG", quality=95)
            else:
                cropped_image.save(buffer, format="PNG", quality=95)
            cropped_image = buffer.getvalue()
            cropped_images.append(cropped_image)
            
        return cropped_images

    def process_pdf(self, pdf_path: str, start_page: int = 0, end_page: int = -1) -> None:
        """Process a PDF file and extract exercises from each page."""
        logger.info("Starting exercise extraction in PDF...")

        try:
            images = pdf_pages_to_images(pdf_path)
            for page_idx, image in tqdm(enumerate(images[start_page: end_page]), desc="Processing PDF pages:"):
                image_chunks = self.chunking_image(image)
                for chunk_idx,image_chunk in enumerate(image_chunks):
                    self.process_image(image_chunk, page_idx=page_idx, chunk_idx=chunk_idx)
            logger.success("Extracted exercises successfully.")
        except Exception as e:
            logger.error(f"Error processing PDF: {e} in line {e.__traceback__.tb_lineno}, code: {e.__traceback__.tb_frame.f_code.co_name}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract exercises from images or PDFs.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image or PDF file.")
    parser.add_argument("--output", type=str, default="output.json", help="Path to save the extracted exercises.")
    parser.add_argument("--step-by-step", action="store_true", help="Whether to require step by step solution for each exercise.")
    parser.add_argument("--start_page", type=int, default=1, help="Start page for PDF processing (default: 0).")
    parser.add_argument("--end_page", type=int, default=None, help="End page for PDF processing (default: -1).")

    args = parser.parse_args()

    start_page, end_page = args.start_page, args.end_page  
    data_generator = MainDataGeneration(save_path=args.output, require_step_by_step_solution=args.step_by_step)

    if args.input.lower().endswith('.pdf'):
        data_generator.process_pdf(args.input, start_page, end_page)
        logger.info(f"Processed pdf: {args.input}")
    else:
        data_generator.process_image(args.input)
        logger.info(f"Processed image: {args.input}")

    logger.info(f"Output saved to: {args.output}")
        