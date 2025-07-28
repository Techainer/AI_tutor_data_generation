import os 
import io
import cv2
import copy
import json
from typing import Union, List
from dotenv import load_dotenv
from PIL import Image
import numpy as np
from tqdm import tqdm
from modules.exercise_extractor import ExerciseExtractor
from modules.step_by_step_solver_model import StepByStepSolver
from modules.crop_exercise_db_model import CropExerciseDBModel
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
                api_key=google_api_key,
                llm_model="gemini-2.5-flash",
                provider="google",
                llm_host=self_host_api_url
            )
        else:
            self.step_by_step_solver = None
            
        self.crop_exercise_db_model = CropExerciseDBModel(debug=debug)

        self.standard_height = 720 # height
        if debug:
            os.makedirs("debug",exist_ok=True)
    
    def process_image(self, input_image: np.ndarray, page_idx: int, padding_exercise: int=0, padding_pixel_top: int=5, padding_pixel_bot: int=5) -> None:
        """
            Describle: Run the exercise extraction process.
            Args:
                input_image: Image of page.
                page_idx: index of page.
                padding_exercise: number of padding exercise to bot and top.
                padding_pixel_top: number of padding pixel in top.
                padding_pixel_bot: number of padding pixel in bot.
            Returns:
                _type_: None
        """
        logger.info("Starting exercise extraction in image...")
        os.makedirs(f"output_image/page_{page_idx}", exist_ok=True)
        if debug:
            os.makedirs(f"debug/page_{page_idx}", exist_ok=True)
        try: 
            w = input_image.shape[1]
            if debug:
                debug_image = input_image.copy()
            split_y_coordinates = self.crop_exercise_db_model.process(input_image, page_idx=page_idx)
            if len(split_y_coordinates) == 0:
                logger.info("No exercise found, return without calling LLM")
                return
            for idx, split_y_coordinate in enumerate(split_y_coordinates):

                if debug:
                    debug_image = cv2.line(debug_image, (0, split_y_coordinate), (w, split_y_coordinate), color=(0,255,0), thickness=2)
                    cv2.imwrite(f"debug/page_{page_idx}/debug_box_image.png", debug_image)
                    logger.info(f"Save image to debug/page_{page_idx}/debug_box_image.png")
                
                if idx == len(split_y_coordinates)-1:
                    cropped_exercise = input_image[split_y_coordinate-padding_pixel_top: , 0: w]
                else:
                    y2 = split_y_coordinates[idx+1]
                    cropped_exercise = input_image[split_y_coordinate-padding_pixel_top: y2+padding_pixel_bot , 0: w]
                    
                viz_image = cropped_exercise.copy()
                cv2.imwrite(f"output_image/page_{page_idx}/exercise_image_{idx}.png", viz_image)
                logger.success(f"Save exercise image to output_image/page_{page_idx}/exercise_image_{idx}.png", viz_image)
                
                if debug:
                    cv2.imwrite(f"debug/page_{page_idx}/exercise_crop_image{idx}.png", viz_image)
                    logger.info(f"Save image to debug/page_{page_idx}/exercise_crop_image{idx}.png")
                    
                buffer = io.BytesIO()
                cropped_exercise = Image.fromarray(cropped_exercise)
                cropped_exercise.save(buffer, format="JPEG", quality=95)
                cropped_exercise = buffer.getvalue()
                
                exercise_list = self.exercise_extractor.process(cropped_exercise)
                
                if self.require_step_by_step_solution:
                    logger.info("Starting give step by step solution for each exercise...")
                    for exercise in exercise_list.exercise_list:
                        solution = self.step_by_step_solver.process(exercise.question)
                        exercise.answer = solution

                for exercise in exercise_list.exercise_list:
                    D = exercise.model_dump()
                    D['image'] = f"output_image/page_{page_idx}/exercise_image_{idx}.png"
                    with open(self.save_path, 'a', encoding='utf-8') as f:
                        f.write(f"{json.dumps(D, ensure_ascii=False)},\n")

                logger.success("Extracted exercises successfully.")
        except Exception as e:
            logger.error(f"Error during exercise extraction: {e} in line {e.__traceback__.tb_lineno}, code: {e.__traceback__.tb_frame.f_code.co_name}")
    
    def process_pdf(self, pdf_path: str, start_page: int = 0, end_page: int = -1) -> None:
        """Process a PDF file and extract exercises from each page."""
        logger.info("Starting exercise extraction in PDF...")

        try:
            images = pdf_pages_to_images(pdf_path)
            for page_idx, image in tqdm(enumerate(images[start_page: end_page]), desc="Processing PDF pages:"):
                image = np.array(Image.open(io.BytesIO(image)))
                self.process_image(image, page_idx=page_idx)
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
        