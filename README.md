<p align="center">
  <a href="https://github.com/docling-project/docling-sdg">
    <img loading="lazy" alt="Docling" src="https://github.com/docling-project/docling-sdg/raw/main/docs/assets/docling-sdg-pic.png" width="40%"/>
  </a>
</p>

# AI Tutor DG: Synthetic QA Generation 📚

[![Platforms](https://img.shields.io/badge/platform-macos%20|%20linux%20|%20windows-blue)](https://github.com/docling-project/docling-parse/)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://docs.pydantic.dev/latest/contributing/#badges)

Welcome to AI Tutor DG! This project empowers you to generate high-quality, synthetic question-answer pairs from your documents. These pairs are invaluable for various AI applications, such as evaluating language models or creating ground truth datasets for training.

## ✨ Features
AI Tutor DG is designed with flexibility and ease of use in mind, offering powerful capabilities for generating synthetic data:

- 🧬 **Diverse Content Generation:** Generate high-quality question-answer pairs, including both text and image-based questions, from various document formats.

- 📄 **Broad Document Support:** Seamlessly process content from multiple document types, including PDF, PNG, JPG, and JPEG files.

- 💻 **Intuitive Command-Line Interface (CLI):** Enjoy a straightforward and convenient command-line tool for quick data generation.

### 🔜 Future Enhancements
We're continuously working to improve AI Tutor DG. Here's a glimpse of what's coming soon:

- 📝 **Enhanced Prompt Optimization:** Integrate more sophisticated prompt optimization techniques for even better question and answer quality.

- 🖼️ **Improved Image Chunking:** Implement overlapping when chunking images to ensure more comprehensive content capture.

- 📈 **Advanced Exercise Detection:** Further refine the underlying models to significantly increase their ability to detect and understand exercises within documents.

## ⚙️ Installation

To get started with AI Tutor DG, follow these simple steps to set up your environment and install the necessary dependencies.

1. ⬇️ Clone the Repository:

Begin by cloning the project repository to your local machine:

```
git clone https://https://github.com/TranMinhThang-dev/AI_tutor_data_generation.git
cd AI_tutor_data_generation
```

2. 🌿 Switch to Development Branch:

Ensure you are on the dev branch to access the latest features and updates:

```
git checkout dev
```

3. 📦 Install Dependencies:

Install all required Python packages using pip. It's recommended to do this within a virtual environment.

```
pip install -r requirements.txt
```

4. 🚀 Pull and Run Models:

This project relies on Text Detection and OCR models. Execute the following scripts to pull and run them:

```
sh pull_model.sh
sh run_model.sh
```

## 🚀 Getting Started & Usage
AI Tutor DG allows you to automatically generate question-answer pairs from specified sections of your documents. These generated pairs can then be used to enhance or evaluate AI applications, such as training a language model or creating robust evaluation benchmarks.

### 💡 Example Usage

You can download sample data from this Google Drive link and place it in the root directory of the repository to follow along with the examples below.

#### 📄 Generating Data from a PDF (CLI)

To generate data from a PDF document using the command-line interface, specify the input file path, the starting and ending pages, and the --step-by-step flag if you require detailed solutions:

```
python main.py --input data/Chuyên\ đề\ SỐ\ PHỨC\ đầy\ đủ\ -\ Bùi\ Trần.pdf --start_page 14 --end_page 17 --step-by-step
```

#### 🐍 Generating Data via Python API

For more programmatic control, you can integrate AI Tutor DG directly into your Python scripts. The example below demonstrates how to process multiple PDFs:

```python
import glob
from main import MainDataGeneration

# Initialize the data generator with desired options
data_generator = MainDataGeneration(
    "data/eval.json", # Output file path (optional, can be overridden)
    require_step_by_step_solution=True,
    extract_final_answer=True
)

# Find all PDF files in a specified directory
pdfs = glob.glob("/mnt/ssd/jon/project/crawl_data/AI_tutor_data_generation/data/dethidaihoc2018/*.pdf")

# Process each PDF
for pdf in pdfs:
    data_generator.process_pdf(pdf)
```

By default, the generated question-answer pairs will be exported to a file named output.json in the root directory. Each line in this file represents a single question-answer pair.

**Note:** You must select start page and end page carefully, this flow work best for page that have sequence of exercise

<p align="center">
  <a href="https://github.com/TranMinhThang-dev/AI_tutor_data_generation/blob/dev/img/example_image.png">
    <img loading="lazy" alt="good_example" src="https://github.com/TranMinhThang-dev/AI_tutor_data_generation/blob/dev/img/example_image.png" width=40%/>
  </a>
</p>

## Get help and support

Please feel free to connect with us using the [discussion section](https://github.com/TranMinhThang-dev/AI_tutor_data_generation/discussions).

## Current flow(subject to change)

<p align="center">
  <a href="https://github.com/TranMinhThang-dev/AI_tutor_data_generation/blob/dev/img/architecture.png">
    <img loading="lazy" alt="Architecture" src="https://github.com/TranMinhThang-dev/AI_tutor_data_generation/blob/dev/img/architecture.png"/>
  </a>
</p>

### Techainer ❤️

The project was started by the AI for knowledge team at Techainer.
