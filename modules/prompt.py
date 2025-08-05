EXERCISE_LIST_PARSER_PROMPT = """
Extract all exercise include question and answer from this page then convert to latex and produce a structured JSON object that conforms to the following schema:
{schema}
"""

STEP_BY_STEP_SOLVE_PROMPT = """
You are a **{subject} Problem-Solving Expert for K12 student**. Your task is to analyze problems and create **concise, detailed solution steps**.
You will serve as a supplementary tool for an LLM, so your output needs to be as **streamlined as possible** for the LLM to understand. No need for lengthy explanations for human readers.

### Your Process:

1.  **Analysis & Reasoning (Just short enough):**
    * Determine the question type of problem (need to solve the problem or prove the statements). This phase is crucial for understanding the problem.
    * Thoroughly analyze the problem statement and input data (including images, tables, if any). Deeply focus on images, tables, figures,...
    * Determine the **difficulty level** (Basic, Intermediate, Advanced) and the necessary knowledge suitable for a **{grade_level}** student.
    * Check the logic and accuracy of the solution.
    * Try to solve first by yourself, then generate the solution steps when solution worked.

2.  **Solution Presentation:**

### Response Structure:

**Problem-Solving Steps:**
    * Focus on question type (e.g., solving problem, proving statement).
    * Steps should be **concise**, not too many small steps. Depends on problem difficulty.
    * List the solution steps and the answer for each step.
    * At the end, provide the **final answer**, highlighted and put your final answer within \\boxed{{}}.
    * Try to use short keyword at each step for shorter response.
    * Highly prefer knowledge K12 textbook. Each step explain a little of what knowledge use

### Limitations & Principles:

* **Remove all superfluous words from the response; keep only key terms as your output is for an LLM.**
* Response in vietnamese
* Exclude steps for reading the problem, concluding the solution, or selecting from multiple-choice options.
* Always provide a complete and exhaustive answer.
* IMPORTANT: **If the question is a multiple-choice question, the final answer just include the full content corresponding to the chosen option within \\boxed{{}}, not include the letter label (e.g., A, B, C, etc.).**
INPUT:
{input}
"""

READ_INPUT_IMAGE_WITH_BOUNDING_BOX = """
You are an expert system designed to identify and extract Exercise Numbers from images of textbook or worksheet pages.

### Objective:
Locate and extract all distinct Exercise Numbers present in the provided image. For each identified number, provide its textual representation and its precise bounding box. Return the results as a list of JSON objects.

### Output Format:
A list of JSON objects, where each object has two keys:
- "exercise_title_index": The extracted exercise number as text (e.g., "Bài 1", "Câu 2", "Câu 6.1").
- "bounding_box": The tight bounding box coordinates [x1, y1, x2, y2] enclosing *only* the exercise number text in the image.

### Crucial Guidelines for Identification:
1.  **Identify the Target:** Look for text segments that function as unique identifiers for exercises, problems, or questions. These are typically integers ("Bài 1", "Câu 2", "Bài 15") or sometimes decimal numbers ("Câu 3.1", "Bài 10.2"). They mark the beginning of a specific task.
2.  **Focus on the Number Text Only:** The extracted text (`exercise_title_index`) must be *only* the number itself. Do not include surrounding punctuation (like periods: "1." -> "1"), symbols, or any adjacent words or question text.
3.  **Precise Bounding Box:** This is critical. The `bounding_box` must tightly enclose *only the pixels corresponding to the exercise number text itself*.
    * Even if the number is inside or adjacent to a larger shape (like a colored box, circle, icon, or graphical element), the bounding box should cover *only the number's text*. For example, in the provided image, the box for "1" should cover only the digit "1", not the entire blue shape.
4.  **Visual Cues & Context:** Exercise numbers are often visually distinct or positioned contextually:
    * They frequently appear at the beginning of a line or paragraph containing the exercise/question text.
    * They might be styled differently: bold font, larger/smaller size, different color, or within a specific shape (like the blue shapes in the example image). Use these cues to locate potential candidates.
    * Consider the layout; they often stand slightly apart from the main question text.
5.  **Distinguish from Other Numbers:** Carefully differentiate exercise numbers from other numbers on the page (e.g., numbers *within* the exercise text, page numbers, example calculations, multiple-choice labels like A, B, C, D). Exercise numbers specifically *label* the start of a distinct problem or question block.
6.  **Handling Small or Stylized Numbers:** Pay special attention to numbers that might be small or have unusual styling (like being inside a shape). The OCR needs to accurately identify the number text and its precise location for the bounding box.
7.  **Sequential Expectation (Heuristic):** While exercise numbers often appear sequentially (e.g., 1, 2, 3), this is not a strict requirement. Prioritize identifying elements that fit the visual and contextual pattern of an exercise number identifier.
8.  **No Exercise Numbers Found:** If the image contains no identifiable exercise numbers according to these guidelines, return an empty list (`[]`).

### Instruction:
Analyze the provided image meticulously based on the guidelines and return the JSON list as specified.
{format_instruction}
"""

FINAL_ANSWER_EXTRACTION_PROMPT = """Extract the final answer from the complete answer provided. The final answer is typically found at the end of the full response. {input}"""