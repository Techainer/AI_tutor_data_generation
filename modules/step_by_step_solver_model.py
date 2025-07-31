from .base import BaseLLMModule
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from .prompt import STEP_BY_STEP_SOLVE_PROMPT

class StepByStepSolver(BaseLLMModule):
    
    @property
    def parser(self) -> StrOutputParser:
        """Returns the output parser for step by step solution."""
        return OutputFixingParser.from_llm(self.llm, StrOutputParser()) 

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        """Returns the prompt template for step by step solution."""
        msg_template = HumanMessagePromptTemplate.from_template(
            template=[
                {"type": "text", "text": STEP_BY_STEP_SOLVE_PROMPT},
                {"type": "image_url", "image_url": "{encoded_image_url}"},
            ]
        )
        return ChatPromptTemplate(messages=[msg_template])
    
    @property
    def chain(self):
        """Returns the LangChain chain for step by step solution."""
        return self.prompt_template | self.llm | self.parser
    
    def process(self, input_str: str, input_image: bytes) -> str:
        """Returns a step by step solution."""
        try:
            encoded_image = self._process_image_input(input_image)
            llm_output = self.chain.invoke({"subject":"math","grade_level":"12","input": input_str, "encoded_image_url": encoded_image})
            return llm_output
        except Exception as e:
            from loguru import logger
            logger.error(f"Error {e} in line {e.__traceback__.tb_lineno}, code: {e.__traceback__.tb_frame.f_code.co_name}")
