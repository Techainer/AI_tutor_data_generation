from .base import BaseLLMModule
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from .prompt import FINAL_ANSWER_EXTRACTION_PROMPT

class FinalAnswerExtractor(BaseLLMModule):
    
    @property
    def parser(self) -> StrOutputParser:
        """Returns the output parser for step by step solution."""
        return OutputFixingParser.from_llm(self.llm, StrOutputParser()) 

    @property
    def prompt_template(self) -> ChatPromptTemplate:
        """Returns the prompt template for step by step solution."""
        msg_template = HumanMessagePromptTemplate.from_template(
            template=[
                {"type": "text", "text": FINAL_ANSWER_EXTRACTION_PROMPT},
            ]
        )
        return ChatPromptTemplate(messages=[msg_template])
    
    @property
    def chain(self):
        """Returns the LangChain chain for step by step solution."""
        return self.prompt_template | self.llm | self.parser
    
    def process(self, input_str: str) -> str:
        """Returns a step by step solution."""
        try:
            llm_output = self.chain.invoke({"input": input_str})
            return llm_output
        except Exception as e:
            from loguru import logger
            logger.error(f"Error {e} in line {e.__traceback__.tb_lineno}, code: {e.__traceback__.tb_frame.f_code.co_name}")
