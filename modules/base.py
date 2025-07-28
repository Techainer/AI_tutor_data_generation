from typing import Any, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod
from tenacity import retry, stop_after_attempt, wait_fixed
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from loguru import logger
from .utils import image_to_base64


class ModelProvider(Enum):
    OPENAI = "openai"
    AZURE = "azure"
    GOOGLE = "google"
    CLAUDE = "claude"

    @classmethod
    def from_string(cls, value: str):
        """Convert a string to the corresponding Enum value."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"Invalid type: {value}")

class BaseEngine(ABC):
    """Base class for all engines."""
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Process the input data using the engine."""
        pass


class BaseLLMModule(BaseEngine):
    """Base class for all LLM-based modules."""

    def __init__(
        self, 
        api_key: str, 
        llm_model: str,
        provider: Union[str, ModelProvider] = ModelProvider.OPENAI,
        llm_host: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.75,
        presence_penalty: float = 0.5,
        frequency_penalty: float = 0.5,
        max_tokens: int = 16384,
        timeout: Optional[int] = 45,
        max_retries: int = 3,
        system_prompt: Optional[str] = None,
        trace_cost: bool = False,
        thinking_budget: Optional[int] = None,
    ):
        """
        Initialize the analysis pipeline with the specified configuration.
        
        Args:
            api_key: Authentication key for the LLM API
            llm_model: Model identifier to use
            provider: Provider of the LLM (OPENAI or AZURE)
            llm_host: Base URL for API calls
            api_version: API version for Azure OpenAI
            temperature: Controls randomness in output (0 to 1)
            top_p: Controls diversity via nucleus sampling
            presence_penalty: Penalizes topic repetition
            frequency_penalty: Penalizes token repetition
            max_tokens: Maximum tokens to generate
            timeout: API call timeout in seconds
            max_retries: Number of retry attempts for API calls
            system_prompt: Optional system prompt to use for all requests
        """
        self.api_key = api_key
        self.llm_model = llm_model
        self.provider = provider
        self.llm_host = llm_host
        self.api_version = api_version
        self.system_prompt = system_prompt
        
    
        
        # Initialize Langfuse callback if keys are provided
        self.callbacks = []
        
        # Store LLM parameters
        self.llm_params = {
            "model": llm_model,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "max_tokens": max_tokens,
            "request_timeout": timeout,
            "max_retries": max_retries,
            "api_key": api_key,
        }
        if thinking_budget:
            self.llm_params["thinking_budget"] = thinking_budget
        
        # Add callbacks to LLM params if available
        if self.callbacks:
            self.llm_params["callbacks"] = self.callbacks
        
        if isinstance(self.provider, str):
            self.provider = ModelProvider.from_string(self.provider)
            
        if self.provider == ModelProvider.AZURE:
            if not llm_host or not api_version:
                raise ValueError("Azure OpenAI requires llm_host and api_version")
            self.llm_params["azure_endpoint"] = llm_host
            self.llm_params["api_version"] = api_version
            self.llm = AzureChatOpenAI(**self.llm_params)
        elif self.provider == ModelProvider.OPENAI:
            if llm_host:
                self.llm_params["base_url"] = llm_host
            self.llm = ChatOpenAI(**self.llm_params)
        elif self.provider == ModelProvider.GOOGLE:
            self.llm = ChatGoogleGenerativeAI(**self.llm_params)
        elif self.provider == ModelProvider.CLAUDE:
            self.llm_params.pop('presence_penalty')
            self.llm_params.pop('frequency_penalty')
            self.llm = ChatAnthropic(**self.llm_params)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        logger.info(f"Initialized {self.provider.value} pipeline with model {llm_model}")
    
    def _process_image_input(self, image_input: Union[str, bytes]) -> str:
        if isinstance(image_input, bytes):
            image_input = image_to_base64(image_input)
        return image_input

    ### Add retry for LLM module
    @abstractmethod
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def process(self, *args, **kwargs) -> Any:
        """Process the input data using the engine."""
        pass

    @property
    @abstractmethod
    def chain(self):
        """Get the module-specific chain."""
        pass

    @property
    @abstractmethod
    def prompt_template(self):
        """Get the module-specific prompt template."""
        pass

    @property
    @abstractmethod
    def parser(self):
        """Get the module-specific output parser."""
        pass