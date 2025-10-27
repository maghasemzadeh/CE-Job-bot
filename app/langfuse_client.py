from langfuse import get_client
from langfuse.openai import openai
from threading import Lock
from app.config import Settings

class LangfuseSingleton:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(LangfuseSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        # Initialize the langfuse client
        self.langfuse = get_client()
        self._initialized = True

    def ask(self, prompt_name: str, temperature=0, parser=None, **kwargs):
        """
        Use a named Langfuse prompt and pass inputs to the LLM, return the response.
        
        :param prompt_name: Name of the prompt stored in Langfuse
        :param kwargs: Input variables for the prompt
        :return: The output of the LLM call (usually a string or a completion object)
        """
        from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
        from langchain_openai import ChatOpenAI
        from langfuse.langchain import CallbackHandler

        langfuse_text_prompt = self.langfuse.get_prompt(prompt_name)
        langfuse_handler = CallbackHandler()

        langchain_text_prompt = PromptTemplate.from_template(
            langfuse_text_prompt.get_langchain_prompt(),
            metadata={"langfuse_prompt": langfuse_text_prompt},
        )
        
        llm = ChatOpenAI(
            model=Settings.LLM_MODEL_NAME if hasattr(Settings, "LLM_MODEL_NAME") else "gpt-4o-mini",
            temperature=temperature,
            base_url=Settings.LLM_BASE_URL,
            api_key=Settings.OPENAI_API_KEY,
            timeout=60,
            max_retries=2,
        )
        if parser:
            completion_chain = langchain_text_prompt | llm | parser
        else:
            completion_chain = langchain_text_prompt | llm

        
        result = completion_chain.invoke(input=kwargs, config={"callbacks": [langfuse_handler]})

        return getattr(result, "content", result)

