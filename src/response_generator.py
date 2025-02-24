"""
Module for generating responses using OpenAI Chat models.
"""

from langchain_openai.chat_models import ChatOpenAI
import logging
from typing import List

from src.config import CONFIG

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates responses using a Chat model based on provided context."""

    def __init__(
        self,
        model_name: str = CONFIG.CHAT_MODEL,
        temperature: float = CONFIG.TEMPERATURE,
    ) -> None:
        """
        Args:
            model_name: The chat model to use.
            temperature: Temperature setting for response variability.
        """
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def generate(self, query: str, relevant_docs: List[str]) -> str:
        """Generates a response based on the context.

        Args:
            query: The user query.
            relevant_docs: A list of relevant documents (as context).

        Returns:
            The generated response as a string.
        """
        context = "\n".join(relevant_docs)
        prompt = (
            f"Use the following context to answer the query:\n\n"
            f"Context: {context}\n\nQuery: {query}"
        )
        logger.debug("Generating response for query: %s", query)

        response = self.llm.invoke(prompt)

        return response.content if hasattr(response, "content") else str(response)
