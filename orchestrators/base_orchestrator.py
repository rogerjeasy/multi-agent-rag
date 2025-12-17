
from abc import ABC, abstractmethod
from typing import Any

class BaseOrchestrator(ABC):
    """
    Base class for all implemented orchestrators
    """
    def __init__(self, llm: Any, retriever: Any):
        self.llm = llm
        self.retriever = retriever

    @abstractmethod
    def run(self, query: str) -> str:
        """
        Process the query to return a response
        """
        pass
