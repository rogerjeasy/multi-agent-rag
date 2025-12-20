"""
Base Orchestrator for Multi-Agent RAG System

Handles agent initialization, configuration loading, and provides
base functionality for concrete orchestrator implementations.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

from agents.base_agent import (
    BaseAgent, AgentMessage, AgentConfig, AgentType,
    create_agent_message, Language
)


class BaseOrchestrator(ABC):
    """
    Base orchestrator that loads configuration and initializes agents.
    
    Subclasses implement specific orchestration strategies:
    - ParallelFusionOrchestrator: Parallel retrieval + fusion
    - CriticLoopOrchestrator: Iterative retrieval with critic feedback
    """
    
    def __init__(self, config_path: str):
        """
        Initialize orchestrator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = self._setup_logging()
        
        self._initialize_agents()
    
    def _load_config(self, path: str) -> dict:
        """Load YAML configuration file"""
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(config_file) as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ValueError(f"Empty configuration file: {path}")
        
        return config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup orchestrator logging"""
        log_level = self.config.get('global', {}).get('log_level', 'INFO')
        log_dir = self.config.get('global', {}).get('log_dir', 'logs')
        
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(getattr(logging, log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(
            Path(log_dir) / f'{self.__class__.__name__.lower()}.log'
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_agents(self):
        """Initialize all enabled agents from configuration"""
        agents_config = self.config.get('agents', {})
        
        if not agents_config:
            self.logger.warning("No agents configured")
            return
        
        for agent_name, agent_config in agents_config.items():
            if not agent_config.get('enabled', True):
                self.logger.info(f"Skipping disabled agent: {agent_name}")
                continue
            
            try:
                agent = self._create_agent(agent_name, agent_config)
                self.agents[agent_name] = agent
                self.logger.info(f"Initialized agent: {agent_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {agent_name}: {e}")
                raise
    
    def _create_agent(self, name: str, config_dict: dict) -> BaseAgent:
        """
        Factory method to create agents based on configuration.
        
        Args:
            name: Agent name from config
            config_dict: Agent configuration dictionary
            
        Returns:
            Initialized agent instance
        """
        agent_type = config_dict.get('type')
        if not agent_type:
            raise ValueError(f"Agent {name} missing 'type' field")
        
        # Extract agent config
        agent_config_data = config_dict.get('config', {})
        
        # Add global settings
        global_config = self.config.get('global', {})
        agent_config_data.setdefault('device', global_config.get('device', 'cpu'))
        agent_config_data.setdefault('cache_enabled', global_config.get('cache_enabled', True))
        agent_config_data.setdefault('cache_dir', global_config.get('cache_dir', 'cache'))
        agent_config_data.setdefault('log_dir', global_config.get('log_dir', 'logs'))
        
        # Create AgentConfig
        agent_config = AgentConfig(
            name=name,
            agent_type=AgentType(agent_type),
            extra_config=agent_config_data
        )
        
        # Import and instantiate agent based on type
        if agent_type == "query_understanding":
            from agents.query_understanding_agent import QueryUnderstandingAgent
            return QueryUnderstandingAgent(agent_config)
        
        elif agent_type == "bm25_retriever":
            from agents.bm25_retriever_agent import BM25RetrieverAgent
            return BM25RetrieverAgent(agent_config)
        
        elif agent_type == "dense_retriever":
            from agents.dense_retriever_agent import DenseRetrieverAgent
            return DenseRetrieverAgent(agent_config)
        
        elif agent_type == "graphrag_retriever":
            from agents.graphrag_retriever_agent import GraphRAGRetrieverAgent
            return GraphRAGRetrieverAgent(agent_config)
        
        elif agent_type == "fusion":
            from agents.fusion_agent import FusionAgent
            return FusionAgent(agent_config)
        
        elif agent_type == "reranker":
            from agents.reranker_agent import RerankerAgent
            return RerankerAgent(agent_config)
        
        elif agent_type == "answer_synthesizer":
            from agents.answer_synthesizer_agent import AnswerSynthesizerAgent
            return AnswerSynthesizerAgent(agent_config)
        
        elif agent_type == "critic":
            from agents.critic_agent import CriticAgent
            return CriticAgent(agent_config)
        
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get agent by name"""
        return self.agents.get(name)
    
    def has_agent(self, name: str) -> bool:
        """Check if agent is initialized"""
        return name in self.agents
    
    @abstractmethod
    def process_query(self, query: str, language: str = "en", **kwargs) -> AgentMessage:
        """
        Process a query through the orchestration pipeline.
        
        Must be implemented by subclasses.
        
        Args:
            query: User query
            language: Query language ("en" or "de")
            **kwargs: Additional parameters
            
        Returns:
            AgentMessage with final results
        """
        raise NotImplementedError("Subclasses must implement process_query()")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            'config_path': str(self.config_path),
            'num_agents': len(self.agents),
            'agents': list(self.agents.keys()),
            'enabled_agents': [
                name for name, config in self.config.get('agents', {}).items()
                if config.get('enabled', True)
            ]
        }