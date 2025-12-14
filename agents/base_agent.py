"""
Base Agent Architecture for Multi-Agent RAG System

This module provides the foundational classes and utilities for building
specialized agents in a multi-agent RAG pipeline. All agents inherit from
BaseAgent and follow a standardized message protocol.

Design Principles:
- Stateless agents (except for model/index loading)
- Standardized input/output message format
- Comprehensive logging and error handling
- Type validation and schema enforcement
- Performance tracking and metrics
"""

import json
import time
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# MESSAGE PROTOCOL & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

class QueryType(Enum):
    """Types of queries the system can handle"""
    FACTUAL = "factual"              # Who, What, When, Where
    ANALYTICAL = "analytical"        # Why, How
    COMPARATIVE = "comparative"      # Compare X vs Y
    TEMPORAL = "temporal"            # Time-based questions
    LIST = "list"                    # Requests for lists/enumerations
    DEFINITION = "definition"        # What is X?
    PROCEDURAL = "procedural"        # How to do X?
    UNKNOWN = "unknown"              # Cannot classify


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    GERMAN = "de"
    UNKNOWN = "unknown"


class AgentType(Enum):
    """Types of agents in the system"""
    QUERY_UNDERSTANDING = "query_understanding"
    BM25_RETRIEVER = "bm25_retriever"
    DENSE_RETRIEVER = "dense_retriever"
    GRAPHRAG_RETRIEVER = "graphrag_retriever"
    FUSION = "fusion"
    RERANKER = "reranker"
    SYNTHESIZER = "answer_synthesizer"
    CRITIC = "critic"
    ORCHESTRATOR = "orchestrator"


@dataclass
class RetrievedDocument:
    """Represents a single retrieved document chunk"""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str  # "bm25", "dense", "graphrag"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional fields from your existing implementation
    language: Optional[str] = None
    date: Optional[str] = None
    entities: Optional[List[Dict[str, str]]] = None
    keywords: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RetrievedDocument':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class QueryContext:
    """Metadata about a query"""
    query_type: QueryType = QueryType.UNKNOWN
    language: Language = Language.UNKNOWN
    complexity: float = 0.5  # 0-1 score
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    reformulated_queries: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_type": self.query_type.value,
            "language": self.language.value,
            "complexity": self.complexity,
            "entities": self.entities,
            "keywords": self.keywords,
            "reformulated_queries": self.reformulated_queries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryContext':
        return cls(
            query_type=QueryType(data.get("query_type", "unknown")),
            language=Language(data.get("language", "unknown")),
            complexity=data.get("complexity", 0.5),
            entities=data.get("entities", []),
            keywords=data.get("keywords", []),
            reformulated_queries=data.get("reformulated_queries", [])
        )


@dataclass
class Provenance:
    """Tracking information for debugging and explainability"""
    agents_called: List[str] = field(default_factory=list)
    retrieval_methods: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_time_ms: float = 0.0
    model_calls: Dict[str, int] = field(default_factory=dict)  # Track API calls
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentMessage:
    """
    Standard message format for inter-agent communication.
    This is the core protocol that all agents use.
    """
    # Core fields (always present)
    query: str
    message_id: str = field(default_factory=lambda: hashlib.md5(
        f"{time.time()}".encode()).hexdigest())
    
    # Context about the query
    context: QueryContext = field(default_factory=QueryContext)
    
    # Retrieved documents (populated by retrievers)
    results: List[RetrievedDocument] = field(default_factory=list)
    
    # Generated answer (populated by synthesizer)
    answer: Optional[str] = None
    
    # Confidence/quality scores
    confidence: float = 0.0  # 0-1
    relevance_score: float = 0.0  # 0-1
    
    # Tracking and debugging
    provenance: Provenance = field(default_factory=Provenance)
    
    # Agent-specific data (flexible extension point)
    agent_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "query": self.query,
            "message_id": self.message_id,
            "context": self.context.to_dict(),
            "results": [doc.to_dict() for doc in self.results],
            "answer": self.answer,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "provenance": self.provenance.to_dict(),
            "agent_data": self.agent_data
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Deserialize from dictionary"""
        return cls(
            query=data["query"],
            message_id=data.get("message_id", ""),
            context=QueryContext.from_dict(data.get("context", {})),
            results=[RetrievedDocument.from_dict(d) for d in data.get("results", [])],
            answer=data.get("answer"),
            confidence=data.get("confidence", 0.0),
            relevance_score=data.get("relevance_score", 0.0),
            provenance=Provenance(**data.get("provenance", {})),
            agent_data=data.get("agent_data", {})
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """Base configuration for all agents"""
    name: str
    agent_type: AgentType
    version: str = "1.0.0"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_dir: str = "logs"
    
    # Performance
    cache_enabled: bool = True
    cache_dir: str = "cache"
    timeout_seconds: int = 300
    
    # Model/Resource paths (from your existing implementation)
    device: str = "cpu"  # "cpu" or "cuda"
    
    # Agent-specific config (override in subclasses)
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["agent_type"] = self.agent_type.value
        return result


# ═══════════════════════════════════════════════════════════════
# LOGGING UTILITIES
# ═══════════════════════════════════════════════════════════════

class AgentLogger:
    """
    Structured logger for agents with JSON output support.
    Based on patterns from your existing code.
    """
    
    def __init__(self, agent_name: str, config: AgentConfig):
        self.agent_name = agent_name
        self.config = config
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Initialize logger with file and console handlers"""
        logger = logging.getLogger(self.agent_name)
        logger.setLevel(getattr(logging, self.config.log_level))
        logger.handlers.clear()
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console.setFormatter(console_fmt)
        logger.addHandler(console)
        
        # File handler (if enabled)
        if self.config.log_to_file:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{self.agent_name}.log"
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_fmt = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_fmt)
            logger.addHandler(file_handler)
        
        return logger
    
    def log_message(self, level: str, message: str, **kwargs):
        """Log with structured data"""
        log_data = {
            "agent": self.agent_name,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        
        log_method = getattr(self.logger, level.lower())
        if kwargs:
            log_method(f"{message} | Data: {json.dumps(kwargs, default=str)}")
        else:
            log_method(message)
    
    def info(self, message: str, **kwargs):
        self.log_message("INFO", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        self.log_message("DEBUG", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self.log_message("WARNING", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self.log_message("ERROR", message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        self.log_message("CRITICAL", message, **kwargs)


# ═══════════════════════════════════════════════════════════════
# EXCEPTION HANDLING
# ═══════════════════════════════════════════════════════════════

class AgentException(Exception):
    """Base exception for all agent errors"""
    def __init__(self, agent_name: str, message: str, details: Optional[Dict] = None):
        self.agent_name = agent_name
        self.message = message
        self.details = details or {}
        super().__init__(f"[{agent_name}] {message}")


class ValidationError(AgentException):
    """Raised when input/output validation fails"""
    pass


class ProcessingError(AgentException):
    """Raised when agent processing fails"""
    pass


class TimeoutError(AgentException):
    """Raised when agent exceeds timeout"""
    pass


class ConfigurationError(AgentException):
    """Raised when agent configuration is invalid"""
    pass


# ═══════════════════════════════════════════════════════════════
# BASE AGENT CLASS
# ═══════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the multi-agent RAG system.
    
    All agents must implement:
    - _process_impl(): Core processing logic
    - validate_input(): Input message validation
    
    All agents get:
    - Standardized logging
    - Error handling
    - Performance tracking
    - Input/output validation
    - Configuration management
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize base agent.
        
        Args:
            config: Agent configuration object
        """
        self.config = config
        self.name = config.name
        self.agent_type = config.agent_type
        
        # Setup logging
        self.logger = AgentLogger(self.name, config)
        
        # Performance tracking
        self._call_count = 0
        self._total_time_ms = 0.0
        self._error_count = 0
        
        # Cache setup (if enabled)
        self._cache = {} if config.cache_enabled else None
        if config.cache_enabled:
            self.cache_dir = Path(config.cache_dir) / self.name
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized {self.agent_type.value} agent: {self.name}")
    
    
    @abstractmethod
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """
        Core processing logic - must be implemented by each agent.
        
        Args:
            message: Input message
            
        Returns:
            Processed message
            
        Raises:
            ProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def validate_input(self, message: AgentMessage) -> bool:
        """
        Validate that input message has required fields for this agent.
        
        Args:
            message: Input message to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        pass
    
    def process(self, message: AgentMessage) -> AgentMessage:
        """
        Main entry point for processing. Handles validation, timing, errors.
        
        Args:
            message: Input message
            
        Returns:
            Processed message with updated provenance
            
        Raises:
            ValidationError: If input is invalid
            ProcessingError: If processing fails
            TimeoutError: If processing exceeds timeout
        """
        start_time = time.time()
        self._call_count += 1
        
        try:
            # Input validation
            self.logger.debug(f"Processing message {message.message_id}")
            self.validate_input(message)
            
            # Check cache
            if self._cache is not None:
                cached_result = self._get_from_cache(message)
                if cached_result is not None:
                    self.logger.info(f"Cache hit for message {message.message_id}")
                    return cached_result
            
            # Core processing
            result = self._process_impl(message)
            
            # Update provenance
            result.provenance.agents_called.append(self.name)
            result.provenance.execution_time_ms += (time.time() - start_time) * 1000
            
            # Output validation
            self.validate_output(result)
            
            # Cache result
            if self._cache is not None:
                self._save_to_cache(message, result)
            
            # Update metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self._total_time_ms += elapsed_ms
            
            self.logger.info(
                f"Processed successfully",
                elapsed_ms=f"{elapsed_ms:.2f}",
                message_id=message.message_id
            )
            
            return result
            
        except ValidationError as e:
            self._error_count += 1
            self.logger.error(f"Validation error: {e}", message_id=message.message_id)
            raise
            
        except Exception as e:
            self._error_count += 1
            self.logger.error(
                f"Processing error: {str(e)}",
                message_id=message.message_id,
                error_type=type(e).__name__
            )
            raise ProcessingError(self.name, str(e), {"message_id": message.message_id})
    
    def validate_output(self, message: AgentMessage) -> bool:
        """
        Validate output message (can be overridden by subclasses).
        
        Args:
            message: Output message to validate
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        # Base validation - ensure message is valid
        if not isinstance(message, AgentMessage):
            raise ValidationError(
                self.name,
                "Output must be AgentMessage",
                {"type": type(message).__name__}
            )
        
        return True
    
    def _get_cache_key(self, message: AgentMessage) -> str:
        """Generate cache key from message"""
        # Hash query + context for cache key
        cache_input = f"{message.query}_{message.context.language.value}"
        return hashlib.md5(cache_input.encode()).hexdigest()
    
    def _get_from_cache(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Retrieve from cache if available"""
        if self._cache is None:
            return None
        
        cache_key = self._get_cache_key(message)
        
        # Check in-memory cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with cache_file.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                result = AgentMessage.from_dict(data)
                self._cache[cache_key] = result  # Load into memory
                return result
            except Exception as e:
                self.logger.warning(f"Cache read error: {e}", cache_key=cache_key)
        
        return None
    
    def _save_to_cache(self, message: AgentMessage, result: AgentMessage):
        """Save result to cache"""
        if self._cache is None:
            return
        
        cache_key = self._get_cache_key(message)
        
        # Save to memory
        self._cache[cache_key] = result
        
        # Save to disk
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with cache_file.open('w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"Cache write error: {e}", cache_key=cache_key)
    
    def clear_cache(self):
        """Clear agent's cache"""
        if self._cache is not None:
            self._cache.clear()
            # Optionally clear disk cache
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            self.logger.info("Cache cleared")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        avg_time_ms = (
            self._total_time_ms / self._call_count if self._call_count > 0 else 0.0
        )
        
        return {
            "agent_name": self.name,
            "agent_type": self.agent_type.value,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "total_time_ms": self._total_time_ms,
            "avg_time_ms": avg_time_ms,
            "error_rate": (
                self._error_count / self._call_count if self._call_count > 0 else 0.0
            ),
            "cache_size": len(self._cache) if self._cache is not None else 0
        }
    
    def reset_metrics(self):
        """Reset performance counters"""
        self._call_count = 0
        self._total_time_ms = 0.0
        self._error_count = 0
        self.logger.info("Metrics reset")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', type={self.agent_type.value})"
    
    def __str__(self) -> str:
        return self.name

def create_agent_message(
    query: str,
    language: str = "unknown",
    results: Optional[List[Dict]] = None
) -> AgentMessage:
    """
    Convenience function to create an AgentMessage.
    
    Args:
        query: The user's question
        language: Query language ("en", "de", or "unknown")
        results: Optional list of retrieved documents (as dicts)
    
    Returns:
        AgentMessage instance
    """
    context = QueryContext(language=Language(language))
    
    doc_results = []
    if results:
        for r in results:
            doc_results.append(RetrievedDocument.from_dict(r))
    
    return AgentMessage(
        query=query,
        context=context,
        results=doc_results
    )


def merge_results(
    results1: List[RetrievedDocument],
    results2: List[RetrievedDocument],
    deduplicate: bool = True
) -> List[RetrievedDocument]:
    """
    Merge two lists of retrieved documents.
    
    Args:
        results1: First list of documents
        results2: Second list of documents
        deduplicate: Whether to remove duplicates (by chunk_id)
    
    Returns:
        Merged list
    """
    merged = results1 + results2
    
    if deduplicate:
        seen = set()
        deduplicated = []
        for doc in merged:
            if doc.chunk_id not in seen:
                seen.add(doc.chunk_id)
                deduplicated.append(doc)
        return deduplicated
    
    return merged