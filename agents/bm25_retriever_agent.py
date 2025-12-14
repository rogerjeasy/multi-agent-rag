"""
BM25 Retriever Agent

Wraps the BilingualBM25 retriever from Step 2.1 into the agent framework.
"""

import pickle
from typing import Optional, List
from pathlib import Path
from langdetect import detect

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentMessage,
    RetrievedDocument,
    ValidationError,
    ProcessingError,
    Language,
)


class BM25RetrieverAgent(BaseAgent):
    """
    BM25-based retrieval agent with bilingual support (EN/DE).
    
    Loads a pickled BilingualBM25 index and retrieves documents
    using lexical matching.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Extract configuration
        index_path = config.extra_config.get("index_path")
        if not index_path:
            raise ValidationError(
                self.name,
                "index_path required in config",
                {"extra_config": config.extra_config}
            )
        
        self.index_path = Path(index_path)
        self.top_k = config.extra_config.get("top_k", 20)
        self.k1 = config.extra_config.get("k1", 1.2)
        self.b = config.extra_config.get("b", 0.75)
        
        # Load BM25 index
        self.bm25_index = self._load_index()
        
        self.logger.info(
            "BM25 index loaded",
            index_path=str(self.index_path),
            top_k=self.top_k
        )
    
    def _load_index(self):
        """Load pickled BilingualBM25 index"""
        try:
            with open(self.index_path, 'rb') as f:
                index = pickle.load(f)
            return index
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Failed to load BM25 index: {str(e)}",
                {"index_path": str(self.index_path)}
            )
    
    def validate_input(self, message: AgentMessage) -> bool:
        """Validate input message"""
        if not message.query or not message.query.strip():
            raise ValidationError(
                self.name,
                "Query cannot be empty",
                {"query": message.query}
            )
        
        if len(message.query) < 3:
            raise ValidationError(
                self.name,
                "Query too short (minimum 3 characters)",
                {"query_length": len(message.query)}
            )
        
        return True
    
    def _detect_language(self, query: str) -> str:
        """Detect query language"""
        try:
            lang = detect(query)
            if lang not in ("en", "de"):
                lang = "en"
        except:
            lang = "en"
        return lang
    
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """Execute BM25 retrieval"""
        query = message.query
        
        # Detect language and update context
        detected_lang = self._detect_language(query)
        message.context.language = Language(detected_lang)
        
        self.logger.debug(
            "Executing BM25 search",
            query=query,
            language=detected_lang,
            top_k=self.top_k
        )
        
        try:
            # Search using BilingualBM25
            langchain_docs = self.bm25_index.search(
                query=query,
                top_k=self.top_k
            )
            
            # Convert LangChain documents to RetrievedDocument format
            documents = []
            for lc_doc in langchain_docs:
                meta = lc_doc.metadata
                
                doc = RetrievedDocument(
                    chunk_id=meta.get("record_id") or meta.get("chunk_id", "unknown"),
                    doc_id=meta.get("doc_id", "unknown"),
                    text=lc_doc.page_content,
                    score=meta.get("bm25_score", 0.0),
                    source="bm25",
                    metadata=meta,
                    language=meta.get("language"),
                    date=meta.get("date"),
                    entities=meta.get("entities"),
                    keywords=meta.get("keywords")
                )
                documents.append(doc)
            
            # Update message
            message.results = documents
            message.provenance.retrieval_methods.append("bm25")
            
            self.logger.info(
                "BM25 retrieval completed",
                num_results=len(documents),
                avg_score=sum(d.score for d in documents) / len(documents) if documents else 0.0
            )
            
            return message
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"BM25 search failed: {str(e)}",
                {
                    "query": query,
                    "message_id": message.message_id
                }
            )