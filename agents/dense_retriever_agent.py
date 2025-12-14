"""
Dense Retriever Agent

Wraps the Chroma vector store with E5 embeddings from Step 2.1 into the agent framework.
"""

import json
from typing import Optional
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
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


class DenseRetrieverAgent(BaseAgent):
    """
    Dense retrieval agent using E5 embeddings and Chroma vector store.
    
    Retrieves documents based on semantic similarity in vector space.
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Extract configuration
        model_name = config.extra_config.get("model_name", "intfloat/multilingual-e5-large")
        index_path = config.extra_config.get("index_path")
        device = config.extra_config.get("device", config.device)
        
        if not index_path:
            raise ValidationError(
                self.name,
                "index_path required in config",
                {"extra_config": config.extra_config}
            )
        
        self.index_path = Path(index_path)
        self.top_k = config.extra_config.get("top_k", 20)
        self.similarity_metric = config.extra_config.get("similarity_metric", "cosine")
        self.batch_size = config.extra_config.get("batch_size", 32)
        self.normalize_embeddings = config.extra_config.get("normalize_embeddings", True)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={
                "batch_size": self.batch_size,
                "normalize_embeddings": self.normalize_embeddings
            }
        )
        
        # Load Chroma index
        self.vectorstore = self._load_index()
        
        self.logger.info(
            "Dense index loaded",
            index_path=str(self.index_path),
            model_name=model_name,
            top_k=self.top_k
        )
    
    def _load_index(self):
        """Load Chroma vector store from disk"""
        try:
            vectorstore = Chroma(
                persist_directory=str(self.index_path),
                embedding_function=self.embeddings
            )
            return vectorstore
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Failed to load Chroma index: {str(e)}",
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
    
    def _format_query(self, query: str) -> str:
        """Format query for E5 model (add 'query:' prefix)"""
        return f"query: {query.strip()}"
    
    def _parse_metadata(self, metadata: dict) -> dict:
        """Parse JSON-serialized metadata fields back to original types"""
        parsed = {}
        for k, v in metadata.items():
            if isinstance(v, str) and (v.startswith('[') or v.startswith('{')):
                try:
                    parsed[k] = json.loads(v)
                except:
                    parsed[k] = v
            else:
                parsed[k] = v
        return parsed
    
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """Execute dense retrieval"""
        query = message.query
        
        # Detect language and update context
        detected_lang = self._detect_language(query)
        message.context.language = Language(detected_lang)
        
        # Format query with E5 prefix
        formatted_query = self._format_query(query)
        
        self.logger.debug(
            "Executing dense search",
            query=query,
            language=detected_lang,
            top_k=self.top_k
        )
        
        try:
            # Retrieve with similarity scores
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                formatted_query,
                k=self.top_k
            )
            
            # Convert to RetrievedDocument format
            documents = []
            for lc_doc, distance in docs_with_scores:
                # Convert distance to similarity (1 - distance for cosine)
                similarity = 1.0 - float(distance)
                
                # Parse metadata (some fields might be JSON strings)
                meta = self._parse_metadata(lc_doc.metadata)
                meta["dense_score"] = similarity
                
                # Remove "passage: " prefix from text if present
                text = lc_doc.page_content
                if text.startswith("passage: "):
                    text = text[9:]
                
                doc = RetrievedDocument(
                    chunk_id=meta.get("chunk_id") or meta.get("record_id", "unknown"),
                    doc_id=meta.get("doc_id", "unknown"),
                    text=text,
                    score=similarity,
                    source="dense",
                    metadata=meta,
                    language=meta.get("language"),
                    date=meta.get("date"),
                    entities=meta.get("entities"),
                    keywords=meta.get("keywords") or meta.get("topic_tags")
                )
                documents.append(doc)
            
            # Update message
            message.results = documents
            message.provenance.retrieval_methods.append("dense")
            
            self.logger.info(
                "Dense retrieval completed",
                num_results=len(documents),
                avg_score=sum(d.score for d in documents) / len(documents) if documents else 0.0
            )
            
            return message
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Dense search failed: {str(e)}",
                {
                    "query": query,
                    "message_id": message.message_id
                }
            )