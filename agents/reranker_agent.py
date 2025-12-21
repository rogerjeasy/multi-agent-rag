"""
Reranker Agent

Reorders retrieved documents using neural reranking models.
Supports multiple backends: Cohere, OpenAI, local models (GTE, BGE).
"""

import os
from typing import List, Optional
import numpy as np

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentMessage,
    RetrievedDocument,
    ValidationError,
    ProcessingError,
)


class RerankerAgent(BaseAgent):
    """
    Reranker agent that reorders documents by relevance.
    
    Supports:
    - Cohere rerank API
    - OpenAI-based reranking
    - Local models (GTE, BGE)
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Extract configuration
        self.reranker_type = config.extra_config.get("reranker_type", "cohere")
        self.rerank_top_k = config.extra_config.get("rerank_top_k", 20)
        self.output_top_k = config.extra_config.get("output_top_k", 10)
        
        # Initialize reranker based on type
        if self.reranker_type == "cohere":
            self._init_cohere(config)
        elif self.reranker_type == "openai":
            self._init_openai(config)
        elif self.reranker_type == "gte":
            self._init_gte(config)
        elif self.reranker_type == "local":
            self._init_local(config)
        else:
            raise ValidationError(
                self.name,
                f"Unknown reranker type: {self.reranker_type}",
                {"reranker_type": self.reranker_type}
            )
        
        self.logger.info(
            "Reranker initialized",
            type=self.reranker_type,
            rerank_top_k=self.rerank_top_k,
            output_top_k=self.output_top_k
        )
    
    def _init_cohere(self, config: AgentConfig):
        """Initialize Cohere reranker"""
        try:
            import cohere
        except ImportError:
            raise ValidationError(
                self.name,
                "Cohere not installed. Run: pip install cohere",
                {}
            )
        
        cohere_config = config.extra_config.get("cohere", {})
        self.cohere_model = cohere_config.get("model", "rerank-english-v3.0")
        self.max_chunks_per_doc = cohere_config.get("max_chunks_per_doc", 10)
        
        api_key_env = cohere_config.get("api_key_env", "COHERE_API_KEY")
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValidationError(
                self.name,
                f"Cohere API key not found: {api_key_env}",
                {"api_key_env": api_key_env}
            )
        
        self.cohere_client = cohere.Client(api_key)
    
    def _init_openai(self, config: AgentConfig):
        """Initialize OpenAI-based reranker"""
        try:
            from openai import OpenAI
        except ImportError:
            raise ValidationError(
                self.name,
                "OpenAI not installed. Run: pip install openai",
                {}
            )
        
        openai_config = config.extra_config.get("openai", {})
        self.openai_model = openai_config.get("model", "gpt-4o-mini")
        
        api_key_env = openai_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValidationError(
                self.name,
                f"OpenAI API key not found: {api_key_env}",
                {"api_key_env": api_key_env}
            )
        
        self.openai_client = OpenAI(api_key=api_key)
    
    def _init_gte(self, config: AgentConfig):
        """Initialize GTE reranker"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ValidationError(
                self.name,
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                {}
            )
        
        gte_config = config.extra_config.get("gte", {})
        model_name = gte_config.get("model_name", "Alibaba-NLP/gte-large-en-v1.5")
        
        self.cross_encoder = CrossEncoder(
            model_name,
            max_length=512,
            device=config.device
        )
    
    def _init_local(self, config: AgentConfig):
        """Initialize local BGE reranker"""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ValidationError(
                self.name,
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                {}
            )
        
        local_config = config.extra_config.get("local", {})
        model_name = local_config.get("model_name", "BAAI/bge-reranker-large")
        
        self.cross_encoder = CrossEncoder(
            model_name,
            max_length=512,
            device=config.device
        )
    
    def validate_input(self, message: AgentMessage) -> bool:
        """Validate input message"""
        if not message.query or not message.query.strip():
            raise ValidationError(
                self.name,
                "Query cannot be empty",
                {"query": message.query}
            )
        
        if not message.results or len(message.results) == 0:
            raise ValidationError(
                self.name,
                "No documents to rerank",
                {"num_results": len(message.results) if message.results else 0}
            )
        
        return True
    
    def _cohere_rerank(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Rerank using Cohere API"""
        # Take top rerank_top_k documents
        docs_to_rerank = documents[:self.rerank_top_k]
        
        # Prepare texts
        texts = [doc.text for doc in docs_to_rerank]
        
        try:
            # Call Cohere rerank API
            results = self.cohere_client.rerank(
                model=self.cohere_model,
                query=query,
                documents=texts,
                top_n=self.output_top_k,
                return_documents=False
            )
            
            # Build reranked list
            reranked = []
            for result in results.results:
                idx = result.index
                score = result.relevance_score
                
                doc = docs_to_rerank[idx]
                doc.score = score
                doc.metadata["rerank_score"] = score
                doc.metadata["rerank_rank"] = len(reranked) + 1
                reranked.append(doc)
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"Cohere reranking failed: {e}")
            # Fallback: return original order
            return docs_to_rerank[:self.output_top_k]
    
    def _openai_rerank(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Rerank using OpenAI (LLM-based scoring)"""
        docs_to_rerank = documents[:self.rerank_top_k]
        
        # Build prompt
        doc_texts = []
        for i, doc in enumerate(docs_to_rerank, 1):
            snippet = doc.text[:400]  # Limit length
            doc_texts.append(f"[{i}] {snippet}")
        
        docs_list = "\n\n".join(doc_texts)
        
        prompt = f"""Given the query and documents below, score each document's relevance to the query on a scale of 0.0 to 1.0.

Query: {query}

Documents:
{docs_list}

Output only a JSON array of scores in order, e.g.: [0.9, 0.7, 0.3, ...]
Output {len(docs_to_rerank)} scores total."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a relevance scorer. Output only valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse JSON
            import json
            scores = json.loads(result_text)
            
            # Validate scores
            if len(scores) != len(docs_to_rerank):
                raise ValueError(f"Expected {len(docs_to_rerank)} scores, got {len(scores)}")
            
            # Attach scores and sort
            scored_docs = []
            for doc, score in zip(docs_to_rerank, scores):
                doc.score = float(score)
                doc.metadata["rerank_score"] = float(score)
                scored_docs.append((doc, float(score)))
            
            # Sort by score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Build final list
            reranked = []
            for doc, _ in scored_docs[:self.output_top_k]:
                doc.metadata["rerank_rank"] = len(reranked) + 1
                reranked.append(doc)
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"OpenAI reranking failed: {e}")
            return docs_to_rerank[:self.output_top_k]
    
    def _cross_encoder_rerank(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Rerank using local cross-encoder (GTE or BGE)"""
        docs_to_rerank = documents[:self.rerank_top_k]
        
        # Prepare pairs
        pairs = [[query, doc.text] for doc in docs_to_rerank]
        
        try:
            # Get scores
            scores = self.cross_encoder.predict(pairs)
            
            # Attach scores
            scored_docs = []
            for doc, score in zip(docs_to_rerank, scores):
                doc.score = float(score)
                doc.metadata["rerank_score"] = float(score)
                scored_docs.append((doc, float(score)))
            
            # Sort by score
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # Build final list
            reranked = []
            for doc, _ in scored_docs[:self.output_top_k]:
                doc.metadata["rerank_rank"] = len(reranked) + 1
                reranked.append(doc)
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"Cross-encoder reranking failed: {e}")
            return docs_to_rerank[:self.output_top_k]
    
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """Execute reranking"""
        query = message.query
        documents = message.results
        
        self.logger.debug(
            "Executing reranking",
            type=self.reranker_type,
            num_input_docs=len(documents),
            rerank_top_k=self.rerank_top_k
        )
        
        try:
            # Select reranking method
            if self.reranker_type == "cohere":
                reranked = self._cohere_rerank(query, documents)
            elif self.reranker_type == "openai":
                reranked = self._openai_rerank(query, documents)
            elif self.reranker_type in ["gte", "local"]:
                reranked = self._cross_encoder_rerank(query, documents)
            else:
                raise ValidationError(
                    self.name,
                    f"Unknown reranker type: {self.reranker_type}",
                    {"type": self.reranker_type}
                )
            
            # Update message
            message.results = reranked
            message.provenance.retrieval_methods.append(f"rerank_{self.reranker_type}")
            
            self.logger.info(
                "Reranking completed",
                type=self.reranker_type,
                input_docs=len(documents),
                output_docs=len(reranked),
                avg_score=sum(d.score for d in reranked) / len(reranked) if reranked else 0.0
            )
            
            return message
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Reranking failed: {str(e)}",
                {
                    "type": self.reranker_type,
                    "query": query,
                    "message_id": message.message_id
                }
            )