"""
Critic Agent

Evaluates the quality of generated answers and triggers re-retrieval if needed.
Uses LLM-based verification, similarity checks, or keyword matching.
"""

import os
from typing import Optional, Dict, List
from openai import OpenAI
import numpy as np

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentMessage,
    ValidationError,
    ProcessingError,
)


class CriticAgent(BaseAgent):
    """
    Critic agent that evaluates answer quality.
    
    Provides:
    - Confidence scoring
    - Relevance assessment
    - Re-retrieval recommendations
    - Quality feedback
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Extract configuration
        self.min_confidence = config.extra_config.get("min_confidence", 0.6)
        self.min_relevance = config.extra_config.get("min_relevance", 0.5)
        self.verification_method = config.extra_config.get("verification_method", "llm")
        self.trigger_reretrival = config.extra_config.get("trigger_reretrival", True)
        self.max_reretrieval_attempts = config.extra_config.get("max_reretrieval_attempts", 3)
        
        # LLM settings (if using LLM verification)
        if self.verification_method == "llm":
            self.model = config.extra_config.get("model", "gpt-4o-mini")
            api_key_env = config.extra_config.get("api_key_env", "OPENAI_API_KEY")
            api_key = os.getenv(api_key_env)
            
            if not api_key:
                raise ValidationError(
                    self.name,
                    f"API key not found in environment: {api_key_env}",
                    {"api_key_env": api_key_env}
                )
            
            self.client = OpenAI(api_key=api_key)
        
        self.logger.info(
            "Critic agent initialized",
            verification_method=self.verification_method,
            min_confidence=self.min_confidence,
            min_relevance=self.min_relevance
        )
    
    def validate_input(self, message: AgentMessage) -> bool:
        """Validate that message contains an answer and source documents"""
        if not message.answer or not message.answer.strip():
            raise ValidationError(
                self.name,
                "No answer to evaluate",
                {"answer": message.answer}
            )
        
        if not message.query or not message.query.strip():
            raise ValidationError(
                self.name,
                "No query provided for evaluation",
                {"query": message.query}
            )
        
        return True
    
    def _llm_verification(self, query: str, answer: str, documents: List) -> Dict:
        """
        Use LLM to verify answer quality.
        
        Returns:
            Dict with confidence, relevance, feedback, and recommendation
        """
        # Build context from documents
        context_snippets = []
        for i, doc in enumerate(documents[:5], 1):
            snippet = doc.text[:300] if hasattr(doc, 'text') else str(doc)[:300]
            context_snippets.append(f"[{i}] {snippet}...")
        
        context_text = "\n\n".join(context_snippets)
        
        # Evaluation prompt
        prompt = f"""You are an expert evaluator assessing the quality of a RAG system's answer.

Query: {query}

Answer: {answer}

Source Documents:
{context_text}

Evaluate the answer on these dimensions:

1. CONFIDENCE (0.0-1.0): How confident are you that the answer is correct based on the sources?
2. RELEVANCE (0.0-1.0): How well does the answer address the query?
3. GROUNDING (0.0-1.0): How well is the answer supported by the source documents?
4. COMPLETENESS (0.0-1.0): Does the answer fully address all aspects of the query?

Provide your evaluation in this exact JSON format:
{{
    "confidence": <float 0.0-1.0>,
    "relevance": <float 0.0-1.0>,
    "grounding": <float 0.0-1.0>,
    "completeness": <float 0.0-1.0>,
    "issues": ["list", "of", "issues"],
    "recommendation": "accept|reretrieve|reformulate",
    "feedback": "Brief explanation of scores and recommendation"
}}

Only output valid JSON, nothing else."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise answer quality evaluator. Output only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            result_text = response.content[0].text.strip()
            
            # Parse JSON
            import json
            result = json.loads(result_text)
            
            return result
            
        except Exception as e:
            self.logger.error(f"LLM verification failed: {e}")
            # Return conservative fallback
            return {
                "confidence": 0.5,
                "relevance": 0.5,
                "grounding": 0.5,
                "completeness": 0.5,
                "issues": ["LLM evaluation failed"],
                "recommendation": "accept",
                "feedback": f"Error during evaluation: {str(e)}"
            }
    
    def _similarity_verification(self, query: str, answer: str, documents: List) -> Dict:
        """
        Use embedding similarity to verify answer quality.
        
        Simple heuristic: check if answer is semantically close to query and documents.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            # Initialize embedder (cached)
            if not hasattr(self, '_embedder'):
                self._embedder = SentenceTransformer(
                    'sentence-transformers/all-MiniLM-L6-v2',
                    device=self.config.device
                )
            
            # Encode
            query_emb = self._embedder.encode([query], normalize_embeddings=True)[0]
            answer_emb = self._embedder.encode([answer], normalize_embeddings=True)[0]
            
            # Get top doc embeddings
            doc_texts = [d.text for d in documents[:5] if hasattr(d, 'text')]
            if doc_texts:
                doc_embs = self._embedder.encode(doc_texts, normalize_embeddings=True)
                doc_sims = doc_embs @ answer_emb
                max_doc_sim = float(np.max(doc_sims))
                avg_doc_sim = float(np.mean(doc_sims))
            else:
                max_doc_sim = 0.0
                avg_doc_sim = 0.0
            
            # Query-answer similarity
            qa_sim = float(query_emb @ answer_emb)
            
            # Heuristic scoring
            confidence = (max_doc_sim + avg_doc_sim) / 2
            relevance = qa_sim
            grounding = max_doc_sim
            
            # Determine recommendation
            if confidence < self.min_confidence or relevance < self.min_relevance:
                recommendation = "reretrieve"
            else:
                recommendation = "accept"
            
            return {
                "confidence": confidence,
                "relevance": relevance,
                "grounding": grounding,
                "completeness": (confidence + relevance) / 2,
                "issues": [] if recommendation == "accept" else ["Low similarity scores"],
                "recommendation": recommendation,
                "feedback": f"QA_sim={qa_sim:.2f}, Doc_sim={max_doc_sim:.2f}"
            }
            
        except Exception as e:
            self.logger.error(f"Similarity verification failed: {e}")
            return {
                "confidence": 0.5,
                "relevance": 0.5,
                "grounding": 0.5,
                "completeness": 0.5,
                "issues": ["Similarity check failed"],
                "recommendation": "accept",
                "feedback": str(e)
            }
    
    def _keyword_verification(self, query: str, answer: str, documents: List) -> Dict:
        """
        Simple keyword-based verification.
        
        Checks if important query terms appear in answer.
        """
        # Extract keywords from query (simple tokenization)
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Simple word tokenization
        import re
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        answer_words = set(re.findall(r'\b\w+\b', answer_lower))
        
        # Remove stopwords (basic list)
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'what', 'when', 'where', 'who', 'how'}
        
        query_keywords = query_words - stopwords
        
        if not query_keywords:
            # Query has no meaningful keywords
            return {
                "confidence": 0.7,
                "relevance": 0.7,
                "grounding": 0.7,
                "completeness": 0.7,
                "issues": [],
                "recommendation": "accept",
                "feedback": "No keywords to verify"
            }
        
        # Check overlap
        matched_keywords = query_keywords & answer_words
        keyword_coverage = len(matched_keywords) / len(query_keywords)
        
        # Simple scoring
        confidence = keyword_coverage
        relevance = keyword_coverage
        
        recommendation = "accept" if keyword_coverage >= 0.3 else "reretrieve"
        
        return {
            "confidence": confidence,
            "relevance": relevance,
            "grounding": confidence,
            "completeness": confidence,
            "issues": [] if recommendation == "accept" else [f"Low keyword coverage: {keyword_coverage:.2f}"],
            "recommendation": recommendation,
            "feedback": f"Keyword coverage: {keyword_coverage:.2f} ({len(matched_keywords)}/{len(query_keywords)})"
        }
    
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """Evaluate answer quality"""
        query = message.query
        answer = message.answer
        documents = message.results or []
        
        self.logger.debug(
            "Evaluating answer quality",
            method=self.verification_method,
            answer_length=len(answer),
            num_docs=len(documents)
        )
        
        try:
            # Run verification based on method
            if self.verification_method == "llm":
                evaluation = self._llm_verification(query, answer, documents)
            elif self.verification_method == "similarity":
                evaluation = self._similarity_verification(query, answer, documents)
            elif self.verification_method == "keyword":
                evaluation = self._keyword_verification(query, answer, documents)
            else:
                raise ValidationError(
                    self.name,
                    f"Unknown verification method: {self.verification_method}",
                    {"method": self.verification_method}
                )
            
            # Extract scores
            confidence = evaluation.get("confidence", 0.5)
            relevance = evaluation.get("relevance", 0.5)
            recommendation = evaluation.get("recommendation", "accept")
            feedback = evaluation.get("feedback", "")
            
            # Update message
            message.confidence = confidence
            message.metadata["critic_evaluation"] = evaluation
            message.metadata["critic_recommendation"] = recommendation
            message.provenance.agents_called.append(self.name)
            
            # Determine if re-retrieval should be triggered
            needs_reretrival = (
                self.trigger_reretrival and
                recommendation == "reretrieve" and
                confidence < self.min_confidence and
                relevance < self.min_relevance
            )
            
            message.metadata["needs_reretrival"] = needs_reretrival
            
            self.logger.info(
                "Evaluation completed",
                confidence=f"{confidence:.3f}",
                relevance=f"{relevance:.3f}",
                recommendation=recommendation,
                needs_reretrival=needs_reretrival
            )
            
            if needs_reretrival:
                self.logger.warning(
                    "Answer quality below threshold - re-retrieval recommended",
                    confidence=confidence,
                    min_confidence=self.min_confidence,
                    relevance=relevance,
                    min_relevance=self.min_relevance
                )
            
            return message
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Evaluation failed: {str(e)}",
                {
                    "query": query,
                    "message_id": message.message_id
                }
            )