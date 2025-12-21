"""
Answer Synthesizer Agent

Generates answers using LLM with retrieved document context.
Supports:
- Context fusion with metadata
- Citation generation
- Multilingual answer synthesis
- Confidence estimation
"""

import os
from typing import List, Optional
from openai import OpenAI

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentMessage,
    RetrievedDocument,
    ValidationError,
    ProcessingError,
)


class AnswerSynthesizerAgent(BaseAgent):
    """
    Answer synthesizer that generates responses using LLM.
    
    Features:
    - Context-aware prompt construction
    - Metadata integration (dates, sources)
    - Citation support
    - Multilingual responses
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Extract configuration
        self.model = config.extra_config.get("model", "gpt-4o")
        self.max_context_chunks = config.extra_config.get("max_context_chunks", 5)
        self.max_context_tokens = config.extra_config.get("max_context_tokens", 4000)
        self.temperature = config.extra_config.get("temperature", 0.3)
        self.max_tokens = config.extra_config.get("max_tokens", 500)
        self.include_metadata = config.extra_config.get("include_metadata", True)
        self.include_citations = config.extra_config.get("include_citations", True)
        self.answer_in_query_language = config.extra_config.get("answer_in_query_language", True)
        
        # Initialize OpenAI client
        api_key_env = config.extra_config.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env)
        
        if not api_key:
            raise ValidationError(
                self.name,
                f"OpenAI API key not found: {api_key_env}",
                {"api_key_env": api_key_env}
            )
        
        self.client = OpenAI(api_key=api_key)
        
        self.logger.info(
            "Answer synthesizer initialized",
            model=self.model,
            max_context_chunks=self.max_context_chunks,
            temperature=self.temperature
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
            self.logger.warning("No context documents provided, will generate answer without context")
        
        return True
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)"""
        return len(text) // 4
    
    def _build_context(self, documents: List[RetrievedDocument], max_tokens: int) -> str:
        """
        Build context from documents with metadata.
        
        Format:
        [1] (Source: ETH News, Date: 2023-05-01)
        Text content here...
        
        [2] (Source: ETH News, Date: 2023-06-15)
        More text here...
        """
        context_parts = []
        total_tokens = 0
        
        for i, doc in enumerate(documents[:self.max_context_chunks], 1):
            # Build document header
            if self.include_metadata:
                metadata_parts = []
                
                if doc.metadata.get("source"):
                    metadata_parts.append(f"Source: {doc.metadata['source']}")
                
                if doc.metadata.get("date"):
                    metadata_parts.append(f"Date: {doc.metadata['date']}")
                
                if doc.metadata.get("language"):
                    metadata_parts.append(f"Language: {doc.metadata['language']}")
                
                header = f"[{i}] ({', '.join(metadata_parts)})" if metadata_parts else f"[{i}]"
            else:
                header = f"[{i}]"
            
            # Build document text
            doc_text = f"{header}\n{doc.text}\n"
            doc_tokens = self._estimate_tokens(doc_text)
            
            # Check if we exceed token limit
            if total_tokens + doc_tokens > max_tokens:
                self.logger.debug(
                    "Context token limit reached",
                    num_docs=i-1,
                    total_tokens=total_tokens
                )
                break
            
            context_parts.append(doc_text)
            total_tokens += doc_tokens
        
        context = "\n".join(context_parts)
        
        self.logger.debug(
            "Context built",
            num_docs=len(context_parts),
            estimated_tokens=total_tokens
        )
        
        return context
    
    def _build_system_prompt(self, language: str) -> str:
        """Build system prompt based on configuration"""
        if language == "de":
            base_prompt = """Du bist ein hilfreicher Assistent, der Fragen zu ETH News-Artikeln beantwortet.

Deine Aufgaben:
1. Beantworte die Frage präzise basierend auf den bereitgestellten Dokumenten
2. Wenn die Antwort nicht in den Dokumenten enthalten ist, sage das ehrlich
3. Verwende nur Informationen aus den bereitgestellten Quellen"""
        else:
            base_prompt = """You are a helpful assistant that answers questions about ETH News articles.

Your tasks:
1. Answer the question precisely based on the provided documents
2. If the answer is not in the documents, say so honestly
3. Only use information from the provided sources"""
        
        if self.include_citations:
            if language == "de":
                base_prompt += "\n4. Zitiere die relevanten Dokumente mit [1], [2], etc."
            else:
                base_prompt += "\n4. Cite relevant documents using [1], [2], etc."
        
        return base_prompt
    
    def _build_user_prompt(self, query: str, context: str, language: str) -> str:
        """Build user prompt with query and context"""
        if context:
            if language == "de":
                prompt = f"""Kontext aus ETH News-Artikeln:

{context}

Frage: {query}

Bitte beantworte die Frage basierend auf dem obigen Kontext."""
            else:
                prompt = f"""Context from ETH News articles:

{context}

Question: {query}

Please answer the question based on the context above."""
        else:
            # No context available
            if language == "de":
                prompt = f"""Frage: {query}

Hinweis: Es wurden keine relevanten Dokumente gefunden. Bitte gib an, dass du die Frage ohne Kontext nicht beantworten kannst."""
            else:
                prompt = f"""Question: {query}

Note: No relevant documents were found. Please indicate that you cannot answer the question without context."""
        
        return prompt
    
    def _extract_confidence(self, answer: str) -> float:
        """
        Estimate confidence from answer text.
        
        Simple heuristics:
        - High confidence: Specific facts, numbers, dates
        - Low confidence: Hedging language, uncertainty markers
        """
        answer_lower = answer.lower()
        
        # Uncertainty markers
        uncertainty_markers = [
            'might', 'maybe', 'possibly', 'perhaps', 'unclear', 'uncertain',
            'not sure', 'cannot determine', 'no information', 'not found',
            'könnte', 'vielleicht', 'möglicherweise', 'unklar', 'nicht sicher'
        ]
        
        # Check for uncertainty
        has_uncertainty = any(marker in answer_lower for marker in uncertainty_markers)
        
        # Check for specific information (numbers, dates)
        import re
        has_numbers = bool(re.search(r'\d+', answer))
        has_dates = bool(re.search(r'\d{4}|\d{1,2}\.\d{1,2}\.\d{4}', answer))
        
        # Heuristic scoring
        if has_uncertainty:
            confidence = 0.3
        elif has_dates or has_numbers:
            confidence = 0.9
        elif len(answer) > 100:  # Detailed answer
            confidence = 0.75
        else:
            confidence = 0.6
        
        return confidence
    
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """Generate answer using LLM"""
        query = message.query
        documents = message.results or []
        language = message.context.language.value if message.context.language else "en"
        
        self.logger.debug(
            "Generating answer",
            query=query[:100],
            num_docs=len(documents),
            language=language
        )
        
        try:
            # Build context from documents
            context = self._build_context(documents, self.max_context_tokens)
            
            # Build prompts
            system_prompt = self._build_system_prompt(language)
            user_prompt = self._build_user_prompt(query, context, language)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract confidence
            confidence = self._extract_confidence(answer)
            
            # Update message
            message.answer = answer
            message.confidence = confidence
            message.metadata["synthesis_model"] = self.model
            message.metadata["synthesis_tokens"] = response.usage.total_tokens
            message.metadata["synthesis_context_docs"] = len([d for d in documents if d.text in context])
            message.provenance.agents_called.append(self.name)
            
            self.logger.info(
                "Answer generated",
                answer_length=len(answer),
                confidence=f"{confidence:.3f}",
                tokens_used=response.usage.total_tokens
            )
            
            return message
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Answer synthesis failed: {str(e)}",
                {
                    "query": query,
                    "message_id": message.message_id,
                    "num_docs": len(documents)
                }
            )