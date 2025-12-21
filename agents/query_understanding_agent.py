"""
Query Understanding Agent

Parses and enriches user queries with:
- Language detection
- Query classification (factual, temporal, entity-based, etc.)
- Named entity recognition
- Keyword extraction
- Query expansion
"""

import os
from typing import List, Optional, Dict
from langdetect import detect
import re

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentMessage,
    ValidationError,
    ProcessingError,
    Language,
    QueryType,
)


class QueryUnderstandingAgent(BaseAgent):
    """
    Query understanding agent that enriches queries with metadata.
    
    Provides:
    - Language detection
    - Query type classification
    - Entity extraction
    - Keyword extraction
    - Query expansion (optional)
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Extract configuration
        self.spacy_model_en = config.extra_config.get("spacy_model_en", "en_core_web_sm")
        self.spacy_model_de = config.extra_config.get("spacy_model_de", "de_core_news_sm")
        self.expand_queries = config.extra_config.get("expand_queries", True)
        self.max_expansions = config.extra_config.get("max_expansions", 3)
        
        # Initialize spaCy models
        self._init_spacy_models()
        
        self.logger.info(
            "Query understanding agent initialized",
            expand_queries=self.expand_queries,
            max_expansions=self.max_expansions
        )
    
    def _init_spacy_models(self):
        """Initialize spaCy models for NER"""
        try:
            import spacy
        except ImportError:
            raise ValidationError(
                self.name,
                "spaCy not installed. Run: pip install spacy",
                {}
            )
        
        try:
            self.nlp_en = spacy.load(self.spacy_model_en)
            self.logger.debug(f"Loaded spaCy model: {self.spacy_model_en}")
        except:
            self.logger.warning(f"Could not load {self.spacy_model_en}, NER disabled for EN")
            self.nlp_en = None
        
        try:
            self.nlp_de = spacy.load(self.spacy_model_de)
            self.logger.debug(f"Loaded spaCy model: {self.spacy_model_de}")
        except:
            self.logger.warning(f"Could not load {self.spacy_model_de}, NER disabled for DE")
            self.nlp_de = None
    
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
            # Map to supported languages
            if lang not in ("en", "de"):
                self.logger.debug(f"Detected language {lang}, defaulting to 'en'")
                lang = "en"
        except:
            self.logger.warning("Language detection failed, defaulting to 'en'")
            lang = "en"
        
        return lang
    
    def _classify_query_type(self, query: str) -> str:
        """
        Classify query type based on patterns.
        
        Types:
        - factual: Who/What/When/Where questions
        - temporal: Time-related queries
        - entity: About specific entities
        - comparison: Comparing entities
        - procedural: How-to questions
        - other: Default
        """
        query_lower = query.lower()
        
        # Temporal patterns
        temporal_keywords = ['when', 'wann', 'date', 'datum', 'year', 'jahr', 'time', 'zeit']
        if any(kw in query_lower for kw in temporal_keywords):
            return "temporal"
        
        # Entity/person patterns
        if re.search(r'\b(who|wer|president|präsident|ceo|director|professor)\b', query_lower):
            return "entity"
        
        # Comparison patterns
        if re.search(r'\b(versus|vs|compared|compare|vergleich|unterschied)\b', query_lower):
            return "comparison"
        
        # Procedural patterns
        if re.search(r'\b(how|wie|steps|schritte|process|prozess)\b', query_lower):
            return "procedural"
        
        # Factual patterns (what, where, why)
        if re.search(r'\b(what|was|where|wo|why|warum|which|welche)\b', query_lower):
            return "factual"
        
        return "other"
    
    def _extract_entities(self, query: str, language: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy"""
        # Select appropriate model
        if language == "en" and self.nlp_en:
            nlp = self.nlp_en
        elif language == "de" and self.nlp_de:
            nlp = self.nlp_de
        else:
            self.logger.debug("No spaCy model available for entity extraction")
            return []
        
        try:
            doc = nlp(query)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from query.
        
        Simple approach: remove stopwords, keep important terms.
        """
        # Basic stopwords
        stopwords_en = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over',
            'under', 'again', 'further', 'then', 'once'
        }
        
        stopwords_de = {
            'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einer', 'eines',
            'und', 'oder', 'aber', 'in', 'auf', 'zu', 'von', 'mit', 'bei', 'nach',
            'für', 'als', 'ist', 'war', 'sind', 'waren', 'sein', 'haben', 'hat',
            'hatte', 'hatten', 'wird', 'wurde', 'werden', 'kann', 'könnte', 'über',
            'unter', 'durch', 'während', 'vor', 'nach'
        }
        
        stopwords = stopwords_en | stopwords_de
        
        # Tokenize
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Filter stopwords and short words
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    def _expand_query(self, query: str, keywords: List[str], entities: List[Dict]) -> List[str]:
        """
        Generate query expansions.
        
        Simple approach:
        - Add synonyms or related terms
        - Rephrase with entities
        - Add context
        """
        expansions = []
        
        # Original query
        expansions.append(query)
        
        # If we have entities, create entity-focused queries
        if entities and len(expansions) < self.max_expansions:
            entity_texts = [e['text'] for e in entities]
            if entity_texts:
                entity_query = f"information about {' and '.join(entity_texts)}"
                expansions.append(entity_query)
        
        # Keyword-based expansion
        if keywords and len(expansions) < self.max_expansions:
            keyword_query = ' '.join(keywords[:5])  # Top 5 keywords
            if keyword_query != query.lower():
                expansions.append(keyword_query)
        
        return expansions[:self.max_expansions]
    
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """Parse and enrich query"""
        query = message.query
        
        self.logger.debug(
            "Processing query understanding",
            query=query[:100]
        )
        
        try:
            # 1. Detect language
            detected_lang = self._detect_language(query)
            message.context.language = Language(detected_lang)
            
            # 2. Classify query type
            query_type = self._classify_query_type(query)
            message.context.query_type = QueryType(query_type)
            
            # 3. Extract entities
            entities = self._extract_entities(query, detected_lang)
            message.context.entities = [e['text'] for e in entities]
            
            # 4. Extract keywords
            keywords = self._extract_keywords(query)
            message.context.keywords = keywords
            
            # 5. Query expansion (optional)
            if self.expand_queries:
                expansions = self._expand_query(query, keywords, entities)
                message.context.query_expansions = expansions
            
            # 6. Store detailed entity info in metadata
            if entities:
                message.metadata["entity_details"] = entities
            
            # Update provenance
            message.provenance.agents_called.append(self.name)
            
            self.logger.info(
                "Query understanding completed",
                language=detected_lang,
                query_type=query_type,
                num_entities=len(entities),
                num_keywords=len(keywords),
                num_expansions=len(message.context.query_expansions) if message.context.query_expansions else 0
            )
            
            self.logger.debug(
                "Query enrichment details",
                entities=[e['text'] for e in entities] if entities else [],
                keywords=keywords,
                expansions=message.context.query_expansions if message.context.query_expansions else []
            )
            
            return message
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Query understanding failed: {str(e)}",
                {
                    "query": query,
                    "message_id": message.message_id
                }
            )