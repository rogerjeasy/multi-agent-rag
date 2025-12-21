"""
Fusion Agent

Merges results from multiple retrieval agents using various fusion strategies:
- RRF (Reciprocal Rank Fusion)
- Weighted scoring
- Z-score normalization
"""

from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

from agents.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentMessage,
    RetrievedDocument,
    ValidationError,
    ProcessingError,
)


class FusionAgent(BaseAgent):
    """
    Fusion agent that merges retrieval results from multiple sources.
    
    Supports three fusion strategies:
    - RRF: Reciprocal Rank Fusion (default)
    - weighted: Weighted score combination
    - zscore: Z-score normalization then weighted
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Extract configuration
        self.strategy = config.extra_config.get("strategy", "rrf")
        self.weights = config.extra_config.get("weights", {
            "bm25": 0.3,
            "dense": 0.1,
            "graphrag": 0.6
        })
        self.rrf_k = config.extra_config.get("rrf_k", 60)
        self.deduplicate = config.extra_config.get("deduplicate", True)
        self.top_k = config.extra_config.get("top_k", 20)
        
        # Validate weights sum to 1.0
        if self.strategy == "weighted" or self.strategy == "zscore":
            weight_sum = sum(self.weights.values())
            if not np.isclose(weight_sum, 1.0, atol=0.01):
                self.logger.warning(
                    f"Weights sum to {weight_sum:.3f}, normalizing to 1.0"
                )
                total = sum(self.weights.values())
                self.weights = {k: v/total for k, v in self.weights.items()}
        
        self.logger.info(
            "Fusion agent initialized",
            strategy=self.strategy,
            weights=self.weights,
            rrf_k=self.rrf_k if self.strategy == "rrf" else None
        )
    
    def validate_input(self, message: AgentMessage) -> bool:
        """Validate that message contains retrieval results"""
        if not message.results or len(message.results) == 0:
            raise ValidationError(
                self.name,
                "No retrieval results to fuse",
                {"num_results": len(message.results) if message.results else 0}
            )
        
        return True
    
    def _group_by_source(self, documents: List[RetrievedDocument]) -> Dict[str, List[RetrievedDocument]]:
        """Group documents by retrieval source"""
        grouped = defaultdict(list)
        for doc in documents:
            grouped[doc.source].append(doc)
        return dict(grouped)
    
    def _rrf_fusion(self, grouped_docs: Dict[str, List[RetrievedDocument]]) -> List[RetrievedDocument]:
        """
        Reciprocal Rank Fusion
        
        RRF_score(doc) = sum over sources of: 1 / (k + rank_in_source)
        """
        # Build rank mapping
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for source, docs in grouped_docs.items():
            for rank, doc in enumerate(docs, start=1):
                rrf_score = 1.0 / (self.rrf_k + rank)
                doc_scores[doc.chunk_id] += rrf_score
                
                # Keep track of document object (use first occurrence)
                if doc.chunk_id not in doc_objects:
                    doc_objects[doc.chunk_id] = doc
        
        # Sort by RRF score
        sorted_chunks = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final list with updated scores
        fused = []
        for chunk_id, rrf_score in sorted_chunks[:self.top_k]:
            doc = doc_objects[chunk_id]
            # Update metadata with fusion info
            doc.metadata["rrf_score"] = rrf_score
            doc.metadata["fusion_rank"] = len(fused) + 1
            doc.score = rrf_score  # Update main score
            fused.append(doc)
        
        return fused
    
    def _weighted_fusion(self, grouped_docs: Dict[str, List[RetrievedDocument]]) -> List[RetrievedDocument]:
        """
        Weighted score fusion
        
        Final_score(doc) = sum over sources of: weight[source] * normalized_score
        """
        # Collect all documents with weighted scores
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for source, docs in grouped_docs.items():
            weight = self.weights.get(source, 0.0)
            
            for doc in docs:
                # Normalize score to [0, 1] if needed
                normalized_score = doc.score
                if normalized_score > 1.0:
                    normalized_score = normalized_score / 100.0  # Assume 0-100 scale
                
                weighted_score = weight * normalized_score
                doc_scores[doc.chunk_id] += weighted_score
                
                if doc.chunk_id not in doc_objects:
                    doc_objects[doc.chunk_id] = doc
        
        # Sort by weighted score
        sorted_chunks = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final list
        fused = []
        for chunk_id, weighted_score in sorted_chunks[:self.top_k]:
            doc = doc_objects[chunk_id]
            doc.metadata["weighted_score"] = weighted_score
            doc.metadata["fusion_rank"] = len(fused) + 1
            doc.score = weighted_score
            fused.append(doc)
        
        return fused
    
    def _zscore_fusion(self, grouped_docs: Dict[str, List[RetrievedDocument]]) -> List[RetrievedDocument]:
        """
        Z-score normalization + weighted fusion
        
        1. Normalize scores within each source using z-scores
        2. Apply weighted combination
        """
        # Calculate z-scores per source
        doc_scores = defaultdict(float)
        doc_objects = {}
        
        for source, docs in grouped_docs.items():
            if len(docs) == 0:
                continue
            
            # Extract scores
            scores = np.array([d.score for d in docs])
            
            # Calculate z-scores
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            if std_score == 0:
                std_score = 1.0  # Avoid division by zero
            
            weight = self.weights.get(source, 0.0)
            
            for doc, score in zip(docs, scores):
                z_score = (score - mean_score) / std_score
                weighted_z = weight * z_score
                
                doc_scores[doc.chunk_id] += weighted_z
                
                if doc.chunk_id not in doc_objects:
                    doc_objects[doc.chunk_id] = doc
        
        # Sort by weighted z-score
        sorted_chunks = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final list
        fused = []
        for chunk_id, zscore in sorted_chunks[:self.top_k]:
            doc = doc_objects[chunk_id]
            doc.metadata["zscore"] = zscore
            doc.metadata["fusion_rank"] = len(fused) + 1
            doc.score = zscore
            fused.append(doc)
        
        return fused
    
    def _deduplicate_results(self, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """Remove duplicate chunks (keep highest scored)"""
        seen = set()
        deduped = []
        
        for doc in documents:
            if doc.chunk_id not in seen:
                seen.add(doc.chunk_id)
                deduped.append(doc)
        
        return deduped
    
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """Execute fusion strategy"""
        documents = message.results
        
        self.logger.debug(
            "Executing fusion",
            strategy=self.strategy,
            num_input_docs=len(documents)
        )
        
        try:
            # Group by source
            grouped = self._group_by_source(documents)
            
            self.logger.debug(
                "Documents grouped by source",
                sources={source: len(docs) for source, docs in grouped.items()}
            )
            
            # Apply fusion strategy
            if self.strategy == "rrf":
                fused = self._rrf_fusion(grouped)
            elif self.strategy == "weighted":
                fused = self._weighted_fusion(grouped)
            elif self.strategy == "zscore":
                fused = self._zscore_fusion(grouped)
            else:
                raise ValidationError(
                    self.name,
                    f"Unknown fusion strategy: {self.strategy}",
                    {"strategy": self.strategy}
                )
            
            # Optional deduplication
            if self.deduplicate:
                before_dedup = len(fused)
                fused = self._deduplicate_results(fused)
                
                if len(fused) < before_dedup:
                    self.logger.debug(
                        "Deduplication applied",
                        before=before_dedup,
                        after=len(fused),
                        removed=before_dedup - len(fused)
                    )
            
            # Update message
            message.results = fused
            message.provenance.retrieval_methods.append(f"fusion_{self.strategy}")
            
            self.logger.info(
                "Fusion completed",
                strategy=self.strategy,
                input_docs=len(documents),
                output_docs=len(fused),
                avg_score=sum(d.score for d in fused) / len(fused) if fused else 0.0
            )
            
            return message
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Fusion failed: {str(e)}",
                {
                    "strategy": self.strategy,
                    "num_docs": len(documents),
                    "message_id": message.message_id
                }
            )