"""
GraphRAG Retriever Agent

Wraps the GraphRAG retrieval system from Step 2.1 into the agent framework.
Uses entity graphs, community detection, and LLM summaries for context-aware retrieval.
"""

import json
import pickle
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from sentence_transformers import SentenceTransformer
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


class GraphRAGRetrieverAgent(BaseAgent):
    """
    GraphRAG retrieval agent using entity graphs and community detection.
    
    Retrieves documents by:
    1. Finding relevant communities via embedding similarity
    2. Collecting member chunks from those communities
    3. Ranking chunks by cosine similarity to query
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        
        # Extract configuration
        graph_path = config.extra_config.get("graph_path")
        entity_index_path = config.extra_config.get("entity_index_path")
        
        if not graph_path or not entity_index_path:
            raise ValidationError(
                self.name,
                "graph_path and entity_index_path required in config",
                {"extra_config": config.extra_config}
            )
        
        self.graph_path = Path(graph_path)
        self.entity_index_path = Path(entity_index_path)
        self.top_k = config.extra_config.get("top_k", 20)
        self.level = config.extra_config.get("level", "C1")  # C0, C1, or C2
        self.k_comms = config.extra_config.get("k_comms", 24)
        self.max_hops = config.extra_config.get("max_hops", 2)
        self.entity_threshold = config.extra_config.get("entity_threshold", 0.7)
        
        # Load GraphRAG components
        self._load_graphrag_index()
        
        self.logger.info(
            "GraphRAG index loaded",
            graph_path=str(self.graph_path),
            level=self.level,
            top_k=self.top_k
        )
    
    def _load_graphrag_index(self):
        """Load GraphRAG embeddings, community mappings, and chunks"""
        try:
            # Load embeddings for community summaries
            L = int(self.level.lstrip("C"))
            emb_file = self.entity_index_path / f"EMB_fixed_C{L}.npy"
            cid_file = self.entity_index_path / f"CID_fixed_C{L}.json"
            
            if not emb_file.exists() or not cid_file.exists():
                raise FileNotFoundError(
                    f"Missing embedding files: {emb_file} or {cid_file}"
                )
            
            self.community_embeddings = np.load(emb_file)
            with open(cid_file, 'r') as f:
                self.community_ids = json.load(f)
            
            # Load community-to-chunk mapping
            comm2chunk_file = self.entity_index_path / "comm2chunk_fixed.json"
            if not comm2chunk_file.exists():
                raise FileNotFoundError(f"Missing file: {comm2chunk_file}")
            
            with open(comm2chunk_file, 'r') as f:
                self.comm2chunk = json.load(f)
            
            # Load chunk documents
            chunks_pickle = self.graph_path / "docs_fixed_norm.pkl"
            if not chunks_pickle.exists():
                raise FileNotFoundError(f"Missing file: {chunks_pickle}")
            
            with open(chunks_pickle, 'rb') as f:
                docs_list = pickle.load(f)
            
            # Build chunk lookup dictionary
            self.chunk_by_id = {}
            for doc in docs_list:
                chunk_id = doc.metadata.get("chunk_id") or doc.metadata.get("record_id")
                if chunk_id:
                    self.chunk_by_id[chunk_id] = doc
            
            # Initialize embedder for chunk-level similarity
            self.embedder = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2',
                device=config.device
            )
            
            # Cache for chunk embeddings
            self.chunk_vec_cache: Dict[str, np.ndarray] = {}
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"Failed to load GraphRAG index: {str(e)}",
                {
                    "graph_path": str(self.graph_path),
                    "entity_index_path": str(self.entity_index_path)
                }
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
    
    def _get_chunk_embedding(self, chunk_id: str) -> np.ndarray:
        """Get or compute chunk embedding (cached)"""
        if chunk_id not in self.chunk_vec_cache:
            doc = self.chunk_by_id.get(chunk_id)
            if not doc:
                return None
            
            text = doc.page_content
            embedding = self.embedder.encode(
                [text],
                normalize_embeddings=True
            )[0]
            self.chunk_vec_cache[chunk_id] = embedding
        
        return self.chunk_vec_cache[chunk_id]
    
    def _process_impl(self, message: AgentMessage) -> AgentMessage:
        """Execute GraphRAG retrieval"""
        query = message.query
        
        # Detect language and update context
        detected_lang = self._detect_language(query)
        message.context.language = Language(detected_lang)
        
        self.logger.debug(
            "Executing GraphRAG search",
            query=query,
            language=detected_lang,
            level=self.level,
            k_comms=self.k_comms,
            top_k=self.top_k
        )
        
        try:
            # 1. Encode query
            q_vec = self.embedder.encode(
                [query],
                normalize_embeddings=True
            )[0]
            
            # 2. Find k most similar communities
            sims_comm = self.community_embeddings @ q_vec
            best_comm_idx = sims_comm.argsort()[::-1][:self.k_comms]
            
            self.logger.debug(
                "Found relevant communities",
                num_communities=len(best_comm_idx),
                top_score=float(sims_comm[best_comm_idx[0]]) if len(best_comm_idx) > 0 else 0.0
            )
            
            # 3. Collect all member chunks from those communities
            cand_chunks = set()
            for idx in best_comm_idx:
                comm_id = self.community_ids[idx]
                member_chunks = self.comm2chunk.get(comm_id, [])
                cand_chunks.update(member_chunks)
            
            if not cand_chunks:
                self.logger.warning("No candidate chunks found")
                message.results = []
                message.provenance.retrieval_methods.append("graphrag")
                return message
            
            # 4. Rank candidate chunks by chunk-level cosine similarity
            scored = []
            for chunk_id in cand_chunks:
                chunk_vec = self._get_chunk_embedding(chunk_id)
                if chunk_vec is not None:
                    sim = float(chunk_vec @ q_vec)  # [-1, 1]
                    scored.append((chunk_id, sim))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            scored = scored[:self.top_k]
            
            # 5. Convert to RetrievedDocument format
            documents = []
            for chunk_id, sim in scored:
                lc_doc = self.chunk_by_id.get(chunk_id)
                if not lc_doc:
                    continue
                
                # Normalize cosine similarity from [-1,1] to [0,1]
                normalized_score = (sim + 1) / 2
                
                meta = dict(lc_doc.metadata)
                meta["grag_score"] = normalized_score
                
                doc = RetrievedDocument(
                    chunk_id=chunk_id,
                    doc_id=meta.get("doc_id", "unknown"),
                    text=lc_doc.page_content,
                    score=normalized_score,
                    source="graphrag",
                    metadata=meta,
                    language=meta.get("language"),
                    date=meta.get("date"),
                    entities=meta.get("entities"),
                    keywords=meta.get("keywords") or meta.get("topic_tags")
                )
                documents.append(doc)
            
            # Update message
            message.results = documents
            message.provenance.retrieval_methods.append("graphrag")
            
            self.logger.info(
                "GraphRAG retrieval completed",
                num_results=len(documents),
                num_candidates=len(cand_chunks),
                avg_score=sum(d.score for d in documents) / len(documents) if documents else 0.0
            )
            
            return message
            
        except Exception as e:
            raise ProcessingError(
                self.name,
                f"GraphRAG search failed: {str(e)}",
                {
                    "query": query,
                    "message_id": message.message_id
                }
            )