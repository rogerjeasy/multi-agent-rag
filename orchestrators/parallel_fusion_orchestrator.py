from typing import List
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import base classes and protocols
from orchestrators.base_orchestrator import BaseOrchestrator
from agents.base_agent import AgentMessage, create_agent_message, ProcessingError, RetrievedDocument

# hybrid RAG pipeline with parallel retrieval:
# 1. Query understanding
# 2. Parallel retrieval (BM25,Dense,raphrag)
# 3. Fusion (fusion)
# 4. Reranking (cross-encoder)
# 5. Answer synthesis

class ParallelFusionOrchestrator(BaseOrchestrator):
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load orchestrator-specific settings
        orch_config = self.config.get('orchestrators', {}).get('parallel_fusion', {})
        
        self.parallel_retrieval = orch_config.get('parallel_retrieval', True)
        
        # Define the set of retriever agents to look for
        self.retriever_names = [
            'bm25_retriever', 
            'dense_retriever', 
            'graphrag_retriever'
        ]
        
        # Verify essential agents exist
        if not self.has_agent('synthesizer'):
            self.logger.warning("Synthesizer agent missing! Pipeline will not generate answers.")

    def process_query(self, query: str, language: str = "en", **kwargs) -> AgentMessage:
        self.logger.info(f"Starting Parallel Fusion Pipeline for: {query}")
        
        message = create_agent_message(query=query, language=language)
        
        # 1. Query understanding
        if self.has_agent('query_understanding'):
            self.logger.info("Running Query Understanding...")
            query_agent = self.get_agent('query_understanding')
            message = query_agent.process(message)
            self.logger.info(f"Query processed. Reformulated: {message.context.reformulated_queries}")
        
        # 2. Retrieval
        active_retrievers = [name for name in self.retriever_names if self.has_agent(name)]
        
        if not active_retrievers:
            self.logger.error("No retriever agents found!")
            raise ProcessingError("Orchestrator", "No retrievers available")

        all_results: List[RetrievedDocument] = []
        
        if self.parallel_retrieval and len(active_retrievers) > 1:
            self.logger.info(f"Executing parallel retrieval with: {active_retrievers}")
            
            with ThreadPoolExecutor(max_workers=len(active_retrievers)) as executor:
                future_to_agent = {
                    executor.submit(self._run_agent_safe, name, copy.deepcopy(message)): name 
                    for name in active_retrievers
                }
                
                for future in as_completed(future_to_agent):
                    agent_name = future_to_agent[future]
                    try:
                        result_message = future.result()
                        if result_message and result_message.results:
                            self.logger.info(f"Agent {agent_name} returned {len(result_message.results)} docs")
                            all_results.extend(result_message.results)
                    except Exception as exc:
                        self.logger.error(f"Retriever {agent_name} generated an exception: {exc}")
        else:
            self.logger.info("Executing sequential retrieval")
            for name in active_retrievers:
                agent = self.get_agent(name)
                res_msg = agent.process(copy.deepcopy(message))
                all_results.extend(res_msg.results)

        # Update the main message with aggregated results
        message.results = all_results
        self.logger.info(f"Total documents retrieved before fusion: {len(message.results)}")

        # 3. Fusion 
        if self.has_agent('fusion'):
            self.logger.info("Running Fusion...")
            fusion_agent = self.get_agent('fusion')
            message = fusion_agent.process(message)
            self.logger.info(f"Documents after fusion: {len(message.results)}")
        
        # 4. Reranking 
        if self.has_agent('reranker'):
            self.logger.info("Running Reranker...")
            reranker_agent = self.get_agent('reranker')
            message = reranker_agent.process(message)
        
        # 5. Synthesis
        if self.has_agent('synthesizer'):
            self.logger.info("Synthesizing answer...")
            synthesizer_agent = self.get_agent('synthesizer')
            message = synthesizer_agent.process(message)
        
        self.logger.info("Pipeline completed successfully")
        return message

    def _run_agent_safe(self, agent_name: str, message: AgentMessage) -> AgentMessage:
        """Helper to run an agent and catch errors so one failure doesn't crash the pipeline"""
        agent = self.get_agent(agent_name)
        return agent.process(message)