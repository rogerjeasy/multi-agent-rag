from typing import List, Dict, Optional
import copy
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from orchestrators.base_orchestrator import BaseOrchestrator
from agents.base_agent import (
    AgentMessage, 
    create_agent_message, 
    ProcessingError, 
    RetrievedDocument
)

class ParallelFusionOrchestrator(BaseOrchestrator):
    def __init__(self, config_path: str):
        # Initialize BaseOrchestrator 
        super().__init__(config_path)
        
        # Load orchestrator settings from the config
        orch_config = self.config.get('orchestrators', {}).get('parallel_fusion', {})
        
        self.parallel_retrieval = orch_config.get('parallel_retrieval', True)
        # Toggle for using expanded queries from QueryUnderstandingAgent
        self.use_query_expansion = orch_config.get('use_query_expansion', False) 
        
        # Define the set of retriever agents to look for
        self.retriever_names = [
            'bm25_retriever', 
            'dense_retriever', 
            'graphrag_retriever'
        ]
        
        # Verify essential agents exist (BaseOrchestrator initializes them)
        if not self.has_agent('synthesizer'):
            self.logger.warning("Synthesizer agent missing! Pipeline will not generate answers.")

    def process_query(self, query: str, language: str = "en", **kwargs) -> AgentMessage:
        """
        Implementation of the abstract process_query method.
        Executes the parallel RAG pipeline.
        """
        self.logger.info(f"Starting Parallel Fusion Pipeline for: {query}")
        
        # Initialize protocol
        message = create_agent_message(query=query, language=language)
        
        # 1. Query understanding
        if self.has_agent('query_understanding'):
            self.logger.info("Running Query Understanding...")
            query_agent = self.get_agent('query_understanding')
            message = query_agent.process(message)
            self.logger.info(f"Query Context: {message.context.query_type}, Expansions: {message.context.reformulated_queries}")
        
        # 2. Retrieval
        active_retrievers = [name for name in self.retriever_names if self.has_agent(name)]
        
        if not active_retrievers:
            self.logger.error("No retriever agents found!")
            raise ProcessingError("Orchestrator", "No retrievers available")

        queries_to_run = [message.query]
        if self.use_query_expansion and message.context.reformulated_queries:
            queries_to_run.extend(message.context.reformulated_queries[:2])

        all_results: List[RetrievedDocument] = []
        
        # 3. Sequential or parallel execution  
        # Create a list of tasks
        tasks = []
        for q in queries_to_run:
            for name in active_retrievers:
                tasks.append((name, q))

        self.logger.info(f"Preparing {len(tasks)} retrieval tasks (Parallel={self.parallel_retrieval})")

        if self.parallel_retrieval and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                future_to_task = {}
                for agent_name, query_text in tasks:
                    # Create a deep copy for each thread to race conditions
                    task_msg = copy.deepcopy(message)
                    task_msg.query = query_text 
                    
                    future = executor.submit(self._run_agent_safe, agent_name, task_msg)
                    future_to_task[future] = f"{agent_name} [{query_text[:15]}...]"
                
                for future in as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        result_msg = future.result()
                        if result_msg and result_msg.results:
                            self.logger.info(f"Task {task_name} returned {len(result_msg.results)} docs")
                            all_results.extend(result_msg.results)
                            
                            self._merge_provenance(message, result_msg)
                    except Exception as exc:
                        self.logger.error(f"Task {task_name} generated an exception: {exc}")
        else:
            for agent_name, query_text in tasks:
                self.logger.info(f"Running {agent_name} sequentially on query: {query_text}")
                task_msg = copy.deepcopy(message)
                task_msg.query = query_text
                
                agent = self.get_agent(agent_name)
                res_msg = agent.process(task_msg)
                
                if res_msg and res_msg.results:
                    all_results.extend(res_msg.results)
                    self._merge_provenance(message, res_msg)

        message.results = all_results
        self.logger.info(f"Total documents retrieved: {len(message.results)}")

        # SAFEGUARD: Check for empty results before Fusion
        if not message.results:
            self.logger.warning("No documents retrieved! Skipping Fusion and Reranking.")
            if self.has_agent('synthesizer'):
                 synthesizer_agent = self.get_agent('synthesizer')
                 message = synthesizer_agent.process(message)
            return message

        # 4. Fusion 
        if self.has_agent('fusion'):
            self.logger.info("Running Fusion...")
            fusion_agent = self.get_agent('fusion')
            message = fusion_agent.process(message)
            self.logger.info(f"Documents after fusion: {len(message.results)}")
        
        # 5. Reranking 
        if self.has_agent('reranker'):
            self.logger.info("Running Reranker...")
            reranker_agent = self.get_agent('reranker')
            message = reranker_agent.process(message)
        
        # 6. Synthesis
        if self.has_agent('synthesizer'):
            self.logger.info("Synthesizing answer...")
            synthesizer_agent = self.get_agent('synthesizer')
            message = synthesizer_agent.process(message)
        
        self.logger.info(f"Pipeline completed. Total time: {message.provenance.execution_time_ms:.2f}ms")
        return message

    def _run_agent_safe(self, agent_name: str, message: AgentMessage) -> AgentMessage:

        # Retrieve the agent instance 
        agent = self.get_agent(agent_name)
        return agent.process(message)

    def _merge_provenance(self, main_msg: AgentMessage, task_msg: AgentMessage):
        # Merge agents called list
        for agent in task_msg.provenance.agents_called:
            if agent not in main_msg.provenance.agents_called:
                main_msg.provenance.agents_called.append(agent)
        
        # Merge retrieval methods list
        for method in task_msg.provenance.retrieval_methods:
            if method not in main_msg.provenance.retrieval_methods:
                main_msg.provenance.retrieval_methods.append(method)
        
        # Add execution time
        main_msg.provenance.execution_time_ms += task_msg.provenance.execution_time_ms
        
        # Merge model calls counts
        for model, count in task_msg.provenance.model_calls.items():
            main_msg.provenance.model_calls[model] = main_msg.provenance.model_calls.get(model, 0) + count