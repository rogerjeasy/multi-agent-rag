
# DB - Needs to be adapted once answer synthesizer agent is implemented
# ADDING A CRITIC LOOP
## 1) Generator to produce an initial draft answer based on the retrieved documents
## 2) Implementation of an LLM prompt to check the draft for missing info, inaccuracies, hallucinations
## 3) Process - if it passes, we return the answer, if not, we revise the answer based on the feedback and return the revised answer

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from typing import Any
import logging

# Import of orchestrator
from orchestrators.base_orchestrator import BaseOrchestrator
from agents.base_agent import AgentMessage, create_agent_message, ProcessingError

 # Takes config_path, vs llm/retriever directly
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load orchestrator settings from config
        orch_config = self.config.get('orchestrators', {}).get('critic_loop', {})
        if not orch_config.get('enabled', False):
            self.logger.warning("Critic Loop Orchestrator is disabled in config")
        
        self.max_iterations = orch_config.get('max_iterations', 3)
        self.initial_retriever_name = orch_config.get('initial_retriever', 'bm25_retriever')
        
        #Validate that required agents exist
        required_agents = [self.initial_retriever_name, 'synthesizer', 'critic']
        for agent_name in required_agents:
            if not self.has_agent(agent_name):
                self.logger.error(f"Required agent '{agent_name}' not found for CriticLoopOrchestrator")

    # Process_query to match BaseOrchestrator abstract method
    def process_query(self, query: str, language: str = "en", **kwargs) -> AgentMessage:
        self.logger.info(f"Starting Critic Loop for query: {query}")
        
        # 1.Start
        message = create_agent_message(query=query, language=language)
        
        # 2.Retrieve documents
        retriever = self.get_agent(self.initial_retriever_name)
        if not retriever:
            raise ProcessingError("Orchestrator", f"Retriever {self.initial_retriever_name} missing")
            
        message = retriever.process(message)
        self.logger.info(f"Retrieved {len(message.results)} documents")

        # 3. Critique
        current_feedback = "None"
        
        synthesizer = self.get_agent('synthesizer')
        critic = self.get_agent('critic')

        for i in range(self.max_iterations):
            self.logger.info(f"--- Iteration {i + 1} ---")
            
            message.agent_data['previous_feedback'] = current_feedback
            message.agent_data['is_revision'] = (i > 0)
            
            # 1) Create draft
            message = synthesizer.process(message)
            
            # 2) Critique
            message = critic.process(message)
            
            status = message.agent_data.get('critique_status', 'FAIL')
            feedback = message.agent_data.get('critique_feedback', 'No feedback provided.')
            
            if status == "PASS":
                self.logger.info(f"Draft accepted at iteration {i + 1}")
                return message
            else:
                current_feedback = feedback
                self.logger.info(f"Draft rejected. Feedback: {current_feedback}")
        
        self.logger.warning("Max iterations reached. Returning best effort.")
        return message