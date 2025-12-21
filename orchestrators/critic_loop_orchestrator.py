

# DB - with typing imports and wo langchain imports
from typing import Optional, Dict, Any
import logging

# Import the new BaseOrchestrator and message protocol
from orchestrators.base_orchestrator import BaseOrchestrator
from agents.base_agent import AgentMessage, create_agent_message, ProcessingError

# Class definition
class CriticLoopOrchestrator(BaseOrchestrator):

    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load settings from self.config
        orch_config = self.config.get('orchestrators', {}).get('critic_loop', {})
        if not orch_config.get('enabled', False):
            self.logger.warning("Critic Loop Orchestrator is disabled in config")
        
        self.max_iterations = orch_config.get('max_iterations', 3)
        self.initial_retriever_name = orch_config.get('initial_retriever', 'bm25_retriever')
        
        # Validate agents are loaded
        required_agents = [self.initial_retriever_name, 'synthesizer', 'critic']
        for agent_name in required_agents:
            if not self.has_agent(agent_name):
                self.logger.error(f"Required agent '{agent_name}' not found for CriticLoopOrchestrator")

    def process_query(self, query: str, language: str = "en", **kwargs) -> AgentMessage:
        self.logger.info(f"Starting Critic Loop for query: {query}")
        
        # Using AgentMessage protocol
        message = create_agent_message(query=query, language=language)
        
        # 1. Retrieves documents
        retriever = self.get_agent(self.initial_retriever_name)
        if not retriever:
            raise ProcessingError("Orchestrator", f"Retriever {self.initial_retriever_name} missing")
            
        message = retriever.process(message)
        self.logger.info(f"Retrieved {len(message.results)} documents")

        # 2. Critique loop
        current_feedback = "None"
        
        # Uses ynthesizer and critic agents
        synthesizer = self.get_agent('synthesizer')
        critic = self.get_agent('critic')

        for i in range(self.max_iterations):
            self.logger.info(f"--- Iteration {i + 1} ---")
            
            # Pass context via message.agent_data
            message.agent_data['previous_feedback'] = current_feedback
            message.agent_data['is_revision'] = (i > 0)
            
            # A) Synthesizer agent creates draft
            message = synthesizer.process(message)
            
            # B) Critic Agent reviews draft
            message = critic.process(message)
            
            # Reads structured output from Critic agent
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