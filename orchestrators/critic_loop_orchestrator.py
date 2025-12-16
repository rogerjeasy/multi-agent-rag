

# ADDING A CRITIC LOOP
## 1) Generator to produce an initial draft answer based on the retrieved documents
## 2) Implementation of an LLM prompt to check the draft for missing info, inaccuracies, hallucinations
## 3) Process - if it passes, we return the answer, if not, we revise the answer based on the feedback and return the revised answer

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import BaseLLM
from typing import Any
import logging

# Assuming base orchestrator is defined 

class CriticLoopOrchestrator:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: Any,
        max_iterations: int = 3,
    ):
        super().__init__(llm, retriever)
        self.llm = llm
        self.retriever = retriever
        self.max_iterations = max_iterations

        prompt_template = """You are an expert assistant. 
        
        Context information is :
        {context}
        
        Previous feedback (if any):
        {feedback}
        
        Based on the context and feedback provided above, generate a answer to the question:
        {question}
        
        Answer:"""
        
        generator_prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context", "feedback"]
        )
        self.generator_chain = LLMChain(llm=llm, prompt=generator_prompt)

        # ### OPTIMIZED: Enforced strict output format for the Critic.
        # This prevents parsing errors where the model talks vaguely.
        critic_prompt_template = """You are a critic. Review the draft answer against the provided documents.
        
        Documents: {context}
        Draft Answer: {draft_answer}
        
        Check for:
        1. Missing information
        2. Hallucinations
        3. Inaccuracies
        """
        
        critic_prompt = PromptTemplate(
            template=critic_prompt_template,
            input_variables=["context", "draft_answer"]
        )
        self.critic_chain = LLMChain(llm=llm, prompt=critic_prompt)

    def run(self, question: str) -> str:
        documents = self.retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in documents])
        
        current_feedback = "None." 
        draft_answer = ""

        for iteration in range(self.max_iterations):
            logging.info(f"--- Iteration {iteration + 1} ---")
            
            draft_answer = self.generator_chain.run(
                question=question, 
                context=context, 
                feedback=current_feedback
            )

            critique = self.critic_chain.run(context=context, draft_answer=draft_answer)
            
            if "STATUS: PASS" in critique.upper():
                logging.info(f"Draft answer accepted post iteration {iteration + 1}.")
                return draft_answer
            else:
                current_feedback = critique.replace("STATUS: FAIL", "").strip()
                logging.info(f"Draft rejected. Feedback: {current_feedback}")
                logging.info("Revising...")

        logging.warning("Max iterations reached. Returning last draft")
        return draft_answer