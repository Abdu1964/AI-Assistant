from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
import logging

logger = logging.getLogger(__name__)

class BioGPTAgent:
    """
    Agent that uses BioGPT to answer common biological questions.
    """

    def __init__(self, model_name="microsoft/biogpt", device=None):
        self.tokenizer = BioGptTokenizer.from_pretrained(model_name)
        self.model = BioGptForCausalLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        logger.info(f"BioGPT loaded on {self.device}")

    def generate_answer(self, query: str, max_length: int = 256) -> str:
        """
        Generate a biological answer to the user query.
        """
        try:
            inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error in BioGPT generation: {str(e)}", exc_info=True)
            return f"Error generating biological answer: {str(e)}"


    def biogpt_agent_function(self, query: str, user_id: str, token: str) -> dict:
        """
        Interface for AiAssistance._biogpt_agent
        """
        try:
            agent = BioGPTAgent()
            answer = agent.generate_answer(query)
        except Exception as e:
            return f"Error generating biological answer"
