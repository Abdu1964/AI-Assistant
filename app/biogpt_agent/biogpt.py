from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class BioGPTAgent:
    """
    Lazy-loaded BioGPT agent.
    Loads model/tokenizer ONLY when needed (Claude-style implementation).
    """

    _model = None
    _tokenizer = None
    _device = None
    _lock = Lock()

    def __init__(self, llm=None, model_name="microsoft/biogpt"):
        self.model_name = model_name
        self.llm = llm  # just stored, not used

    def _load_if_needed(self):
        """Load model/tokenizer once lazily."""
        if BioGPTAgent._model is None:
            with BioGPTAgent._lock:
                # double-check inside lock
                if BioGPTAgent._model is None:
                    logger.info("Lazy-loading BioGPT model...")

                    BioGPTAgent._tokenizer = BioGptTokenizer.from_pretrained(self.model_name)
                    BioGPTAgent._model = BioGptForCausalLM.from_pretrained(self.model_name)

                    BioGPTAgent._device = "cuda" if torch.cuda.is_available() else "cpu"
                    BioGPTAgent._model.to(BioGPTAgent._device)
                    BioGPTAgent._model.eval()  # Set to evaluation mode

                    logger.info(f"BioGPT loaded on {BioGPTAgent._device}")

    def generate_answer(self, query: str, max_length: int = 256) -> str:
        try:
            self._load_if_needed()  # Lazy load happens here

            inputs = BioGPTAgent._tokenizer(query, return_tensors="pt").to(BioGPTAgent._device)

            # Use no_grad for inference to save memory
            with torch.no_grad():
                output_ids = BioGPTAgent._model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=BioGPTAgent._tokenizer.eos_token_id,  # Fix padding warning
                )

            answer = BioGPTAgent._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return answer.strip()

        except Exception as e:
            logger.error(f"Error in BioGPT generation: {str(e)}", exc_info=True)
            return f"BIOGPT ERROR: {str(e)}"

    def biogpt_agent_function(self, query: str, user_id: str = None, token: str = None) -> dict:
        """
        Wrapper function for BioGPT agent.
        
        Args:
            query: The input query
            user_id: Optional user identifier
            token: Optional authentication token
            
        Returns:
            Dictionary with 'answer' or 'error' key
        """
        try:
            answer = self.generate_answer(query)
            return {"answer": answer}
        except Exception as e:
            logger.error(f"Error in biogpt_agent_function: {str(e)}", exc_info=True)
            return {"error": str(e)}