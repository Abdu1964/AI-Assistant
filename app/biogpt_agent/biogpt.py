from transformers import BioGptTokenizer, BioGptForCausalLM
import torch
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class BioGPTAgent:
    """
    Lazy-loaded BioGPT agent.
    """

    _model = None
    _tokenizer = None
    _device = None
    _lock = Lock()

    def __init__(self, llm=None, model_name="kirubel1738/biogpt-bioqa-lora-merged"):
        self.model_name = model_name
        self.llm = llm  

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
            self._load_if_needed()

            inputs = BioGPTAgent._tokenizer(query, return_tensors="pt").to(BioGPTAgent._device)

            with torch.no_grad():
                output_ids = BioGPTAgent._model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=BioGPTAgent._tokenizer.eos_token_id,
                )

            answer = BioGPTAgent._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # ✅ Remove the question from the start of the answer
            if answer.startswith(query):
                answer = answer[len(query):].strip()
            
            return answer.strip()

        except Exception as e:
            logger.error(f"Error in BioGPT generation: {str(e)}", exc_info=True)
            return f"BIOGPT ERROR: {str(e)}"
  