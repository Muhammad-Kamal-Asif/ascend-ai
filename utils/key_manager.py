import os
from crewai import LLM

class GroqKeyManager:
    """
    Centralized LLM configuration manager.
    Currently configured for a single Groq key, but structured to easily 
    accommodate Gemini or other providers in the future.
    """
    def __init__(self):
        # Fetch only the primary API key
        self.api_key = os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("No GROQ_API_KEY found in environment.")

    def get_llm(self, model="groq/llama-3.1-8b-instant", temperature=0.0):
        """
        Retrieves an LLM instance using the primary API key.
        """
        return LLM(
            model=model,
            temperature=temperature,
            api_key=self.api_key 
        )

# Singleton instance to be imported across your modules
key_manager = GroqKeyManager()