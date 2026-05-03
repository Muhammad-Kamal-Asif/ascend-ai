import os
from itertools import cycle
from crewai import LLM

class GroqKeyManager:
    """
    Round-robin key rotator that injects keys at LLM creation time.
    """
    def __init__(self):
        raw_keys = [
            os.environ.get("GROQ_API_KEY"),
            os.environ.get("GROQ_API_KEY_2"),
            os.environ.get("GROQ_API_KEY_3"),
            os.environ.get("GROQ_API_KEY_4"),
        ]
        self.keys = [k for k in raw_keys if k]
        
        if not self.keys:
            raise ValueError("No GROQ API keys found in environment.")
            
        self._cycle = cycle(self.keys)
        self._call_count = 0
        self._rotate_every = 3  # Rotate key after every 3 LLM instantiations

    def get_llm(self, model="groq/llama-3.1-8b-instant", temperature=0.0):
        """
        Retrieves an LLM instance with a freshly rotated API key.
        """
        self._call_count += 1
        
        if self._call_count % self._rotate_every == 0:
            key = next(self._cycle)
        else:
            # Peek at current key without advancing the cycle
            key = next(self._cycle)
            # Rebuild cycle to put the key back in place
            current_index = self.keys.index(key)
            self._cycle = cycle(self.keys[current_index:] + self.keys[:current_index])
        
        return LLM(
            model=model,
            temperature=temperature,
            api_key=key 
        )

# Singleton instance to be imported across your modules
key_manager = GroqKeyManager()