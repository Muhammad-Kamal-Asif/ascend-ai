import os
from crewai import LLM

class GeminiKeyManager:
    """
    Centralized LLM configuration manager for Gemini.
    """
    def __init__(self):
        # Fetch the Gemini API key
        self.api_key = os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("No GEMINI_API_KEY found in environment.")

    # FIX: Changed default model to the active 2026 version
    def get_llm(self, model="gemini/gemini-2.5-flash", temperature=0.0):
        """
        Retrieves an LLM instance using the Gemini API.
        """
        # --- ZERO BREAKAGE TRANSLATION LAYER ---
        # Intercept legacy Groq requests AND deprecated 1.5 Gemini requests
        if "groq" in model.lower() or "70b" in model.lower():
            print(f"[SYSTEM] Redirecting legacy heavy model request ({model}) to Gemini 2.5 Pro...")
            model = "gemini/gemini-2.5-pro"
        elif "1.5-flash" in model.lower():
            print(f"[SYSTEM] Upgrading deprecated 1.5 model to 2.5 Flash...")
            model = "gemini/gemini-2.5-flash"

        return LLM(
            model=model,
            temperature=temperature,
            api_key=self.api_key 
        )

# Keep the exact same instance name 'key_manager' so imports don't break!
key_manager = GeminiKeyManager()