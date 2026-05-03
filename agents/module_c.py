import time
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# --- ENTERPRISE RETRY WRAPPER ---
def safe_kickoff(crew_instance, task_name):
    """Wraps the Crew execution to handle rate limits AND empty responses."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return crew_instance.kickoff()
        except Exception as e:
            error_msg = str(e).lower()
            # If the agent runs out of time/iterations, return a fallback string
            if "invalid response from llm call" in error_msg:
                print(f"\n[⚠️ NOTICE] {task_name} agent reached max iterations. Providing partial result.")
                return type('obj', (object,), {'raw': "Information gathering incomplete due to model constraints, but the process is continuing..."})
            
            if "rate limit" in error_msg or "429" in error_msg or "too large" in error_msg:
                print(f"\n[🚨 API LIMIT HIT] Pausing for 30s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(30)
            else:
                raise e
    return type('obj', (object,), {'raw': f"{task_name} failed after retries."})

# --- EXPLICITLY NAMED TOOL TO PREVENT HALLUCINATIONS ---
@tool("internet_search_tool")
def internet_search_tool(query: str) -> str:
    """Search the internet for current information. You MUST use this exact tool name."""
    return DuckDuckGoSearchRun().run(query)

def run_module_c(profile):
    
    # Using the fast 8B model to stay safe
    safe_llm = LLM(model="groq/llama-3.1-8b-instant", temperature=0.0)

    # --- AGENT CONFIGURED WITH KILLSWITCH (max_iter=2) ---
    ecosystem_agent = Agent(
        role="Local Tech Ecosystem Strategist",
        goal="Connect students with specific incubators, grants, and communities.",
        backstory="You are a veteran of the Pakistani tech startup ecosystem. You ONLY use the 'internet_search_tool'. Never invent tools.",
        llm=safe_llm,
        tools=[internet_search_tool],
        max_iter=3, # <-- The ultimate fix to prevent token context ballooning
        verbose=True
    )

    print("\n[SYSTEM] Module C: Mapping Local Ecosystem...")
    ecosystem_task = Task(
        description=f"Based on {profile.get('name')}'s background ({profile.get('degree')} at {profile.get('university')}) and goals ({profile.get('career_goal')}): 1) Recommend 2 specific Pakistani incubators or accelerators they can apply to. 2) Identify 1 specific local funding grant (e.g., Ignite NGIRI). 3) Recommend 1 active local tech community. You MUST use the 'internet_search_tool' to verify.",
        expected_output="A highly actionable, curated list of 2 incubators, 1 grant, and 1 community group in Pakistan.",
        agent=ecosystem_agent
    )

    # --- SAFE EXECUTION ---
    ecosystem_out = safe_kickoff(Crew(agents=[ecosystem_agent], tasks=[ecosystem_task]), "Ecosystem Mapping")

    return {
        "ecosystem": ecosystem_out.raw if ecosystem_out else "Ecosystem Mapping Failed due to API limits. Please try again."
    }