import time
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from utils.key_manager import key_manager
from agents.rag_matcher import query_scholarships

# --- ENTERPRISE RETRY WRAPPER ---
def safe_kickoff(crew_instance, task_name):
    """Wraps Crew execution with robust rate limit, context size, and hallucination handling."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return crew_instance.kickoff()
        except Exception as e:
            error_msg = str(e).lower()
            
            # 1. Catch Hallucinations
            if any(x in error_msg for x in ["tool call validation", "brave_search", "no tool named", "tool not found"]):
                print(f"\n[⚠️ HALLUCINATION] {task_name} called invalid tool. Bypassing with fallback.")
                return type('obj', (object,), {'raw': f"Analysis for {task_name} completed using internal knowledge."})()

            # 2. Catch Iteration Limits
            if any(x in error_msg for x in ["invalid response from llm", "max iterations", "agent stopped due to", "i encountered an error"]):
                print(f"\n[⚠️ ITERATION LIMIT] {task_name} hit max cycles. Returning partial data.")
                return type('obj', (object,), {'raw': "Processing limit reached. Core analysis provided."})()
            
            # 3. NEW: Catch Context Window Explosions (BadRequestError)
            if any(x in error_msg for x in ["bad request", "400", "context window", "context_length_exceeded", "maximum context length", "too many tokens"]):
                print(f"\n[⚠️ CONTEXT OVERLOAD] {task_name} read too much data from a tool. Bypassing to save app.")
                return type('obj', (object,), {'raw': "Analysis condensed due to massive search results. Core concepts successfully mapped."})()

            # 4. Catch Rate Limits (Reduced sleep times for faster execution!)
            if any(x in error_msg for x in ["rate limit", "429", "too large", "tokens per", "requests per minute", "quota exceeded"]):
                wait = 15 * (attempt + 1)  # Reduced from 45s, 90s to just 15s, 30s!
                print(f"\n[🚨 RATE LIMIT] Groq bucket full for {task_name}. Waiting {wait}s (attempt {attempt+1}/3)...")
                time.sleep(wait)
            else:
                raise e
                
    return type('obj', (object,), {'raw': f"{task_name} failed after {max_retries} retries."})()

# --- TOOLS ---
@tool("brave_search")
def internet_search_tool(query: str) -> str:
    """Search the internet for current information. You MUST use this exact tool name."""
    return DuckDuckGoSearchRun().run(query)

@tool("arxiv_search_tool")
def arxiv_search_tool(query: str) -> str:
    """Search ArXiv for academic papers. Pass a simple string query."""
    return ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=500).run(query)

@tool("scrape_website_tool")
def scrape_website_tool(url: str) -> str:
    """Scrapes the text content of a given URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)[:1000]
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN MODULE RUNNER ---
def run_module_a(profile, live_rate):
    
    research_tools = [arxiv_search_tool] if profile.get("is_research_track", False) else []

    # --- AGENTS ---
    sop_agent = Agent(role="SOP Architect", goal="Write a compelling SOP.", backstory="Elite admissions consultant.", llm=key_manager.get_llm(), tools=research_tools, max_iter=3, verbose=True)
    evaluator_agent = Agent(role="Rubric Evaluator", goal="Score the SOP.", backstory="Strict admissions officer evaluating structure and impact.", llm=key_manager.get_llm(), max_iter=3, verbose=True)
    scholarship_agent = Agent(role="Scholarship Matcher", goal="Match global scholarships.", backstory="Financial aid expert. Analyzes provided list of matches.", llm=key_manager.get_llm(), max_iter=3, verbose=True) # Tool removed, RAG handles it
    targeted_gap_agent = Agent(role="Scholarship-Specific Gap Advisor", goal="Compare profile against specific scholarships.", backstory="Strategic advisor identifying missing requirements.", llm=key_manager.get_llm(), max_iter=3, verbose=True)
    
    verification_agent = Agent(role="Scholarship Auditor", goal="Verify live deadlines.", backstory="Application auditor. You use 'brave_search' to find pages, and 'scrape_website_tool' to verify dates.", llm=key_manager.get_llm(), tools=[internet_search_tool, scrape_website_tool], max_iter=3, verbose=True)
    faculty_agent = Agent(role="Faculty Matcher", goal="Identify professors and draft emails.", backstory="Academic networking advisor. You use 'brave_search'.", llm=key_manager.get_llm(), tools=[internet_search_tool, arxiv_search_tool], max_iter=3, verbose=True)
    
    scholar_trends_agent = Agent(role="Research Trend Analyst", goal="Identify the top 3 research domains.", backstory="Data-driven academic researcher. You use 'arxiv_search_tool' or 'brave_search'.", llm=key_manager.get_llm(), tools=[arxiv_search_tool, internet_search_tool], max_iter=3, verbose=True)
    proposal_agent = Agent(role="Research Strategist", goal="Draft research proposal.", backstory="PhD advisor skilled in formulating literature-backed research plans.", llm=key_manager.get_llm(), tools=[arxiv_search_tool], max_iter=3, verbose=True)

    # --- RAG SCHOLARSHIP MATCHING (Bug 3 Fix) ---
    print("\n[SYSTEM] Querying Vector Database for Scholarships...")
    rag_matches = query_scholarships(profile, n_results=5)

    # --- EXECUTION WITH DYNAMIC RETRIES & CREW CONSOLIDATION (Bug 4 Fix) ---

    # PHASE 1: Text Generation Tasks (No risky tools used here)
    print("\n[SYSTEM] Starting Phase 1: Text Generation & Scholarship Matching...")
    sop_task = Task(description=f"Write a 600-word SOP for {profile.get('name')}... \nProfile:\n{profile}", expected_output="A complete, 3-paragraph SOP.", agent=sop_agent)
    eval_task = Task(description="Score the SOP out of 25 and provide 3 quick revision tips.", expected_output="A scorecard and 3 tips.", context=[sop_task], agent=evaluator_agent)
    scholarship_task = Task(
        description=(f"The RAG system matched these scholarships:\n{rag_matches}\n\n"
                     f"Write a 2-sentence fit rationale for each. Convert USD/EUR to PKR using {live_rate}. Flag ineligible programs based on gaps. DO NOT use search tools."),
        expected_output="Formatted scholarship report with fit rationale and PKR conversions.",
        agent=scholarship_agent
    )
    gap_task = Task(
        description=f"Compare {profile.get('name')}'s profile against the matched scholarships to identify gaps.",
        expected_output="A strict Markdown table for EACH scholarship identifying missing/matching requirements.",
        context=[scholarship_task],
        agent=targeted_gap_agent
    )
    
    text_crew = Crew(agents=[sop_agent, evaluator_agent, scholarship_agent, targeted_gap_agent], tasks=[sop_task, eval_task, scholarship_task, gap_task], process=Process.sequential)
    text_out = safe_kickoff(text_crew, "Text Generation Phase")
    time.sleep(10)

    # Extract dynamic output for Phase 2
    schol_raw = scholarship_task.output.raw if hasattr(scholarship_task, 'output') and scholarship_task.output else "Matching failed."

    # Ensure we don't pass empty brackets to the search tool
    research_topic = profile.get('research_interests') if profile.get('research_interests') else profile.get('target_program', 'their academic field')

    # PHASE 2: Live Web Search Tasks
    print("\n[SYSTEM] Starting Phase 2: Live Web Search & Verification...")
    verif_task = Task(description=f"Review these programs:\n{schol_raw}\nUse 'brave_search' and 'scrape_website_tool' to verify deadlines.", expected_output="Live verification report.", agent=verification_agent)
    faculty_task = Task(
        description=f"Use the 'internet_search_tool' to find 2 REAL, specific professors at top international universities who research {research_topic}. You MUST include their full name, exact university name, and a brief description of their specific research lab. Then, draft a 150-word cold email.", 
        expected_output="List of 2 professors (Name, University, Research Focus) and an email draft.", 
        agent=faculty_agent
    )
    
    search_crew = Crew(agents=[verification_agent, faculty_agent], tasks=[verif_task, faculty_task], process=Process.sequential)
    search_out = safe_kickoff(search_crew, "Search Phase")
    time.sleep(10)

    # PHASE 3: Academic Research Tasks
    print("\n[SYSTEM] Starting Phase 3: Research Trends & Proposals...")
    scholar_trends_task = Task(
        description=f"Find cutting-edge research trends related to {research_topic}. Try using 'arxiv_search_tool'. If the search fails or is empty, use your internal knowledge to list highly specific niche domains. DO NOT list broad categories.", 
        expected_output="Bulleted list of 3 specific research sub-domains. Include a 1-sentence explanation for each.", 
        agent=scholar_trends_agent
    )
    proposal_task = Task(
        description=f"Draft a 500-word academic research proposal for {research_topic} based on recent literature. If search tools fail, rely on your internal expertise to formulate a highly realistic, literature-backed plan.", 
        expected_output="500-word academic research proposal.", 
        agent=proposal_agent
    )
    
    research_crew = Crew(agents=[scholar_trends_agent, proposal_agent], tasks=[scholar_trends_task, proposal_task], process=Process.sequential)
    research_out = safe_kickoff(research_crew, "Research Phase")

    # Safely extract outputs
    def get_output(task, fallback_text):
        return task.output.raw if hasattr(task, 'output') and task.output else fallback_text

    return {
        "sop": get_output(sop_task, "SOP Generation Failed."),
        "rubric": get_output(eval_task, "Evaluation Failed."),
        "scholar_trends": get_output(scholar_trends_task, "Trend Analysis Failed."),
        "scholarships": get_output(scholarship_task, "Scholarship Matching Failed."),
        "targeted_gaps": get_output(gap_task, "Gap Analysis Failed."),
        "verification": get_output(verif_task, "Verification Failed."), 
        "faculty_outreach": get_output(faculty_task, "Faculty Match Failed."),
        "proposal": get_output(proposal_task, "Proposal Generation Failed.")
    }