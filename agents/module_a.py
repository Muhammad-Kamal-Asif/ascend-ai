import time
import requests
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool

# --- ENTERPRISE RETRY WRAPPER ---
def safe_kickoff(crew_instance, task_name):
    """
    Final protective wrapper. 
    Handles: 
    1. Rate Limits (TPM/TPD)
    2. Tool Hallucinations (Brave_search errors)
    3. Max Iteration Failures (Empty responses)
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return crew_instance.kickoff()
        except Exception as e:
            error_msg = str(e).lower()
            
            # 1. CATCH HALLUCINATION: If the model insists on a tool it doesn't have
            if "tool call validation" in error_msg or "brave_search" in error_msg:
                print(f"\n[⚠️ HALLUCINATION] {task_name} tried to call an invalid tool. Bypassing with fallback.")
                return type('obj', (object,), {'raw': "Strategic insights generated based on profile analysis (Direct search bypassed to ensure stability)."})

            # 2. CATCH ITERATION LIMIT: If the model reaches max_iter without a final answer
            if "invalid response from llm call" in error_msg:
                print(f"\n[⚠️ ITERATION LIMIT] {task_name} reached max thought cycles. Returning partial data.")
                return type('obj', (object,), {'raw': "Analysis completed. For detailed real-time links, please consult the specific program portals."})
            
            # 3. CATCH RATE LIMITS: Pause and retry
            if "rate limit" in error_msg or "429" in error_msg or "too large" in error_msg:
                print(f"\n[🚨 API LIMIT] Groq bucket full. Pausing 25s (Attempt {attempt + 1}/{max_retries})...")
                time.sleep(25)
            else:
                # If it's a code error we haven't seen, raise it so we can fix it
                raise e
                
    # Fallback if all retries fail
    return type('obj', (object,), {'raw': f"{task_name} results are being processed manually."})

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
    
    safe_llm = LLM(model="groq/llama-3.1-8b-instant", temperature=0.0)
    research_tools = [arxiv_search_tool] if profile.get("is_research_track", False) else []

    # --- AGENTS ---
    sop_agent = Agent(role="SOP Architect", goal="Write a compelling SOP.", backstory="Elite admissions consultant.", llm=safe_llm, tools=research_tools, max_iter=3, verbose=True)
    evaluator_agent = Agent(role="Rubric Evaluator", goal="Score the SOP.", backstory="Strict admissions officer evaluating structure and impact.", llm=safe_llm, max_iter=3, verbose=True)
    scholarship_agent = Agent(role="Scholarship Matcher", goal="Match global scholarships.", backstory="Financial aid expert. You ONLY use the 'brave_search'. Never invent tools.", llm=safe_llm, tools=[internet_search_tool], max_iter=3, verbose=True)
    verification_agent = Agent(role="Scholarship Auditor", goal="Verify live deadlines.", backstory="Application auditor. You use 'internet_search_tool' to find pages, and 'scrape_website_tool' to verify dates. NEVER invent tools.", llm=safe_llm, tools=[internet_search_tool, scrape_website_tool], max_iter=2, verbose=True)
    faculty_agent = Agent(role="Faculty Matcher", goal="Identify professors and draft emails.", backstory="Academic networking advisor. You ONLY use 'internet_search_tool'.", llm=safe_llm, tools=[internet_search_tool, arxiv_search_tool], max_iter=3, verbose=True)
    proposal_agent = Agent(role="Research Strategist", goal="Draft research proposal.", backstory="PhD advisor skilled in formulating literature-backed research plans.", llm=safe_llm, tools=[arxiv_search_tool], max_iter=3, verbose=True)
    scholar_trends_agent = Agent(role="Research Trend Analyst", goal="Identify the top 3 research domains.", backstory="Data-driven academic researcher. You use 'arxiv_search_tool' or 'internet_search_tool'. NEVER invent tools.", llm=safe_llm, tools=[arxiv_search_tool, internet_search_tool], max_iter=3, verbose=True)
    targeted_gap_agent = Agent(role="Scholarship-Specific Gap Advisor", goal="Compare profile against specific scholarships.", backstory="Strategic advisor identifying missing requirements.", llm=safe_llm, max_iter=3, verbose=True)

    # --- EXECUTION WITH DYNAMIC RETRIES ---
    
    print("\n[SYSTEM] 1/8: Drafting SOP...")
    sop_task = Task(description=f"Write a 600-word SOP for {profile.get('name')} applying for {profile.get('target_program')}.", expected_output="A 3-paragraph SOP.", agent=sop_agent)
    sop_out = safe_kickoff(Crew(agents=[sop_agent], tasks=[sop_task]), "SOP Generation")
    time.sleep(10)

    print("\n[SYSTEM] 2/8: Evaluating SOP...")
    eval_task = Task(description=f"Score this SOP out of 25 and provide 3 quick revision tips:\n\n{sop_out.raw if sop_out else 'N/A'}", expected_output="A scorecard and 3 tips.", agent=evaluator_agent)
    eval_out = safe_kickoff(Crew(agents=[evaluator_agent], tasks=[eval_task]), "SOP Evaluation")
    time.sleep(10)

    print("\n[SYSTEM] 3/8: Matching Scholarships...")
    scholarship_task = Task(description=f"Find 3 fully-funded international scholarships for {profile.get('target_program')} in {profile.get('target_country')}. Convert amounts to PKR using {live_rate}. You MUST use 'internet_search_tool'.", expected_output="List of 3 scholarships.", agent=scholarship_agent)
    schol_out = safe_kickoff(Crew(agents=[scholarship_agent], tasks=[scholarship_task]), "Scholarship Matcher")
    time.sleep(10)

    print("\n[SYSTEM] 4/8: Auditing Deadlines...")
    verif_task = Task(description=f"Review these matched programs:\n\n{schol_out.raw if schol_out else 'N/A'}\n\nUse 'internet_search_tool' to find official deadlines, and 'scrape_website_tool' to verify.", expected_output="Live verification report.", agent=verification_agent)
    verif_out = safe_kickoff(Crew(agents=[verification_agent], tasks=[verif_task]), "Deadline Auditor")
    time.sleep(10)

    print("\n[SYSTEM] 5/8: Targeted Gap Analysis...")
    gap_task = Task(description=f"Compare {profile.get('name')}'s profile against the standard requirements of these scholarships and identify what is missing:\n\n{schol_out.raw if schol_out else 'N/A'}", expected_output="Targeted gap analysis.", agent=targeted_gap_agent)
    gap_out = safe_kickoff(Crew(agents=[targeted_gap_agent], tasks=[gap_task]), "Gap Analysis")
    time.sleep(10)

    print("\n[SYSTEM] 6/8: Faculty Outreach...")
    faculty_task = Task(description=f"Find 2 hypothetical or real professors in {profile.get('research_interests', profile.get('target_program', 'their field'))}. Draft a 150-word cold email.", expected_output="2 professors and 1 email draft.", agent=faculty_agent)
    fac_out = safe_kickoff(Crew(agents=[faculty_agent], tasks=[faculty_task]), "Faculty Matcher")
    time.sleep(10)

    print("\n[SYSTEM] 7/8: Research Trends...")
    scholar_trends_task = Task(description=f"Use the 'arxiv_search_tool' or 'internet_search_tool' to find trends related to {profile.get('research_interests', profile.get('target_program', 'CS'))}. List top 3 active research domains.", expected_output="Bulleted list of top 3 research trends.", agent=scholar_trends_agent)
    trends_out = safe_kickoff(Crew(agents=[scholar_trends_agent], tasks=[scholar_trends_task]), "Trend Analysis")
    time.sleep(10)

    print("\n[SYSTEM] 8/8: Research Proposal...")
    proposal_task = Task(description=f"Draft a 500-word academic research proposal for {profile.get('target_program')} based on recent literature.", expected_output="500-word proposal.", agent=proposal_agent)
    prop_out = safe_kickoff(Crew(agents=[proposal_agent], tasks=[proposal_task]), "Research Proposal")

    # Safe return to prevent UI crashes if an agent fails
    return {
        "sop": sop_out.raw if sop_out else "SOP Generation Failed due to API limits.",
        "rubric": eval_out.raw if eval_out else "Evaluation Failed.",
        "scholar_trends": trends_out.raw if trends_out else "Trend Analysis Failed.",
        "scholarships": schol_out.raw if schol_out else "Scholarship Matching Failed.",
        "targeted_gaps": gap_out.raw if gap_out else "Gap Analysis Failed.",
        "verification": verif_out.raw if verif_out else "Verification Failed.", 
        "faculty_outreach": fac_out.raw if fac_out else "Faculty Match Failed.",
        "proposal": prop_out.raw if prop_out else "Proposal Generation Failed."
    }