from langchain_community.tools import YouTubeSearchTool
from crewai import Agent, Task, Crew, Process
from crewai.tools import tool

from utils.key_manager import key_manager

# --- NATIVE CREWAI TOOL WRAPPER ---
@tool
def youtube_search_tool(query: str) -> str:
    """Search YouTube for tutorial videos. You MUST pass a short string query."""
    return YouTubeSearchTool().run(query)

def run_module_b(profile):
    
    career_agent = Agent(
        role="Career Path Mapper",
        goal="Map student skills to high-paying digital economy roles.",
        backstory="You are an expert career counselor focused on the Pakistani freelance and tech job market.",
        llm=key_manager.get_llm(),
        verbose=True
    )

    roadmap_agent = Agent(
        role="Learning Roadmap Builder",
        goal="Create a 12-week skill syllabus with verified YouTube resources.",
        backstory="You are a curriculum designer who uses free internet resources to upskill students. You must extract and provide the actual YouTube URLs in your final response, NEVER raw tool call code or <function> tags.",
        llm=key_manager.get_llm(),
        tools=[youtube_search_tool],
        verbose=True
    )

    freelance_agent = Agent(
        role="Freelance Market Entry Strategist",
        goal="Convert skills into ready-to-sell freelance services on Upwork and Fiverr.",
        backstory="You are a top-rated plus freelancer in Pakistan. You know exactly how to write Upwork bios that get clients and Fiverr gigs that rank.",
        llm=key_manager.get_llm(),
        verbose=True
    )

    # --- NEW AGENT: ACADEMIA VS INDUSTRY ---
    phd_advisor_agent = Agent(
        role="Academic vs Industry Strategist",
        goal="Provide a brutally honest tradeoff analysis between pursuing a PhD versus entering the industry.",
        backstory="You are a pragmatist. You know that a PhD is not for everyone and has huge opportunity costs. You evaluate the student's career goals and tell them the hard truth about whether academia or industry suits them better, providing 2 specific university recommendations if they choose the academic route.",
        llm=key_manager.get_llm(),
        verbose=True
    )

    career_task = Task(
        description=f"Map {profile.get('name')}'s skills ({profile.get('special_skills')}) to 5 modern digital careers. Include realistic entry-level PKR salaries for remote work.",
        expected_output="List of 5 digital careers with PKR salaries.",
        agent=career_agent
    )

    skill_level = "intermediate" if profile.get('work_experience') else "beginner"

    roadmap_task = Task(
        description=f"Create a 12-week YouTube learning roadmap for {profile.get('name')} to master {profile.get('target_program', 'their field')}. Use the youtube_search_tool to find real videos. You MUST format this as a vertical list containing actual 'https://www.youtube.com/...' URLs.",
        expected_output="A strict Markdown list. Each week MUST start with a bullet point (e.g., '- **Week 1:** [Topic] - https://www.youtube.com/watch?v=...'). Do NOT output <function> tags.",
        agent=roadmap_agent
    )

    freelance_task = Task(
        description=f"Based on the top career identified for {profile.get('name')}, generate: 1) A highly converting 150-word Upwork Profile Bio. 2) Three specific Fiverr gig titles with suggested entry-level pricing in PKR. 3) One practical portfolio project they can build in the next 14 days.",
        expected_output="An Upwork bio, 3 Fiverr gig titles with pricing, and 1 portfolio project idea.",
        agent=freelance_agent
    )

    # --- NEW TASK: THE TRADEOFF ---
    phd_advisor_task = Task(
        description=f"Analyze {profile.get('name')}'s profile. Their career goal is: '{profile.get('career_goal')}'. Write a 2-paragraph honest tradeoff analysis: Should they pursue a PhD/Master's by Research, or go straight into the Industry? Give concrete reasons based on their goals and the current tech market. Finally, list 2 specific universities in {profile.get('target_country', 'the world')} known for this field.",
        expected_output="A brutally honest Academia vs Industry tradeoff analysis with 2 university recommendations.",
        agent=phd_advisor_agent
    )

    crew = Crew(
        agents=[career_agent, roadmap_agent, freelance_agent, phd_advisor_agent], 
        tasks=[career_task, roadmap_task, freelance_task, phd_advisor_task], 
        process=Process.sequential
    )
    crew.kickoff()

    # Safely extract outputs in case of an execution hiccup
    def get_output(task, fallback_text):
        return task.output.raw if hasattr(task, 'output') and task.output else fallback_text

    return {
        "careers": get_output(career_task, "Career mapping failed."),
        "roadmap": get_output(roadmap_task, "Roadmap generation failed."),
        "freelance": get_output(freelance_task, "Freelance strategy failed."),
        "phd_tradeoff": get_output(phd_advisor_task, "Tradeoff analysis failed.") 
    }