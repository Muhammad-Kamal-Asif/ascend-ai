import json
from crewai import Agent, Task, Crew, Process

def extract_profile(raw_text):
    profile_agent = Agent(
        role="Profile Intelligence Architect",
        goal="Extract unstructured student data and perform a rigorous profile strength assessment.",
        backstory="You are an elite university admissions consultant. You not only extract data but evaluate it critically against top-tier global scholarship standards (like Fulbright, Chevening, DAAD). You are brutally honest about gaps and highly strategic about improvements.",
        llm="groq/llama-3.3-70b-versatile",
        verbose=True
    )

    extraction_task = Task(
        description=f"""
        Parse the following raw notes or CV into a strict JSON object.
        Raw input: {raw_text}
        
        1. Extract the standard demographic and academic data.
        2. Calculate a 'profile_strength_score' (0-100) based on academic excellence, research output, leadership, and work experience. Be strict. A student with good grades but no research or leadership should score around 50-60.
        3. Provide a brutally honest 'gap_analysis' detailing exactly what is missing to be competitive for top international scholarships (e.g., lack of publications, no volunteer work, insufficient work experience).
        4. Provide an 'improvement_roadmap' with 3 specific, actionable steps they can take in the next 6 months to fix these gaps.

        Return ONLY a valid JSON object matching this exact schema. No markdown fences, no preamble:
        {{
          "name": "", 
          "degree": "", 
          "university": "", 
          "cgpa": "", 
          "graduation_year": "",
          "research_interests": [], 
          "work_experience": [], 
          "extracurriculars": [],
          "target_country": [], 
          "target_program": "", 
          "target_scholarships": [],
          "career_goal": "", 
          "key_strengths": [], 
          "special_skills": [],
          "is_research_track": false,
          "profile_strength_score": 0,
          "gap_analysis": "",
          "improvement_roadmap": []
        }}
        Set 'is_research_track' to true if target_program contains PhD, MS by research, or MPhil.
        """,
        expected_output="Pure JSON string matching the defined schema.",
        agent=profile_agent
    )

    crew = Crew(agents=[profile_agent], tasks=[extraction_task], process=Process.sequential)
    result = crew.kickoff()
    
    try:
        clean_json = result.raw.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json)
    except Exception as e:
        print(f"JSON Parsing Error: {e}")
        return None