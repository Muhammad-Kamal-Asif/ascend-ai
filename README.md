# 🎓 Ascend AI: Global & Local Career Navigator

Ascend AI is a multi-agent, load-balanced AI platform designed to help students map their academic and professional futures. By analyzing a student's profile or CV, Ascend AI autonomously drafts publication-ready Statements of Purpose (SOPs), matches them with fully-funded international scholarships, maps localized career trajectories, and connects them with the local tech ecosystem.

## 🚀 Features
*   **Multi-Agent Orchestration:** Utilizes specialized AI agents (Profile Architect, SOP Writer, Scholarship Matcher, Ecosystem Advisor) working in sequence.
*   **Dynamic Load Balancing:** Built-in API key rotation to seamlessly handle hard rate limits during high-traffic demonstrations.
*   **Real-Time Web Search:** Integrates DuckDuckGo, ArXiv, and YouTube search tools for verified 2026 deadlines, academic papers, and dynamic learning roadmaps.
*   **Multilingual Support:** One-click academic Urdu translation for generated documentation.
*   **Pydantic V2 Aligned:** Engineered on the modern LangChain 0.2.x and CrewAI 0.35+ ecosystem for unbreakable tool validation.

## 🛠️ Architecture & Tech Stack
*   **Frontend:** Streamlit
*   **Orchestration:** CrewAI
*   **LLM Integration:** LangChain (v0.2.x), Groq API
*   **Models:** 
    *   `llama-3.3-70b-versatile` (Core logic and strict JSON generation)
    *   `llama-3.1-8b-instant` (Fast localized translation and ecosystem mapping)
*   **Document Parsing:** pdfplumber, python-docx

## 💻 Local Setup & Installation (Judge's Guide)

Follow these exact steps to run Ascend AI locally without dependency conflicts.

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/ascend-ai.git](https://github.com/your-username/ascend-ai.git)
cd ascend-ai
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
*Note: This project strictly requires the synchronized dependency versions listed in requirements.txt to prevent Pydantic V1/V2 validation errors.*
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a `.env` file in the root directory. To utilize the load balancer, you must provide two active Groq API keys:
```text
GROQ_API_KEY=gsk_your_primary_key_here
GROQ_API_KEY_2=gsk_your_secondary_key_here
```

### 5. Run the Application
```bash
streamlit run app.py
```

## 🎯 Usage
1. Upload your CV (PDF) or click **"Load Demo Profile"** to populate the background data.
2. Click **"Generate My Path"**.
3. Navigate through the tabs to view your structured profile, drafted SOP, matched 2026 scholarships, 12-week YouTube learning roadmap, and recommended local ecosystem connections.

## 👨‍💻 Developed By

*   **Muhammad Kamal**: Co-Founder & Lead Strategist.
*   **Kinza Irshad**: Co-Founder, Web Developer & Ad Expert.

---
*Disclaimer: Ascend AI is an AI advisory tool. Users are encouraged to verify all deadlines on official university portals.*