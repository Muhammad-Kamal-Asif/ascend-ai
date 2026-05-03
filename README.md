# 🎓 Ascend AI: Global & Local Career Navigator

Ascend AI is a multi-agent, load-balanced AI platform designed to help students map their academic and professional futures. By analyzing a student's profile or CV, Ascend AI autonomously drafts publication-ready Statements of Purpose (SOPs), matches them with fully-funded international scholarships, maps localized career trajectories, and connects them with the local tech ecosystem.

## 🚀 Key Features
*   **Multi-Agent Orchestration:** Utilizes specialized AI agents (Profile Architect, SOP Writer, Scholarship Matcher, Ecosystem Advisor) working in a highly optimized sequence.
*   **Fault-Tolerant Execution:** Custom API key rotation and error-handling wrappers seamlessly bypass rate limits and LLM tool hallucinations during high-traffic demonstrations.
*   **Real-Time Web Search:** Integrates DuckDuckGo, ArXiv, and BeautifulSoup for verified 2026 deadlines, academic papers, and live data scraping.
*   **Session History Dashboard:** Built-in state management allows users to instantly recall previous profile generations and toggle between different student roadmaps without spending extra API tokens.
*   **Pydantic V2 Aligned:** Engineered on the modern LangChain 0.2.x and CrewAI 0.35+ ecosystem for unbreakable tool validation.

## 🛠️ Architecture & Tech Stack
*   **Frontend:** Streamlit
*   **Orchestration:** CrewAI
*   **LLM Integration:** LangChain (v0.2.x), Groq API
*   **Primary Model:** `llama-3.1-8b-instant` (Optimized for high-speed multi-agent interaction and dynamic tool calling).
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
4. Access previous generations via the **Session History** in the sidebar.

## 👨‍💻 Developed By

*   **Muhammad Kamal**: Lead AI Architect & Product Strategist
*   **Kinza Irshad**: WordPress & UI/UX Developer

---
*Disclaimer: Ascend AI is an AI advisory tool. Users are encouraged to verify all deadlines on official university portals.*