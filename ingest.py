import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = "data"
DB_DIR = "./chroma_db"
DATA_FILE = os.path.join(DATA_DIR, "scholarships.json")

def ensure_data_exists():
    """Creates the fallback scholarships.json if it doesn't exist to ensure the script is 100% runnable."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(DATA_FILE):
        return

    print(f"Creating foundational dataset at {DATA_FILE}...")
    scholarships = [
        {"name": "Fulbright Degree Program", "country": "USA", "fields": "All fields except clinical medicine", "gpa_min": "3.0", "deadline_month": "February", "bond": "2-year home residency requirement", "url": "https://pk.usembassy.gov/education-culture/fulbright-degree-program/", "note": "Fully funded Master's and PhD for Pakistani citizens. Requires GRE."},
        {"name": "Chevening Scholarship", "country": "UK", "fields": "All fields", "gpa_min": "2.8", "deadline_month": "November", "bond": "2-year home residency requirement", "url": "https://www.chevening.org/scholarship/pakistan/", "note": "Fully funded 1-year Master's. Focus on leadership and networking. Requires 2 years work experience."},
        {"name": "DAAD EPOS Scholarship", "country": "Germany", "fields": "Development-related fields, Engineering, Economics", "gpa_min": "2.8", "deadline_month": "Varies by program (usually Aug-Oct)", "bond": "None", "url": "https://www.daad.de/", "note": "Fully funded Master's and PhD. Requires 2 years of professional experience."},
        {"name": "HEC Overseas Scholarship", "country": "Global", "fields": "Priority areas defined by HEC (STEM, AI, Agriculture)", "gpa_min": "3.0", "deadline_month": "Spring/Fall", "bond": "5-year service bond to Pakistan", "url": "https://www.hec.gov.pk/", "note": "Fully funded PhD programs for Pakistani nationals. Requires HAT test."},
        {"name": "Commonwealth Scholarship", "country": "UK", "fields": "Science, Tech, Health, Rural Development", "gpa_min": "3.0", "deadline_month": "October", "bond": "Must return to home country", "url": "https://cscuk.fcdo.gov.uk/", "note": "Fully funded Master's and PhD for Commonwealth citizens."},
        {"name": "CSC Chinese Government Scholarship", "country": "China", "fields": "All fields", "gpa_min": "2.5", "deadline_month": "March", "bond": "None", "url": "http://www.campuschina.org/", "note": "Fully funded Bachelor's, Master's, and PhD. Bilateral program often routed through HEC."},
        {"name": "Turkiye Burslari", "country": "Turkey", "fields": "All fields", "gpa_min": "3.0", "deadline_month": "February", "bond": "None", "url": "https://turkiyeburslari.gov.tr/", "note": "Fully funded at all levels. Includes 1 year of Turkish language training."},
        {"name": "KAIST International Graduate Scholarship", "country": "South Korea", "fields": "STEM, Computer Science, Engineering", "gpa_min": "3.2", "deadline_month": "March/September", "bond": "None", "url": "https://admission.kaist.ac.kr/", "note": "Highly competitive, fully funded MS and PhD. Strong focus on research output."},
        {"name": "Erasmus Mundus Joint Masters", "country": "Europe (Multiple)", "fields": "Interdisciplinary fields", "gpa_min": "3.0", "deadline_month": "October-January", "bond": "None", "url": "https://erasmus-plus.ec.europa.eu/", "note": "Study in at least two different European countries. Highly prestigious."},
        {"name": "OIST Graduate Program", "country": "Japan", "fields": "Science, Engineering, Interdisciplinary", "gpa_min": "3.0", "deadline_month": "November", "bond": "None", "url": "https://admissions.oist.jp/", "note": "Fully funded 5-year PhD program. No Japanese language requirement. State-of-the-art labs."}
    ]
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(scholarships, f, indent=4)

def run_ingestion():
    # 1. Check if DB already exists to avoid redundant rebuilding
    if os.path.exists(DB_DIR) and os.listdir(DB_DIR):
        print("✅ ChromaDB collection already exists on disk. Skipping ingestion.")
        return

    print("🚀 Initializing ChromaDB vector database setup...")
    
    # 2. Ensure data source exists and load it
    ensure_data_exists()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        scholarships = json.load(f)

    # 3. Initialize ChromaDB persistent client
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Ensure a clean slate if script is forced
    try:
        client.delete_collection("scholarships")
    except Exception:
        pass # Collection didn't exist yet

    collection = client.create_collection("scholarships")

    # 4. Initialize HuggingFace Embedding Model
    print("🧠 Loading HuggingFace sentence-transformer (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # 5. Process and Embed Data
    print("⏳ Embedding scholarships into vector space...")
    ids = []
    documents = []
    metadatas = []

    for i, sch in enumerate(scholarships):
        # Create a rich semantic text representation for embedding
        # We combine country, fields, and notes so semantic search catches all nuances
        semantic_text = f"{sch['name']} in {sch['country']}. Fields: {sch['fields']}. Details: {sch['note']}"
        
        documents.append(semantic_text)
        ids.append(f"scholarship_{i}")
        # Keep the raw data in metadata for easy retrieval later
        metadatas.append({
            "name": sch["name"],
            "country": sch["country"],
            "fields": sch["fields"],
            "gpa_min": sch["gpa_min"],
            "deadline_month": sch["deadline_month"],
            "bond": sch["bond"],
            "url": sch["url"],
            "note": sch["note"]
        })

    # Generate embeddings
    embeddings = embedder.encode(documents).tolist()

    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"✅ Successfully embedded and persisted {len(scholarships)} scholarships to {DB_DIR}!")

if __name__ == "__main__":
    run_ingestion()