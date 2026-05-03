import os
import json
import chromadb
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = "data"
DB_DIR = "./chroma_db"
DATA_FILE = os.path.join(DATA_DIR, "scholarships.json")

def ensure_data_exists():
    """Validates that the real scholarship database exists and has adequate data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"scholarships.json not found at {DATA_FILE}.\n"
            f"Please ensure the data/ folder is present before running ingest.py.\n"
            f"The full 45+ scholarship database must be placed there."
        )
        
    # Count entries and warn if suspiciously low
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if len(data) < 20:
        print(f"⚠️  WARNING: Only {len(data)} scholarships found. Expected 40+. Check your scholarships.json.")
    else:
        print(f"✅ Found {len(data)} scholarships. Proceeding with ingestion.")

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