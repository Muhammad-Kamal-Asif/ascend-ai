import chromadb
from sentence_transformers import SentenceTransformer
import json

_client = None
_collection = None
_embedder = None

def _init_rag():
    global _client, _collection, _embedder
    if _collection is None:
        try:
            _client = chromadb.PersistentClient(path="./chroma_db")
            _collection = _client.get_collection("scholarships")
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[RAG] ChromaDB not found: {e}. Run ingest.py first.")
            _collection = None

def query_scholarships(profile: dict, n_results: int = 5) -> str:
    """
    Query ChromaDB with the student profile.
    Returns a formatted string of top matching scholarships.
    Falls back to JSON file if ChromaDB unavailable.
    """
    _init_rag()
    
    if _collection is None:
        # Fallback: load from JSON directly if DB fails
        try:
            with open("data/scholarships.json") as f:
                all_scholarships = json.load(f)
            return json.dumps(all_scholarships[:5], indent=2)
        except:
            return "Scholarship data unavailable."
    
    # Build a rich semantic query string from the profile
    query_parts = []
    if profile.get("degree"):        query_parts.append(profile["degree"])
    if profile.get("target_program"): query_parts.append(profile["target_program"])
    if profile.get("research_interests"): 
        query_parts.extend(profile["research_interests"])
    if profile.get("target_country"):
        query_parts.extend(profile["target_country"])
    if profile.get("career_goal"):   query_parts.append(profile["career_goal"])
    if profile.get("special_skills"): 
        query_parts.extend(profile["special_skills"])
    
    query_text = " ".join(query_parts)
    
    try:
        query_embedding = _embedder.encode(query_text).tolist()
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        scholarships = []
        for i, metadata in enumerate(results["metadatas"][0]):
            distance = results["distances"][0][i]
            similarity_pct = round((1 - distance) * 100, 1)
            scholarships.append({
                "rank": i + 1,
                "similarity_match": f"{similarity_pct}%",
                **metadata
            })
        
        return json.dumps(scholarships, indent=2)
        
    except Exception as e:
        return f"RAG query failed: {e}"