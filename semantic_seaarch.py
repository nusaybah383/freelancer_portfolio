# src/semantic_search.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load model once to reuse
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("data/freelancers.csv")

def build_profile(row):
    return f"{row['name']} is skilled in {row['skills']}. Experience: {row['experience_years']} years. Rating: {row['rating']}."

df["profile_text"] = df.apply(build_profile, axis=1)
freelancer_embeddings = model.encode(df["profile_text"].tolist(), normalize_embeddings=True)

# Create and populate FAISS index
d = freelancer_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(np.array(freelancer_embeddings))

def find_best_freelancers(project_desc: str, top_k: int = 3):
    query_emb = model.encode([project_desc], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_emb), top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        freelancer = df.iloc[idx]
        results.append({
            "freelancer_id": freelancer["freelancer_id"],
            "name": freelancer["name"],
            "similarity_score": float(score)
        })
    return results