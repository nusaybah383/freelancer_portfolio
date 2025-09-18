# src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from .ranking_logic import rank_proposals
# You would also import find_best_freelancers from semantic_search here for a complete solution

app = FastAPI(title="BizGenie Freelancer Matcher")

class ProjectRequest(BaseModel):
    project_description: str

@app.post("/match_freelancers")
def match_freelancers(request: ProjectRequest):
    # In a real app, you would first use the semantic_search module
    # to find relevant freelancers and generate relevance_scores.
    # For now, we just rank the pre-existing proposals.csv.

    ranked_df = rank_proposals()

    results = [
        {
            "freelancer_id": row["freelancer_id"],
            "score": round(row["final_score"], 2)
        }
        for _, row in ranked_df.head(3).iterrows()
    ]
    return {"matches": results}