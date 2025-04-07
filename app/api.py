from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from engine.recommender import get_top_recommendations
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.get("/")
@app.head("/")
def read_root():
    return {"message": "SHL Assessment Recommender API is running 🚀"}

@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    try:
        results = get_top_recommendations(request.query, request.top_k)

        if results.empty:
            print(f"⚠️ No matches found for query: '{request.query}'")
            return []

        records = results.to_dict(orient="records")

        # ✅ Clean invalid float values (NaN, Inf) for JSON safety
        for record in records:
            for key, value in record.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    record[key] = None

        # 🪵 Log how many results were returned vs requested
        print(f"🔍 Query: '{request.query}' | Requested: {request.top_k}, Returned: {len(records)}")

        return records

    except Exception as e:
        print(f"❌ Error in /recommend endpoint: {str(e)}")
        return {"error": str(e)}
