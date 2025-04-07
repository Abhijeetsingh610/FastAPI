from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from engine.recommender import get_top_recommendations

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
        return results.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}
