import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


embed_model = genai.embed_content

def load_embedding_file(path, source_label):
    with open(path, "r") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["embedding"] = df["embedding"].apply(np.array)
    df["Source"] = source_label

    df.rename(columns={
        "Assessment Length": "Duration",
        "Remote Testing": "Remote Testing Support",
        "Adaptive/IRT": "Adaptive/IRT Support"
    }, inplace=True)

    required_cols = [
        "Assessment Name", "Duration", "Test Type", "Job Level",
        "Remote Testing Support", "Adaptive/IRT Support", "URL"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "N/A"

    return df

def load_combined_data():
    prepackaged = load_embedding_file("data/prepackaged_embeddings.json", "Prepackaged")
    individual = load_embedding_file("data/individual_embeddings.json", "Individual")
    return pd.concat([prepackaged, individual], ignore_index=True)

def get_query_embedding(query: str):
    try:
        res = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_document"
        )
        return np.array(res["embedding"])
    except Exception as e:
        print("❌ Gemini embedding error:", e)
        return np.zeros(768)  # fallback

def get_top_recommendations(query: str, k: int = 10):
    df = load_combined_data()
    query_vec = get_query_embedding(query)

    if np.all(query_vec == 0):
        return pd.DataFrame()

    embeddings = np.stack(df["embedding"].values)
    sims = cosine_similarity([query_vec], embeddings)[0]

    df["score"] = sims
    df_sorted = df.sort_values("score", ascending=False)

    # 🔥 Return up to k matches (even if fewer exist)
    df_top = df_sorted.head(k)

    return df_top[[  # Return only required fields
        "Assessment Name",
        "Source",
        "URL",
        "Job Level",
        "Duration",
        "Test Type",
        "Remote Testing Support",
        "Adaptive/IRT Support",
        "score"
    ]].reset_index(drop=True)
