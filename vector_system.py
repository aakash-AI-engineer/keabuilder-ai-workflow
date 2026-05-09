from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

# Initialize FastAPI, Embedding Model, and Vector DB
app = FastAPI(title="VARYNT Mini Vector System")
model = SentenceTransformer('all-MiniLM-L6-v2') 
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="user_assets")

# --- Schemas ---
class AssetInput(BaseModel):
    text: str = Field(..., min_length=2)
    metadata: dict = Field(default={})

class QueryInput(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(default=3, le=10)

# --- Endpoints ---
@app.post("/api/v1/store")
async def store_input(asset: AssetInput):
    """Generates an embedding and stores it in the vector database."""
    try:
        asset_id = str(uuid.uuid4())
        embedding = model.encode(asset.text).tolist()
        collection.add(
            ids=[asset_id], embeddings=[embedding],
            documents=[asset.text], metadatas=[asset.metadata]
        )
        return {"status": "success", "asset_id": asset_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search")
async def retrieve_similar(query_data: QueryInput):
    """Retrieves the most semantically similar stored inputs."""
    try:
        query_embedding = model.encode(query_data.query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding], n_results=query_data.top_k
        )
        matches = [{"id": results['ids'][0][i], "text": results['documents'][0][i], "distance": results['distances'][0][i]} for i in range(len(results['documents'][0]))] if results['documents'][0] else []
        return {"query": query_data.query, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
