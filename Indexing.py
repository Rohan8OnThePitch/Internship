from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
import uuid
from typesense import Client
import logging
from typesense.exceptions import ObjectNotFound

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
MODEL_PATH = "/home/roh/s_q/Sent2-test/local_model"
model = SentenceTransformer(MODEL_PATH, device="cpu")

# Confirm device
for param in model.parameters():
    logger.info(f"Model is on device: {param.device}")
    break

# FastAPI app
app = FastAPI()

# Pydantic models
class Request(BaseModel):
    sentences: list[str]

class Response(BaseModel):
    embeddings: list[list[float]]

class StoreResponse(BaseModel):
    id: str
    status: str

# Typesense client
client = Client({
    'api_key': 'xyz',
    'nodes': [{
        'host': 'localhost',
        'port': '8108',
        'protocol': 'http'
    }]
})


CUSTOM_SCHEMA = {
    "name": "embeddings_db",
    "fields": [
        {
            "name": "id",
            "type": "string"
        },
        {
            "name": "text",
            "type": "string"
        },
        {
            "name": "sentences",
            "type": "string[]"
        },
        {
            "name": "embedding",
            "type": "float[]",
            "num_dim": 384,
            "hnsw_params": { 
                "M": 16,
                "ef_construction": 200
            },
            "vec_dist": "cosine" 
        }
    ]
}

def collection_exists():
    """Check if embeddings collection exists"""
    try:
        client.collections[CUSTOM_SCHEMA["name"]].retrieve()
        return True
    except ObjectNotFound:
        return False

def create_collection():
    client.collections.create(CUSTOM_SCHEMA)
    logger.info(f"Created collection with schema: {CUSTOM_SCHEMA}")
@app.get("/")
def health_check():
    return {"status": "ok", "model": MODEL_PATH}

# Store to Typesense endpoint
@app.post("/store", response_model=StoreResponse)
def store_to_typesense(request: Request):
    
    try:
        embedding = model.encode(
            request.sentences,
            show_progress_bar=True,
            convert_to_numpy=True
        ).mean(axis=0) 
        if not (collection_exists()):
            create_collection()
        
        doc_id = str(uuid.uuid4())#to generate unique doc_ids

        document = {
            'id': doc_id,
            'text': " ".join(request.sentences),
            'sentences': request.sentences,
            'embedding': embedding.tolist()
        }

        client.collections['embeddings_db'].documents.upsert(document)

        return StoreResponse(id=doc_id, status="success")
    
    except Exception as e:
        logger.error(f"Typesense insert failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
