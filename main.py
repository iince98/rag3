
# pip install huggingface-hub
# huggingface-cli login

# huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 \
#   --local-dir ./Mistral-7B-Instruct-v0.2 \
#   --local-dir-use-symlinks False
# http://100.100.101.84:8000/query?query=Who is Angela Merkel

# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

from config import MODEL_PATH, EMBEDDING_MODEL_PATH, DATA_PATH, PERSIST_DIRECTORY
from models.model_loader import load_model, load_embeddings
from data.document_loader import load_documents
from data.splitter import split_documents
from data.vectorstore import create_vector_database
from rag.rag_chain import build_qa_chain
import uvicorn

# Declare this globally to be used in the endpoint
qa_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    try:
        tokenizer, model = load_model(MODEL_PATH)
        embeddings = load_embeddings(EMBEDDING_MODEL_PATH)
        documents = load_documents(DATA_PATH)

        if not documents:
            raise RuntimeError("No documents found to initialize the system.")

        chunks = split_documents(documents)
        vector_db = create_vector_database(chunks, embeddings, PERSIST_DIRECTORY)
        qa_chain = build_qa_chain(model, tokenizer, vector_db)
        print("RAG chain initialized.")
        yield
    finally:
        # Optional: cleanup code here if needed
        print("Shutting down.")

app = FastAPI(title="RAG API", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str

@app.get("/query")
def query_rag_get(query: str):
    try:
        result = qa_chain.invoke({"query": query})
        return {
            "query": query,
            "answer": result.get("result", ""),
            "sources": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "snippet": doc.page_content[:500]
                }
                for doc in result.get("source_documents", [])
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
