import logging
import os
import shutil
from fastapi import FastAPI, UploadFile, File

logging.basicConfig(level=logging.INFO)
from pydantic import BaseModel
from ingest import ingest_pdf, ingest_media
from rag import ask_question
from db import reset_vectorstore, get_all_embeddings

UPLOAD_DIR = "./uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()


# OpenAI-compatible schemas for Open Web UI
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file, extract and chunk its text, and store it in the vector database."""
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_pdf(file_path)

    return {"status": "PDF ingested successfully"}


@app.post("/upload/media")
async def upload_media(file: UploadFile = File(...)):
    """Upload an audio/media file, transcribe it with Whisper, and store chunks in the vector database."""
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ingest_media(file_path)

    return {"status": "Media ingested successfully"}


@app.post("/embeddings/reset")
def reset_db():
    """Clear the persisted Chroma vector database on disk."""
    reset_vectorstore()
    return {"status": "Vector DB cleared"}


@app.get("/embeddings")
def list_embeddings():
    """Return all embeddings stored in the Chroma vector database."""
    embeddings = get_all_embeddings()
    return {"embeddings": embeddings}


@app.get("/models")
@app.get("/v1/models")
def list_models():
    """OpenAI-style model list so Open Web UI can connect."""
    return {
        "object": "list",
        "data": [
            {
                "id": "test-rag-model",
                "object": "model",
                "created": 1686935002,
                "owned_by": "maksym-krutik",
            }
        ],
    }


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    """OpenAI-style chat completions; runs RAG on the last user message."""
    user_content = ""
    for m in reversed(req.messages):
        if m.role == "user":
            user_content = m.content
            break
    answer = ask_question(user_content or "")
    return {
        "id": "chatcmpl-rag",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
    }
