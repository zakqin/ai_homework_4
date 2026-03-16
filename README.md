# RAG Chatbot Backend

This project is a simple RAG (Retrieval-Augmented Generation) backend built with FastAPI.  
It can ingest PDFs and audio files, store chunked text in a Chroma vector database, and answer questions using a Gemini LLM.

---

## Requirements

- Python 3.11+
- A Google AI (Gemini) API key

Environment variable:

- `GOOGLE_API_KEY` – your Gemini API key


## Running with Docker & Docker Compose

### 1. Set your Gemini API key

```bash
export GOOGLE_API_KEY="..."
```

### 2. Build and start the stack

From the project root:

```bash
docker compose up --build
```

This will:

- Build the `rag-api` image from `docker.Dockerfile`
- Start the FastAPI backend on `http://localhost:8000`
- Start `open-webui` on `http://localhost:3000`
- Mount persistent volumes for:
  - `chroma_db` at `/app/chroma_db`
  - uploads at `/app/uploads`

To stop:

```bash
docker compose down
```

---

## API Endpoints

### 1. Document Ingestion

Add content to your knowledge base by uploading files.

#### **Upload PDF**

Extracts text, chunks it into 500-character segments, and stores it in the vector database.

* **Endpoint:** `POST /upload/pdf`
* **Type:** `multipart/form-data`

```bash
curl -X POST "http://localhost:8000/upload/pdf" \
  -F "file=@/path/to/report.pdf"

```

#### **Upload Media**

Transcribes audio/video using OpenAI Whisper ("base" model) and ingests the text.

* **Endpoint:** `POST /upload/media`
* **Type:** `multipart/form-data`

```bash
curl -X POST "http://localhost:8000/upload/media" \
  -F "file=@/path/to/recording.mp3"

```

---

### 2. Retrieval-Augmented Chat

Query your data using an interface compatible with **Open Web UI** or any OpenAI client.

#### **Chat Completions**

Performs a similarity search on your documents and uses **Gemini 3 Flash** to generate a response.

* **Endpoint:** `POST /v1/chat/completions`

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-rag-model",
    "messages": [
      {"role": "user", "content": "What are the key findings in the uploaded PDF?"}
    ]
  }'

```

---

### 3. Database Management

Inspect or clear the stored embeddings.

#### **List All Embeddings**

Returns all stored chunks, their metadata, and their raw vector arrays.

* **Endpoint:** `GET /embeddings`

```bash
curl -X GET "http://localhost:8000/embeddings"

```

#### **Reset Vector Store**

Clears all documents from the database without deleting the physical directory.

* **Endpoint:** `POST /embeddings/reset`

```bash
curl -X POST "http://localhost:8000/embeddings/reset"

```

---

### 4. System Discovery

Used for integration with external UI tools.

#### **List Models**

Returns the available RAG model ID.

* **Endpoint:** `GET /v1/models`

```bash
curl -X GET "http://localhost:8000/v1/models"

```

---

## 🛠️ Technical Specifications

| Component | Technology |
| --- | --- |
| **LLM** | `gemini-3-flash-preview` |
| **Embeddings** | `models/gemini-embedding-001` |
| **Vector Store** | Chroma DB (Persistent) |
| **Transcription** | OpenAI Whisper (Base) |
| **Text Splitting** | `RecursiveCharacterTextSplitter` (Chunk: 500, Overlap: 150) |

---

## ⚙️ Setup

1. **Environment Variables:** Ensure your `GOOGLE_API_KEY` is exported in your environment.
2. **Directories:** The system automatically creates `./uploads` and `./chroma_db`.
---

## Common Issues

- **Authentication errors**: double-check that `GOOGLE_API_KEY` is set and available to the process (or container) running the API.

