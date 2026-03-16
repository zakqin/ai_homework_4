from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

PERSIST_DIR = "./chroma_db"
_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def get_vectorstore():
    """Return a Chroma vector store instance backed by a singleton Gemini embeddings object and persisted on disk."""
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=_embeddings)


def reset_vectorstore():
    """Clear all documents from the Chroma vector store (avoids deleting the directory while it is in use)."""
    vectorstore = get_vectorstore()
    data = vectorstore.get()
    ids = data.get("ids", [])
    if ids:
        vectorstore.delete(ids)


def get_all_embeddings():
    """Return all stored embeddings and associated metadata/documents from the Chroma vector store."""
    vectorstore = get_vectorstore()
    # `ids` are always returned and must NOT be listed in `include`
    data = vectorstore.get(include=["documents", "metadatas", "embeddings"])
    # Normalize response to a list of records for easier JSON consumption
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    embeddings = data.get("embeddings", [])

    def _to_serializable_embedding(emb):
        """Convert embedding to a plain list of floats if needed."""
        if emb is None:
            return None
        # Chroma often returns numpy arrays; use `tolist` when available
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        # Ensure it's a basic list (e.g. if it's a tuple or other sequence)
        return [float(x) for x in emb]

    records = []
    for idx, _id in enumerate(ids):
        embedding = embeddings[idx] if idx < len(embeddings) else None
        document = documents[idx] if idx < len(documents) else None
        metadata = metadatas[idx] if idx < len(metadatas) else None

        record = {
            "id": _id,
            "document": document,
            "metadata": dict(metadata) if isinstance(metadata, dict) else metadata,
            "embedding": _to_serializable_embedding(embedding),
        }
        records.append(record)

    return records
