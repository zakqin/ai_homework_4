import whisper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from db import get_vectorstore


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150
)


def ingest_pdf(file_path: str):
    """Load a PDF from disk, split its text into chunks, and add them to the vector store."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    split_docs = splitter.split_documents(docs)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(split_docs)
    vectorstore.persist()


def ingest_media(file_path: str):
    """Transcribe an audio/media file with Whisper, chunk the text, and add it to the vector store."""
    model = whisper.load_model("base")
    result = model.transcribe(file_path)

    doc = Document(
        page_content=result["text"],
        metadata={"source": "media"}
    )

    split_docs = splitter.split_documents([doc])

    vectorstore = get_vectorstore()
    vectorstore.add_documents(split_docs)
    vectorstore.persist()