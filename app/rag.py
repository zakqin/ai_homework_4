from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from db import get_vectorstore

# When context is empty (no embeddings), the model is instructed to answer from general knowledge
# instead of replying "I don't know".
QA_PROMPT = PromptTemplate(
    template="""Use the following context to answer the question if it is relevant.
If the context is empty or does not contain relevant information, answer the question using your general knowledge.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


def get_qa_chain():
    """Create a RetrievalQA chain wired to the Chroma vector store and Gemini chat model."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )


def ask_question(question: str):
    """Run a retrieval-augmented QA chain for a given natural-language question."""
    qa = get_qa_chain()
    result = qa({"query": question})
    return result["result"]