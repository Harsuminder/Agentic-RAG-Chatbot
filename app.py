import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="SQuAD v2 RAG Chatbot", layout="wide")

st.title("📄 SQuAD v2 RAG Chatbot")
st.caption("Question-Answering System with RAG")

# Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"
PERSIST_DIRECTORY = "./chroma_db"
K_RETRIEVED = 3

@st.cache_resource
def setup_retriever():
    """Load retriever from persistent Chroma index"""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    
    # Embeddings
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Check if persisted index exists
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        st.info(f"📂 Loading existing vector store from {PERSIST_DIRECTORY}...")
        # Load existing vector store
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_model
        )
        st.success("✓ Loaded existing index!")
    else:
        st.error(f"❌ No index found at {PERSIST_DIRECTORY}. Please run notebook.ipynb first to create the index.")
        st.stop()
    
    # Creating retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": K_RETRIEVED}
    )
    return retriever

# Setup retriever
retriever = setup_retriever()

# LLM
from langchain_groq import ChatGroq
llm = ChatGroq(
    model=LLM_MODEL,
    groq_api_key=GROQ_API_KEY,
    temperature=0
)

# RAG prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer the question using ONLY the provided context. "
        "If the answer cannot be found in the context, respond with 'I don't know' or 'The answer is not available in the provided context.'"
    ),
    MessagesPlaceholder("history"),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])

def format_docs(docs):
    """Format retrieved documents for context"""
    formatted = []
    for i, doc in enumerate(docs, 1):
        formatted.append(f"[Source {i}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# RAG Chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

rag_chain = (
    {
        "docs": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history": itemgetter("history")
    }
    | RunnableLambda(lambda x: {
        "question": x["question"],
        "history": x["history"],
        "context": format_docs(x["docs"]),
        "docs": x["docs"],
    })
    | RunnableLambda(lambda x: {
        "answer": (
            prompt
            | llm
            | StrOutputParser()
        ).invoke({
            "question": x["question"],
            "history": x["history"],
            "context": x["context"],
        }),
        "sources": x["docs"],
    })
)

# Session history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Wrap the rag chain with message history
conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="answer"
)

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
    store[st.session_state.session_id] = ChatMessageHistory()
    st.rerun()

# Chat interface
user_input = st.chat_input("Ask a question about SQuAD v2 dataset...")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for i, doc in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}**")
                    st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)

# Handle user input
if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call RAG
    with st.spinner("Thinking..."):
        result = conversational_rag.invoke(
            {"question": user_input},
            config={
                "configurable": {
                    "session_id": st.session_state.session_id
                }
            }
        )

    answer = result["answer"]
    sources = result["sources"]

    # Show assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
        
        # Show sources
        with st.expander("📚 Retrieved Sources"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}**")
                st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
