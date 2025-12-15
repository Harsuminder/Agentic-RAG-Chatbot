import streamlit as st

st.set_page_config(page_title="Agentic RAG Chatbot", layout="wide")

st.title("📄 Conversational RAG Chatbot")
st.caption("Transformer paper • RAG • Session memory")

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq

groq_api_key= os.getenv("GROQ_API_KEY")


# Load Document 
from langchain_community.document_loaders import WebBaseLoader

URL = "https://arxiv.org/html/1706.03762"

loader = WebBaseLoader(URL)
documents = loader.load()

# Chunking
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunks= text_splitter.split_documents(documents)

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embedding_model= HuggingFaceEmbeddings (model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(documents=chunks,embedding=embedding_model)

# Creating retriever

retriever= vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

# LLM
from langchain_groq import ChatGroq

llm= ChatGroq(model="openai/gpt-oss-20b", groq_api_key= groq_api_key, temperature= 0)

# RAG prompt 

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt= ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. "
            "Answer the question using ONLY the provided context. "
            "Cite sources in your answer using [Source 1], [Source 2], etc. "
            "If the answer is not in the context, say you don't know."
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            "Context:\n{context}\n\nQuestion:\n{question}"
        )
    ]
)


def format_docs(docs):
    formatted=[]
    for i, doc in enumerate(docs,1):
        formatted.append(f"[source {i}\n{doc.page_content}")
    return "\n\n".join(formatted)

# RAG Chain
'''Any normal Python function can be used in a LangChain chain if and only if it is wrapped as a Runnable
(usually with RunnableLambda) and correctly returns an output.'''
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from operator import itemgetter

rag_chain=(
    {
        "docs": itemgetter("question")|retriever, # ans from retriever wrt to question asked        
        "question": itemgetter("question"),                                       
        "history":itemgetter("history")
    }
    | RunnableLambda(lambda x:{
        "question": x["question"],
        "history": x["history"],
        "context": format_docs(x["docs"]),
        "docs": x["docs"],
    })
    |RunnableLambda(lambda x:{
        "answer":(
             prompt
            |llm
            |StrOutputParser()
        ).invoke({
            "question": x["question"],
            "history": x["history"],
            "context": x["context"],
        }),
        "sources": x["docs"],
    })
   
)

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

#in-memory store
store = {}

def get_session_history(session_id: str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


#Wrap the rag chain with message history
conversational_rag= RunnableWithMessageHistory(
    rag_chain,
    get_session_history, 
    input_messages_key="question",
    history_messages_key="history",
    output_messages_key="answer"
    )

user_input = st.chat_input("Ask a question about the Transformer paper...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call RAG
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
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)

        # Optional: expandable sources
        with st.expander("Sources"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}**")
                st.write(doc.page_content[:500])



