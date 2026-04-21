import streamlit as st
import os
import time
import threading
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from tavily import TavilyClient

load_dotenv()

# DB PATH
DB_DIR = "university_db_app"

st.set_page_config(
    page_title="Marah – University Assistant",
    page_icon="🎓",
    layout="wide",
)

# ───────── CSS ─────────
st.markdown("""
<style>
[data-testid="stChatMessage"] .stMarkdown {
    direction: rtl;
    text-align: right;
}
.typing-cursor {
    display: inline-block;
    width: 6px;
    height: 20px;
    background-color: #0d9488;
    margin-left: 4px;
    animation: blink 1s step-end infinite;
}
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0;} }
</style>
""", unsafe_allow_html=True)

# ───────── LOAD ─────────
@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    retriever = db.as_retriever(search_kwargs={"k": 5})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        streaming=True
    )
    return retriever, llm

retriever, llm = load_components()

# ───────── HEADER ─────────
st.title("🎓 Marah Assistant")

# ───────── CHAT ─────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحبًا! كيف أقدر أساعدك؟")

for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask your question...")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    formatted = ""
    for m in history.messages:
        if m.type == "human":
            formatted += f"User: {m.content}\n"
        else:
            formatted += f"AI: {m.content}\n"
    return formatted

if question:
    st.session_state.chat_history.add_user_message(question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_placeholder = st.empty()
        full_response = ""

        # 🧠 Thinking animation (LIVE)
        thinking = True

        def animate():
            i = 0
            while thinking:
                dots = "." * (i % 3 + 1)
                thinking_placeholder.markdown(
                    f"Thinking{dots}"
                )
                time.sleep(0.3)
                i += 1

        t = threading.Thread(target=animate)
        t.start()

        # ─── Context ───
        db_docs = retriever.invoke(question)
        db_context = format_docs(db_docs)

        web_context = ""
        try:
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            result = tavily.search(query=question)
            web_context = "\n".join([r["content"] for r in result["results"]])
        except:
            pass

        final_context = db_context + "\n" + web_context

        # ─── Prompt ───
        prompt = ChatPromptTemplate.from_template("""
        You are a university assistant called Marah.

        - Detect the language of the question.
        - If English → respond in English.
        - If Arabic → respond in Arabic.
        - Always match user's language.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """)

        chain = prompt | llm | StrOutputParser()

        # ─── Streaming ───
        first_chunk = True

        for chunk in chain.stream({
            "context": final_context,
            "question": question
        }):

            if first_chunk:
                thinking = False
                t.join()
                thinking_placeholder.empty()
                first_chunk = False

            full_response += chunk
            message_placeholder.markdown(
                full_response + '<span class="typing-cursor"></span>',
                unsafe_allow_html=True
            )

        message_placeholder.markdown(full_response)
        st.session_state.chat_history.add_ai_message(full_response)
```
