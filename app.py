import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from tavily import TavilyClient

load_dotenv()

# ────────── Config & CSS ──────────
st.set_page_config(page_title="Marah Assistant", page_icon="🎓", layout="wide")

@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory="university_db_app", embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", 
        search_kwargs={"k": 30} )
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, streaming=True)
    return retriever, llm

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    formatted = ""
    for m in list(history.messages)[-4:]:
        if m.type == "human": formatted += f"Student: {m.content}\n"
        else: formatted += f"Marah: {m.content}\n"
    return formatted

retriever, llm = load_components()

st.title("🎓 Marah Assistant (DEBUG MODE)")
st.markdown("⚠️ في هذه النسخة، سأظهر لك ما عثرت عليه في قاعدة البيانات أسفل الإجابة.")

# Chat Interface
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()

for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask about PDFs...")

if question:
    st.session_state.chat_history.add_user_message(question)
    with st.chat_message("user"): st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 1. البحث
        db_docs = retriever.invoke(question)
        db_context = format_docs(db_docs)

        # 2. التشخيص (هذا هو الجزء المهم)
        # سنعرض النتائج في صندوق قابل للطي
        with st.expander("🔍 DIAGNOSTIC: Check what the bot found (Click Here)"):
            if db_docs:
                st.write(f"Found {len(db_docs)} chunks:")
                for i, doc in enumerate(db_docs):
                    st.markdown(f"**Chunk {i+1} (Source: {doc.metadata.get('source', 'Unknown')}):**")
                    st.text_area(f"Content {i+1}", doc.page_content, height=150)
                    st.markdown("---")
            else:
                st.warning("No documents found in database!")

        # 3. Web Search
        web_context = ""
        try:
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            result = tavily.search(query=question, search_depth="basic")
            if "results" in result:
                web_context = "\n\n".join([r["content"] for r in result["results"][:2]])
        except: pass

        final_context = f"--- PDF Context ---\n{db_context}\n\n--- Web Context ---\n{web_context}"
        history_text = format_history(st.session_state.chat_history)

        # 4. Prompt (أعدت تهددته قليلاً ليكون مرناً إذا لم يجد)
        prompt = ChatPromptTemplate.from_template("""
        أنت مساعد جامعي ذكي.
        أجب على السؤال بناءً على "PDF Context".
        إذا لم تجد الإجابة فيه، استخدم "Web Context".
        إذا لم تجد في أي منهما، قل: "لم أجد هذه المعلومة في ملفات الجامعة".

        {context}
        
        السؤال: {question}
        الإجابة:
        """)

        chain = prompt | llm | StrOutputParser()

        try:
            for chunk in chain.stream({"context": final_context, "question": question}):
                full_response += chunk
                message_placeholder.markdown(full_response)
            st.session_state.chat_history.add_ai_message(full_response)
        except Exception as e:
            st.error(f"Error: {e}")
