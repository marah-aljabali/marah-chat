import streamlit as st
import os
import time
import shutil
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from tavily import TavilyClient

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Marah – University Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Design tokens ── */
:root {
    --navy:   #0f1f3d;
    --teal:   #0d9488;
    --amber:  #f59e0b;
    --cream:  #f8f7f4;
    --slate:  #64748b;
    --shadow: 0 4px 24px rgba(15,31,61,.10);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background: var(--cream);
    color: var(--navy);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--navy) !important;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: #94a3b8 !important; }

.block-container { padding-top: 2rem !important; }

/* ── Header Card ── */
.page-hdr {
    background: linear-gradient(120deg, #0f1f3d 0%, #1a3560 60%, #0f5f5a 100%);
    border-radius: 16px;
    padding: 34px 40px;
    margin-bottom: 32px;
    display: flex;
    align-items: center;
    gap: 20px;
    box-shadow: var(--shadow);
}
.page-hdr-icon { font-size: 3rem; }
.page-hdr-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #fff;
    line-height: 1.2;
    margin: 0;
}
.page-hdr-sub { color: #94d5cf; font-size: 1rem; margin-top: 6px; font-weight: 300; }

/* ── Chat Bubbles ── */
[data-testid="stChatMessage"] { background-color: transparent !important; }
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-user"]) { display: flex; justify-content: flex-end; }
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-user"]) .stMarkdown {
    background-color: var(--navy); color: #fff; padding: 12px 20px;
    border-radius: 16px 16px 0 16px; box-shadow: var(--shadow);
}
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-assistant"]) .stMarkdown {
    background-color: #fff; color: var(--navy); padding: 14px 22px;
    border-radius: 16px 16px 16px 0; box-shadow: var(--shadow);
    margin-bottom: 10px; line-height: 1.6;
}

/* ── Input Box ── */
[data-testid="stChatInput"] {
    background-color: #fff; border-radius: 16px; padding: 10px;
    box-shadow: var(--shadow); border: 1px solid #e2e8f0;
}

/* ── Buttons ── */
.stButton > button { background: var(--teal) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 600 !important; }
.stButton > button:hover { opacity: .85 !important; }

/* ── Splash Screen ── */
.splash-screen {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    background: linear-gradient(135deg, #f8f7f4 0%, #e2e8f0 100%);
    z-index: 9999; display: flex; flex-direction: column;
    justify-content: center; align-items: center; animation: fadeIn 0.8s ease-in-out;
}
.splash-icon { font-size: 5rem; margin-bottom: 20px; animation: float 3s ease-in-out infinite; }
.splash-text { font-family: 'DM Serif Display', serif; font-size: 2.5rem; color: var(--navy); margin-bottom: 10px; }
.splash-sub { font-family: 'DM Sans', sans-serif; font-size: 1.1rem; color: var(--slate); letter-spacing: 1px; }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-20px); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* ── Typing Cursor ── */
.typing-cursor {
    display: inline-block; width: 6px; height: 20px; background-color: var(--teal);
    margin-left: 4px; animation: blink 1s step-end infinite; vertical-align: middle;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

/* ── Info/Source Box ── */
.info-box {
    background: #eff6ff; border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0; padding: 12px 16px; font-size: .85rem;
    color: #1e40af; margin-top: 10px; width: 100%;
}
.source-tag {
    background: #e0f2fe; color: #0369a1; padding: 2px 8px;
    border-radius: 4px; font-weight: bold; font-size: 0.8rem; margin-left: 5px;
}

.sidebar-footer {
    border-top: 1px solid rgba(255,255,255,0.1); text-align: center;
    margin-top: auto; padding-bottom: 1rem;
}
.sidebar-footer h4 { color: var(--teal); font-family: 'DM Serif Display', serif; margin-bottom: 10px; font-size: 1rem; }
.sidebar-footer p { font-size: 0.8rem; color: #94a3b8; margin: 5px 0; }
.sidebar-footer span.highlight { color: #fff; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOADING RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

load_overlay = st.empty()
load_overlay.markdown("""
<div class="splash-screen">
    <div class="splash-icon">🎓</div>
    <div class="splash-text">Marah</div>
    <div class="splash-sub">University Assistant (AI Enhanced)</div>
    <div style="margin-top: 20px; font-size: 0.9rem; color: #0d9488;">Loading Database & Models...</div>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_components():
    time.sleep(1) 
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # التحقق من وجود قاعدة البيانات
    if not os.path.exists("university_db_app"):
        return None, None
        
    db = Chroma(persist_directory="university_db_app", embedding_function=embeddings)
    # البحث عن أفضل 4 نتائج
    retriever = db.as_retriever(search_kwargs={"k": 4})
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, streaming=True)
    return retriever, llm

try:
    retriever, llm = load_components()
    load_overlay.empty()
    if retriever is None:
        st.error("❌ قاعدة البيانات غير موجودة! يرجى تشغيل `build_db_app.py` أولاً.")
        st.stop()
except Exception as e:
    load_overlay.empty()
    st.error(f"Initialization Failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# LOGIC & HELPERS
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

def format_docs(docs):
    return "\n\n".join(f"المحتوى: {doc.page_content}\nالمصدر: {doc.metadata.get('source', 'Unknown')}" for doc in docs)

def format_history(history):
    formatted = ""
    for m in history.messages:
        role = "الطالب" if m.type == "human" else "مرح"
        formatted += f"{role}: {m.content}\n"
    return formatted

def get_sources_from_docs(docs):
    """استخراج المصادر الفريدة مع الصفحات"""
    sources = {}
    for doc in docs:
        src = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', '')
        
        # تجاهل روابط الويب لتنظيف العرض (اختياري)
        if "http" in src: continue

        if src not in sources:
            sources[src] = set()
        if page != '':
            sources[src].add(page)
    
    formatted_sources = []
    for src, pages in sources.items():
        page_str = ", ".join(map(str, sorted(list(pages)))) if pages else "الكل"
        formatted_sources.append(f"📄 {src} (ص {page_str})")
    
    return formatted_sources

# ─────────────────────────────────────────────────────────────────────────────
# UI: HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-hdr">
  <div class="page-hdr-icon">🎓</div>
  <div>
    <div class="page-hdr-title">Marah - University Assistant</div>
    <div class="page-hdr-sub">Dedicated answers based on official PDF files & Website data.</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    
    if os.path.exists("last_update.txt"):
        with open("last_update.txt", "r", encoding="utf-8") as f:
            last_update = f.read()
        st.info(f"📅 Last Update: **{last_update}**")
    else:
        st.warning("No Update Log")

    st.metric("Model", "Gemini 1.5 Flash")
    st.metric("Embedding", "HuggingFace (Local)")
    st.success("✅ System Ready")

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-footer">
      <h4>Marah Assistant</h4>
      <p><span class="highlight">Marah Ahmed Aljabali</span></p>
      <p>© All Rights Reserved 2026.</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحبًا!👋 أنا 'مرح'.\nأساعدك في الإجابة على أسئلتك الجامعية باستخدام الملفات الرسمية.")

# عرض الرسائل السابقة
for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask your question here...")

if question:
    st.session_state.chat_history.add_user_message(question)
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 1. البحث في قاعدة البيانات (الملفات)
        db_docs = retriever.invoke(question)
        
        # 2. البحث في الويب (Tavily) - فقط كاحتياط
        web_context = ""
        try:
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            result = tavily.search(query=question, search_depth="basic", max_results=2)
            if "results" in result:
                web_context = "\n\n".join([f"المصدر: {r['url']}\nالمحتوى: {r['content']}" for r in result["results"]])
        except:
            pass

        # تنسيق السياق
        db_context_text = format_docs(db_docs)
        final_context = f"--- بيانات من الملفات الرسمية (الأولوية القصوى) ---\n{db_context_text}\n\n--- بيانات من الويب (للاطلاع فقط) ---\n{web_context}"
        history_text = format_history(st.session_state.chat_history)

        # 🎯 Prompt المطور للدقة
        prompt = ChatPromptTemplate.from_template("""
        أنت مساعد جامعي اسمك "مرح" (Marah). أنت ذكية ومهنية.
        
        قواعد صارمة للإجابة:
        1. لغتك الأساسية هي العربية. إذا سأل بالإنجليزية أجب بالإنجليزية.
        2. **الأولوية للملفات:** ابحث أولاً في قسم "بيانات من الملفات الرسمية". إذا وجدت الإجابة هناك، استخدمها ولا تعتمد على الويب.
        3. **الدقة:** إذا لم تكن الإجابة في الملفات، يمكنك استخدام قسم الويب، ولكن أخبر الطالب بأن المعلومة من الموقع وليست من ملف.
        4. لا تختلق المعلومات.
        
        السياق المتاح:
        {context}
        
        تاريخ المحادثة:
        {history}
        
        السؤال:
        {question}
        
        الإجابة:
        """)

        chain = prompt | llm | StrOutputParser()

        try:
            # حلقة الستريمنج
            for chunk in chain.stream({"context": final_context, "question": question, "history": history_text}):
                full_response += chunk
                message_placeholder.markdown(full_response + '<span class="typing-cursor"></span>', unsafe_allow_html=True)
            
            message_placeholder.markdown(full_response)
            
            # حفظ المصادر لعرضها
            sources = get_sources_from_docs(db_docs)
            
            if sources:
                st.markdown(f"""
                <div class="info-box">
                    <strong>📚 المصادر:</strong><br>
                    {"<br>".join(sources)}
                </div>
                """, unsafe_allow_html=True)
            
            st.session_state.chat_history.add_ai_message(full_response)

        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
