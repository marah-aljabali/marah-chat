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

load_dotenv()
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
# GLOBAL CSS (ثيم EduPredict + انيميشن لودر + كرسور كتابة)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&display=swap');

/* ── Design tokens ── */
:root {
    --navy:   #0f1f3d;
    --teal:   #0d9488;
    --amber:  #f59e0b;
    --pass:   #10b981;
    --fail:   #ef4444;
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

/* ── Remove Streamlit top padding ── */
.block-container { padding-top: 2rem !important; }

/* ── Header Card (Hero) ── */
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
[data-testid="stChatMessage"] {
    background-color: transparent !important;
}
/* رسائل المستخدم */
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-user"]) {
    display: flex;
    justify-content: flex-end;
}
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-user"]) .stMarkdown {
    background-color: var(--navy);
    color: #fff;
    padding: 12px 20px;
    border-radius: 16px 16px 0 16px;
    box-shadow: var(--shadow);
}
/* رسائل المساعد */
[data-testid="stChatMessage"]:has([data-testid="chat-avatar-assistant"]) .stMarkdown {
    background-color: #fff;
    color: var(--navy);
    padding: 14px 22px;
    border-radius: 16px 16px 16px 0;
    box-shadow: var(--shadow);
    margin-bottom: 10px;
    line-height: 1.6;
}

/* ── Input Box ── */
[data-testid="stChatInput"] {
    background-color: #fff;
    border-radius: 16px;
    padding: 10px;
    box-shadow: var(--shadow);
    border: 1px solid #e2e8f0;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--teal) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

/* ── Initial Splash Screen (بديل اللودر) ── */
.splash-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: linear-gradient(135deg, #f8f7f4 0%, #e2e8f0 100%);
    z-index: 9999;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    animation: fadeIn 0.8s ease-in-out;
}
.splash-icon {
    font-size: 5rem;
    margin-bottom: 20px;
    animation: float 3s ease-in-out infinite;
}
.splash-text {
    font-family: 'DM Serif Display', serif;
    font-size: 2.5rem;
    color: var(--navy);
    margin-bottom: 10px;
}
.splash-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    color: var(--slate);
    letter-spacing: 1px;
}
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-20px); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

/* ── FULL SCREEN OVERLAY (لتحديث قاعدة البيانات) ── */
.fullscreen-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background-color: #ffffff;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    animation: fadeIn 0.5s ease-out;
}
/* انيميشن البناء (Stacking) */
.building-anim {
    display: flex;
    align-items: flex-end;
    justify-content: center;
    gap: 10px;
    height: 60px;
    margin-bottom: 30px;
}
.bar {
    width: 15px;
    background-color: var(--navy);
    animation: stack 1.5s infinite ease-in-out;
    border-radius: 4px 4px 0 0;
}
.bar:nth-child(1) { height: 20px; animation-delay: 0.0s; background-color: var(--navy); }
.bar:nth-child(2) { height: 35px; animation-delay: 0.2s; background-color: var(--teal); }
.bar:nth-child(3) { height: 50px; animation-delay: 0.4s; background-color: var(--amber); }
.bar:nth-child(4) { height: 25px; animation-delay: 0.6s; background-color: var(--navy); }
.bar:nth-child(5) { height: 40px; animation-delay: 0.8s; background-color: var(--teal); }

@keyframes stack {
    0%, 100% { transform: scaleY(1); }
    50% { transform: scaleY(0.4); }
}

.overlay-text {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--navy);
    margin-bottom: 10px;
}
.overlay-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: var(--slate);
}

/* ── Typing Cursor Effect (مؤشر الكتابة) ── */
.typing-cursor {
    display: inline-block;
    width: 6px;
    height: 20px;
    background-color: var(--teal);
    margin-left: 4px;
    animation: blink 1s step-end infinite;
    vertical-align: middle;
}
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }

/* ── Info/Warning Boxes ── */
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: .9rem;
    color: #1e40af;
    margin-bottom: 12px;
}
.warn-box {
    background: #fefce8;
    border-left: 4px solid var(--amber);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: .9rem;
    color: #92400e;
    margin-bottom: 12px;
}

/* ── Sidebar Footer Styling ── */
.sidebar-footer {
    border-top: 1px solid rgba(255,255,255,0.1);
    text-align: center;
}
.sidebar-footer h4 {
    color: var(--teal);
    font-family: 'DM Serif Display', serif;
    margin-bottom: 10px;
}
.sidebar-footer p {
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 5px 0;
}
.sidebar-footer span.highlight {
    color: #fff;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOADING RESOURCES (مع شاشة Splash Screen بدلاً من اللودر العادي)
# ─────────────────────────────────────────────────────────────────────────────

# 1. إظهار شاشة الـ Splash
load_overlay = st.empty()
load_overlay.markdown("""
<div class="splash-screen">
    <div class="splash-icon">🎓</div>
    <div class="splash-text">Marah</div>
    <div class="splash-sub">University Assistant</div>
    <div style="margin-top: 20px; font-size: 0.9rem; color: #0d9488;">Loading AI Models...</div>

    st.markdown("---")
    st.markdown("""
    """<div class="sidebar-footer">
      <h4>Marah Assistant</h4>
      <p><span class="highlight">Marah Ahmed Aljabali</span></p>
      <p>© All Rights Reserved 2026.</p>
    </div> """
    """, unsafe_allow_html=True) 
</div>
""", unsafe_allow_html=True)

# 2. تحميل المكونات (سيتم حجب الواجهة الآن)
@st.cache_resource
def load_components():
    time.sleep(1) # تأخير بسيط لضمان ظهور الأنيميشن
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    db = Chroma(persist_directory="university_db_app", embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return retriever, llm

try:
    retriever, llm = load_components()
    # 3. إزالة شاشة الـ Splash بعد التحميل
    load_overlay.empty()
except Exception as e:
    load_overlay.empty()
    st.error(f"Initialization Failed: {e}")
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# LOGIC & HELPERS
# ─────────────────────────────────────────────────────────────────────────────



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def format_history(history):
    formatted = ""
    for m in history.messages:
        if m.type == "human":
            formatted += f"الطالب: {m.content}\n"
        else:
            formatted += f"مرح: {m.content}\n"
    return formatted

# ─────────────────────────────────────────────────────────────────────────────
# UI: HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="page-hdr">
  <div class="page-hdr-icon">🎓</div>
  <div>
    <div class="page-hdr-title">Marah - University Assistant</div>
    <div class="page-hdr-sub">Ask questions about university courses, departments, and regulations.</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR & SMART UPDATE LOGIC + FOOTER
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ System Status")
    st.success("✅ Database is up to date.")
    
    st.markdown("---")
    st.markdown("### 🔄 Database Management")

    st.metric("Status", "Ready")
 
    st.caption("Database is fresh. ")
        
    # ─── الفوتر في السايدبار (النص الذي طلبته) ───
    st.markdown("---")
    st.markdown("""
    <div class="sidebar-footer">
      <h4>Marah Assistant</h4>
      <p><span class="highlight">Marah Ahmed Aljabali</span></p>
      <p>© All Rights Reserved 2026.</p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FULL SCREEN UPDATE LOGIC
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحبًا!👋 أنا 'مرح'، المساعد الجامعي الذكي للجامعة الإسلامية.\nأنا مستعدة للإجابة عن استفساراتك.")

# عرض الرسائل السابقة
for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

# صندوق الإدخال
question = st.chat_input("Ask your question here...")

# معالجة السؤال
if question:
    st.session_state.chat_history.add_user_message(question)
    with st.chat_message("user"):
        st.markdown(question)

    # رسالة التفكير سريعة جداً (اختياري)
    # with st.spinner("Marah is thinking..."): # إزالة السبينر لجعل الستريمنج يبدأ فوراً

    # إنشاء مكان للرسالة
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # إعداد السياق
        contextual_query = f"السؤال الحالي: {question}\nالسياق: {format_history(st.session_state.chat_history)}"
        db_docs = retriever.invoke(contextual_query)
        db_context = format_docs(db_docs)

        # البحث في الويب
        web_context = ""
        try:
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            result = tavily.search(query=question, search_depth="basic")
            if "results" in result:
                web_context = "\n\n".join([r["content"] for r in result["results"]])
        except:
            pass

        final_context = f"من قاعدة البيانات:\n{db_context}\n\nمن الويب:\n{web_context}"

        # 🎯 Prompt
        prompt = ChatPromptTemplate.from_template("""
        أنت مساعد جامعي اسمه "مرح".

        - أجب بنفس لغة السؤال
        - استخدم أسلوب بسيط وواضح
        - اعتمد على السياق لفهم السؤال
        - قم بتزويد مصادر ومراجع للإجابة إذا تطلب الأمر
        - إذا كانت الإجابة في "قاعدة البيانات"، اعتمد عليها.
        - استخدم موقع الويب الخاص بالجامعة دائمًا للبحث عن إجابة سؤال غير موجودة في الملفات

        السياق:
        {context}

        المحادثة:
        {history}

        السؤال:
        {question}

        الإجابة:
        """)

        chain = prompt | llm | StrOutputParser()
        history_text = format_history(st.session_state.chat_history)

        try:
            # تنفيذ التدفق (Streaming)
            for chunk in chain.stream({"context": final_context, "question": question, "history": history_text}):
                full_response += chunk
                # تحديث الرسالة مع مؤشر الكتابة
                message_placeholder.markdown(full_response + '<span class="typing-cursor"></span>', unsafe_allow_html=True)
            
            # إزالة المؤشر بعد الانتهاء
            message_placeholder.markdown(full_response)
            
            # حفظ الإجابة في الذاكرة
            st.session_state.chat_history.add_ai_message(full_response)

        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
