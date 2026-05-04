import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAIError # استدعاء الخطأ

# إعداد الصفحة
st.set_page_config(
  page_title="Marah - Robust Assistant", 
  page_icon="🛡️",
  layout="centered")

# تحميل البيئة
load_dotenv()

# ===== Helpers =====
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

# ===== تحميل الموارد =====
@st.cache_resource
def load_components():
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = Chroma(
        persist_directory="./university_db_app",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 12})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    return retriever, llm

retriever, llm = load_components()

# ===== الذاكرة =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحباً بك! 👋 أنا 'مرح'، مساعدك الجامعي الذكي.\n\nأجيب من ملفات الجامعة الرسمية فقط.")

# ===== UI =====
st.title("🛡️ Marah - University Assistant (Stable)")

for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask your question...")

# ===== منطق "الموجه الآمن" (بدون API Call) =====
def route_question_safely(question: str) -> str:
    """تصفية الأسئلة الشخصية لتجنب الهلوسة وتقليل الطلبات غير الضرورية."""
    personal_keywords = [
        "راتب", "مرتب", "حسابي", "كلمة السر", "معدلي", "تخصصي", 
        "درجاتي", "حالة", "قبولي", "طلبي", "تسجيل", "شحنات", 
        "مرسول", "باقة", "رصيد", "مشتريات", "استمارة", 
        "طلب", "طلب التحاق", "رقم الطالب", "المقررات"
    ]
    
    if any(keyword in question for keyword in personal_keywords):
        return "block"
    return "search"

# ===== التشغيل =====
if question:
    st.session_state.chat_history.add_user_message(question)
    with st.chat_message("user"):
        st.markdown(question)

    decision = route_question_safely(question)

    if decision == "block":
        # رد آلي ومحسن لغوياً
        with st.chat_message("assistant"):
            st.warning("🔒 **تنبيه هام:** أنا نظام مساعد ولا أملك صلاحية الدخول للبيانات الخاصة بك (مثل الحالة، الراتب، أو الدرجات).")
            st.markdown("""
            **للاستفسارات الشخصية يرجى زيارة:**
            - [بوابة الطالب](https://portal.iugaza.edu.ps)
            - عمادة القبول والتسجيل
            """)
    else:
        # === منطق البحث والإجابة مع حماية من الأعطال ===
        with st.spinner("🔍 جاري البحث والتفكير..."):
            
            # 1. محاولة تحسين السؤال (اختياري)
            optimized_query = question
            try:
                rewriter_prompt = ChatPromptTemplate.from_template("""
                أعد صياغة السؤال ليكون مناسبًا للبحث في ملفات الجامعة (خاصة الجداول).
                السؤال: {question}
                المحسن:
                """)
                rewriter_chain = rewriter_prompt | llm | StrOutputParser()
                optimized_query = rewriter_chain.invoke({"question": question})
            except Exception:
                # إذا فشل تحسين السؤال، لا نتوقف، نستخدم الأصلي
                st.info("⚠️ تم استخدام البحث المباشر (فشل التحسين).")
                optimized_query = question

            # 2. البحث
            try:
                db_docs = retriever.invoke(optimized_query)
                db_context = format_docs(db_docs)
            except Exception:
                db_context = ""

            # 3. الإجابة (هنا وضعنا الحماية ضد الـ Crash)
            try:
                if not db_docs:
                    final_answer = "عذراً، لم أجد معلومات مطابقة لسؤالك في ملفات الجامعة الرسمية."
                else:
                    # استخدام System Prompt بسيط وفعال لتقليل الأخطاء
                    prompt = ChatPromptTemplate.from_template("""
                    أنت مساعد جامعي. أجب بناءً على السياق فقط.
                    إذا لم تكن الإجابة واضحة، قل "لا أعلم".
                    السياق:
                    {context}
                    السؤال:
                    {question}
                    الإجابة:
                    """)
                    
                    chain = prompt | llm | StrOutputParser()
                    final_answer = chain.invoke({
                        "context": db_context,
                        "question": question
                    })

                # حفظ وعرض الإجابة
                st.session_state.chat_history.add_ai_message(final_answer)
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                    
                    # زر للمصادر
                    if db_docs:
                        with st.expander("🔍 المصادر"):
                            for doc in db_docs[:3]:
                                st.text(doc.metadata.get("source", "File"))

            # === هنا نلتقط الـ Error (ChatGoogleGenerativeAIError) ===
            except ChatGoogleGenerativeAIError as e:
                st.error("🔴 حدث خطأ مؤقت في الاتصال بالذكاء الاصطناعي.")
                st.markdown("**هذا قد يكون بسبب:**)
                1. ضغط عالي على السيرفر (حاول مرة أخرى بعد ثواني).
                2. مشكلة في النص المستخرج من الملفات.")
                
                # نضيف رسالة في التاريخ لتوضيح أن الخطأ تقني وليس ذكي
                error_msg = "عذراً، حدث خطأ أثناء محاولة الإجابة على سؤالك، يرجى المحاولة مرة أخرى."
                st.session_state.chat_history.add_ai_message(error_msg)
                with st.chat_message("assistant"):
                    st.write(error_msg)
                    
            except Exception as e:
                st.error(f"حدث خطأ غير متوقع: {e}")
