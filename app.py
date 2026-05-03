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

# إعداد الصفحة
st.set_page_config(
  page_title="Marah - University Assistant", 
  page_icon="🎓",
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

def get_last_messages(history, n=3):
    return history.messages[-n*2:]

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

    # 🔧 التعديل الأول: زيادة عدد النتائج من 5 إلى 12
    # هذا يساعد في التقاط معلومات الجداول المبعثرة في ملفات PDF
    retriever = db.as_retriever(search_kwargs={"k": 12})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # التأكد من استخدام المودل الصحيح والمتاح
        temperature=0
    )

    return retriever, llm

retriever, llm = load_components()

# ===== الذاكرة =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحباً بك! 👋 أنا 'مرح'، مساعدك الجامعي الذكي. \n\nيمكنني الإجابة على أسئلتك بناءً على بيانات الجامعة وموقعها الإلكتروني. كيف يمكنني مساعدتك اليوم؟")

# ===== UI =====
st.title("🎓 Marah - University Assistant")

for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask your question...")

# ===== التشغيل =====
if question:
    st.session_state.chat_history.add_user_message(question)

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("مرح تفكر..."):
        try:
            # 🧠 Query مع سياق
            contextual_query = f"""
            السؤال الحالي: {question}
            السياق: {format_history(st.session_state.chat_history)}
            """

            db_docs = retriever.invoke(contextual_query)
            db_context = format_docs(db_docs)

            # 🌐 Fallback للويب (اختياري، لكن سنركز على قاعدة البيانات)
            # ملاحظة: بما أن البيانات موجودة في الـ PDFs، لن نحتاج كثيراً للويب
            # لكن تم ترك الكود هنا كاحتياط
            web_context = ""
            if not db_docs:
                try:
                    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
                    result = tavily.search(query=question)
                    if "results" in result:
                        web_context = "\n\n".join([r["content"] for r in result["results"]])
                except:
                    pass

            final_context = f"""
            من قاعدة البيانات:
            {db_context}

            من الويب:
            {web_context}
            """

            # 🔧 التعديل الثاني: تحسين الـ Prompt لفهم الجداول
            prompt = ChatPromptTemplate.from_template("""
            أنت مساعد جامعي اسمه "مرح".

            - أجب بنفس لغة السؤال).
            - السياق أدناه مأخوذ من ملفات PDF وموقع الجامعة. قد يكون النص غير منظم (مثل الجداول المبعثرة).
            - مهمة: ركّز بشدة على الأرقام والعلاقة بين اسم التخصص والأرقام المجاورة له.
            - إذا كان السؤال عن "معدل القبول" أو "سعر الساعة" أو "مفتاح التنسيق"، ابحث عن الرقم المرتبط بالتخصص المذكور حتى لو كان النص مترابطاً.
            - إذا لم تجد الإجابة في السياق الموجود أدناه، قل بوضوح: "عذراً، لم أجد هذه المعلومة في البيانات الحالية".
            
            السياق:
            {context}

            السؤال:
            {question}

            الإجابة:
            """)

            chain = prompt | llm | StrOutputParser()

            history_text = format_history(
                type("obj", (object,), {
                    "messages": get_last_messages(st.session_state.chat_history)
                })
            )

            answer = chain.invoke({
                "context": final_context,
                "question": question
            })

            st.session_state.chat_history.add_ai_message(answer)

            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
