import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory

# إعداد الصفحة
st.set_page_config(
  page_title="Marah - Stable Assistant", 
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
    # التأكد من وجود مفتاح الـ API
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("⚠️ مفتاح Google API غير موجود في ملف .env")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    db = Chroma(
        persist_directory="./university_db_app",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 15})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    return retriever, llm

retriever, llm = load_components()

# ===== الذاكرة =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحباً بك! 👋 أنا 'مرح'، مساعدك الجامعي الذكي.\n\nأجيب على أسئلتك العامة فقط، ولا يمكنني الوصول لبياناتك الشخصية.")

# ===== UI =====
st.title("🛡️ Marah - University Assistant (Stable)")

for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask your question...")

# ===== منطق "الموجه الآمن" (بدون API Call) =====
def route_question_safely(question: str) -> str:
    """
    دالة ذكية بسيطة تحل محل الوكيل (Agent).
    تقوم بتحليل الكلمات المفتاحية لتقرر هل السؤال شخصي أم لا.
    هذا يمنع أخطاء الاتصال بـ Google API ويجعل التطبيق أسرع.
    """
    # قائمة الكلمات التي تشير لطلب بيانات شخصية
    personal_keywords = [
        "راتب", "مرتب", "حسابي", "كلمة السر", "معدلي", "تخصصي", 
        "درجاتي", "حالة", "قبولي", "طلبي", "تسجيل", "شحنات", 
        "مرسول", "باقة", "رصيد", "مشتريات", "استمارة", 
        "طلب", "طلب التحاق", "رقم الطالب", "المقررات"
    ]
    
    # إذا وُجدت كلمة من هذه القائمة في السؤال -> قم بالحظر
    if any(keyword in question for keyword in personal_keywords):
        return "block"
    
    return "search"

# ===== التشغيل =====
if question:
    st.session_state.chat_history.add_user_message(question)
    with st.chat_message("user"):
        st.markdown(question)

    # استخدام دالة التوجيه الآمن بدلاً من router_chain.invoke
    decision = route_question_safely(question)

    if decision == "block":
        # رد آلي فوري ومنظم
        with st.chat_message("assistant"):
            st.warning("🔒 **تنبيه:** أنا مساعد برمجي ولا أملك صلاحية الوصول لبياناتك الشخصية.")
            st.markdown("""
            للأسئلة المتعلقة بـ:
            - 📜 حالة الطلب / الشحنات
            - 💰 المرتبات / التسهيلات
            - 🎓 الدرجات / المعدل التراكمي
            - 🔑 كلمة المرور
            
            يرجى زيارة [بوابة الطالب الرسمية](https://portal.iugaza.edu.ps) أو التواصل مع عمادة القبول والتسجيل.
            """)
    else:
        # === الخطوة 1: محاولة إعادة كتابة السؤال (Agentic) ===
        # تم وضعها داخل Try/Except لضمان عدم تعطل التطبيق
        with st.spinner("🔍 جاري البحث..."):
            optimized_query = question
            try:
                rewrite_prompt = ChatPromptTemplate.from_template("""
                أعد صياغة السؤال أدناه ليكون مناسباً للبحث في ملفات PDF والجداول.
                أضف كلمات مفتاحية عربية مثل "معدل"، "رسوم"، "سعر الساعة".
                
                السؤال: {question}
                السؤال المحسن:
                """)
                
                rewriter_chain = rewrite_prompt | llm | StrOutputParser()
                optimized_query = rewriter_chain.invoke({"question": question})
                
                with st.expander("🔄 تفاصيل البحث (Agentic)"):
                    st.write(f"**الأصلي:** {question}")
                    st.write(f"**المحسن:** {optimized_query}")
            except Exception as e:
                # إذا فشل إعادة الصياغة، نستخدم السؤال الأصلي (Fallback)
                st.warning("⚠️ فشل تحسين السؤال، سيتم استخدام النص الأصلي للبحث.")
                optimized_query = question

            # === الخطوة 2: البحث ===
            try:
                db_docs = retriever.invoke(optimized_query)
                db_context = format_docs(db_docs)
            except Exception as e:
                st.error(f"حدث خطأ في البحث: {e}")
                db_context = ""

            # === الخطوة 3: صياغة الإجابة ===
            with st.spinner("✍️ جاري كتابة الإجابة..."):
                if not db_docs:
                    final_answer = "عذراً، لم أجد معلومات مطابقة لسؤالك في ملفات الجامعة الرسمية."
                else:
                    prompt = ChatPromptTemplate.from_template("""
                    أنت مساعد جامعي اسمه "مرح". أجب بناءً على السياق فقط.
                    
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

                st.session_state.chat_history.add_ai_message(final_answer)

                with st.chat_message("assistant"):
                    st.markdown(final_answer)
