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
  page_title="Marah - Smart Agentic Assistant", 
  page_icon="🧠",
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

    # استرجاع عدد كبير للسماح للـ Agent باختيار الأفضل
    retriever = db.as_retriever(search_kwargs={"k": 15})

    # نموذج سريع وكفؤ للتفكير والبحث
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )

    return retriever, llm

retriever, llm = load_components()

# ===== الذاكرة =====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = InMemoryChatMessageHistory()
    st.session_state.chat_history.add_ai_message("مرحباً بك! 👋 أنا 'مرح'، مساعدك الجامعي الذكي.\n\nأقوم بتحليل أسئلتك بدقة لأمنحك إجابات من الملفات الرسمية فقط.")

# ===== UI =====
st.title("🧠 Marah - University Assistant (Smart RAG)")

for msg in st.session_state.chat_history.messages:
    with st.chat_message(msg.type):
        st.markdown(msg.content)

question = st.chat_input("Ask your question...")

# ===== منطق الـ Agents الذكي =====
if question:
    st.session_state.chat_history.add_user_message(question)
    with st.chat_message("user"):
        st.markdown(question)

    # === الخطوة 1: الـ Router Agent (من فكرة Auto-RAG) ===
    # هذه الخطوة تحميك من الهلوسة في البيانات الشخصية
    router_prompt = ChatPromptTemplate.from_template("""
    أنت موجه (Router) لنظام جامعي. مهمتك فقط تصنيف السؤال.
    
    أجب بكلمة واحدة فقط إما "search" أو "block":
    
    1. قل "block" إذا كان السؤال يطلب:
       - حالة طلب (طلب، شحنة، مرسول)
       - راتب أو مرتب مالي
       - معدل تراكمي خاص بالطالب
       - كلمة السر
       - حالة قبول شخصية
       - أي بيانات خاصة بحساب الطالب
       
    2. قل "search" إذا كان السؤال عاماً أو أكاديمياً (مثل: القبول، الرسوم، المعدلات، الأنظمة، الكليات).
    
    السؤال: {question}
    Classification:
    """)
    
    with st.spinner("🛡️ جاري تحليل السؤال..."):
        router_chain = router_prompt | llm | StrOutputParser()
        decision = router_chain.invoke({"question": question}).strip().lower()

    if decision == "block":
        # رد آلي فوري لمنع الهلوسة
        with st.chat_message("assistant"):
            st.warning("🔒 **تنبيه هام:** أنا مساعد برمجي ولا أملك صلاحية الوصول لبياناتك الشخصية أو حالات طلباتك.")
            st.markdown("للتعرف على حالتك، يرجى زيارة [بوابة الطالب](https://portal.iugaza.edu.ps) أو التواصل مع عمادة القبول والتسجيل.")
    else:
        # === الخطوة 2: البحث والاسترجاع ===
        with st.spinner("🔍 جاري البحث في قاعدة المعرفة..."):
            # أولاً: محاولة إعادة كتابة السؤال للبحث الجدولي
            rewriter_prompt = ChatPromptTemplate.from_template("""
            أعد كتابة السؤال أدناه لبحث دقيق في الجداول (مثل: معدل قبول، سعر ساعة).
            أضف كلمات مفتاحية عربية.
            السؤال: {question}
            المحسن:
            """)
            rewriter_chain = rewriter_prompt | llm | StrOutputParser()
            optimized_query = rewriter_chain.invoke({"question": question})

            # البحث
            db_docs = retriever.invoke(optimized_query)
            db_context = format_docs(db_docs)

            # عرض السؤال المحسن (اختياري للشفافية)
            with st.expander("🔄 تفكير النظام"):
                st.write(f"**القرار:** {'مسموح بالبحث' if decision == 'search' else 'مرفوض (بيانات شخصية)'}")
                st.write(f"**استعلام البحث:** {optimized_query}")
                st.write(f"**النتائج المسترجعة:** {len(db_docs)} جزء نصي")

            # === الخطوة 3: Verification Agent (من فكرة Agentic RAG) ===
            # هذا الوكيل يتحقق: هل المعلومة موجودة فعلاً في النتائج؟
            verifier_prompt = ChatPromptTemplate.from_template("""
            أنت مدقق دقيق. لديك سياق من ملفات جامعية وسؤال من طالب.
            
            مهمتك:
            1. اقرأ السياق بدقة.
            2. هل الإجابة المحددة (رقم، نسبة، تاريخ) موجودة في السياق؟
            3. إذا كانت موجودة، استخرجها واكتبها.
            4. إذا لم تكن موجودة، أو كان السياق غامضاً جداً، أكتب فقط: "معلومة غير موجودة".
            
            لا تخترع أرقاماً. إجابتها تعتمد 100% على السياق.
            
            السياق:
            {context}
            
            السؤال:
            {question}
            
            الإجابة:
            """)
            
            verifier_chain = verifier_prompt | llm | StrOutputParser()
            
            with st.spinner("🧠 التحقق من دقة المعلومة..."):
                # إذا لم نجد نتائج، لا نحتاج للتحقق
                if not db_docs:
                    final_answer = "عذراً، لم أجد معلومات مطابقة لسؤالك في ملفات الجامعة."
                else:
                    # نجري التحقق
                    final_answer = verifier_chain.invoke({
                        "context": db_context,
                        "question": question
                    })
                    
                    # فحص إضافي بسيط لمنع الـ Agents من المبالغة
                    if "غير موجودة" in final_answer or "لم أجد" in final_answer:
                        final_answer = "عذراً، لم أجد هذه المعلومة بشكل دقيق في البيانات الحالية."

            st.session_state.chat_history.add_ai_message(final_answer)

            with st.chat_message("assistant"):
                st.markdown(final_answer)
                
                # زر لطيف لتوضيح المصدر
                if "غير موجودة" not in final_answer:
                    st.caption("ℹ️ المصدر: الملفات الرسمية للجامعة وموقعها الإلكتروني")
