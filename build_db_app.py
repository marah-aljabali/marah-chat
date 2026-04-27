import os
import requests
from bs4 import BeautifulSoup, SoupStrainer
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import hashlib
import datetime
import shutil
import re # <--- تمت الإضافة لتنظيف النصوص

# ========= إعدادات =========
load_dotenv()

DATA_PATH = "data/pdfs"
DB_PATH = "university_db_app"
SITEMAP_URL = "https://www.iugaza.edu.ps/wp-sitemap.xml"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ========= دالة لتنظيف النص (من الكود 2 لحل مشكلة تكسر النصوص) =========
def clean_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========= جلب الروابط (باستخدام المنطق الذي تفضله) =========
def get_website_urls_from_sitemap(sitemap_url):
    """
    دالة لجلب الروابط.
    ملاحظة: أنت عرفت دالة get_all_urls_from_sitemap بالأعلى ولكنك لم تستخدمها،
    فاحتفظت بهذه الدالة لتنسجم مع منطقك في build_database.
    """
    print("🗺️ جاري جلب خريطة الموقع...")
    try:
        response = requests.get(sitemap_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        
        # فلترة الروابط
        valid_urls = [url for url in urls if url.startswith(UNIVERSITY_BASE_URL)]
        
        # استبعاد الروابط غير المرغوبة (Tags, Authors, etc.)
        skip_keywords = ["tag", "author", "feed", "comment", "replytocom"]
        final_urls = [u for u in valid_urls if not any(s in u for s in skip_keywords)]
        
        print(f"✅ تم العثور على {len(final_urls)} رابط صالح.")
        return final_urls
    except Exception as e:
        print(f"❌ خطأ في جلب خريطة الموقع: {e}")
        # قائمة احتياطية
        return [
            f"{UNIVERSITY_BASE_URL}/",
            f"{UNIVERSITY_BASE_URL}/aboutiug/",
            f"{UNIVERSITY_BASE_URL}/facalties/",
            f"{UNIVERSITY_BASE_URL}/news/",
        ]

# ========= hash =========
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# ========= بناء قاعدة البيانات (الكود المصلح) =========
def build_database():
    print("🚀 بدء بناء قاعدة المعرفة...")
    start_time = datetime.datetime.now()

    all_documents = []

    # ===== 🌐 تحميل الموقع =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    
    # تحديد عدد الروابط
    urls = urls[:200] 
    
    if urls:
        print(f"📥 جاري تحميل المحتوى من {len(urls)} صفحة ويب...")
        try:
            web_loader = WebBaseLoader(
                urls,
                continue_on_failure=True,
                requests_per_second=1, 
                bs_kwargs={"parse_only": SoupStrainer("body")}
            )
            web_documents = web_loader.load()
            
            # تنظيف نصوص الويب
            for doc in web_documents:
                doc.page_content = clean_text(doc.page_content)
                
            all_documents.extend(web_documents)
            print(f"✅ تم تحميل {len(web_documents)} وثيقة ويب.")
        except Exception as e:
            print(f"⚠️ خطأ تحميل الويب: {e}")

    # ===== 📄 تحميل PDF (تم الإصلاح هنا) =====
    # 1. تحويل المسار إلى مسار مطلق لضمان وجوده
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_path = os.path.join(current_dir, DATA_PATH)

    if os.path.exists(absolute_data_path):
        print("📥 جاري تحميل ملفات PDF...")
        try:
            # 2. إضافة silent_errors=True لتجاوز الملفات التالفة
            pdf_loader = DirectoryLoader(absolute_data_path, loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()

            for doc in pdf_docs:
                doc.metadata["source"] = "pdf"
                # 3. تنظيف نصوص الـ PDF
                doc.page_content = clean_text(doc.page_content)

            print(f"✅ تم تحميل {len(pdf_docs)} صفحة من ملفات PDF")
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"❌ خطأ عام أثناء قراءة ملفات PDF: {e}")
    else:
        print(f"⚠️ مجلد PDF غير موجود: {absolute_data_path}")

    if not all_documents:
        print("❌ لا توجد بيانات للمعالجة!")
        return

    # ===== ✂️ تقسيم =====
    print("✂️ تقسيم النصوص...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # زيادة الحجم لتحسين القراءة (كما في الكود 2)
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_documents)
    print(f"✅ عدد الأجزاء: {len(chunks)}")

    # ===== 🧠 Embeddings =====
    print(f"🧠 تحميل نموذج التضمين: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ===== 💾 بناء DB =====
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("💾 بناء قاعدة البيانات...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    # ===== 🕒 حفظ آخر تحديث =====
    with open("last_update.txt", "w", encoding="utf-8") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"🎉 تم بناء قاعدة البيانات بنجاح! (المدة: {duration:.2f} ثانية)")

# ========= تشغيل =========
if __name__ == "__main__":
    build_database()
