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

# ========= إعدادات =========
load_dotenv()

DATA_PATH = "data/pdfs"
DB_PATH = "university_db_app"
SITEMAP_URL = "https://www.iugaza.edu.ps/wp-sitemap.xml"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ========= تنظيف النصوص (جديد) =========
def clean_text(text):
    """تنظيف النصوص من المسافات الزائدة والأسطر الفارغة لزيادة دقة البحث"""
    text = text.replace('\r', '\n')
    # إزالة الأسطر المتكررة
    lines = text.split('\n')
    unique_lines = []
    prev_line = ""
    for line in lines:
        stripped = line.strip()
        if stripped and stripped != prev_line:
            unique_lines.append(stripped)
            prev_line = stripped
    return " ".join(unique_lines)

# ========= جلب الروابط (محسّن) =========
def get_website_urls_from_sitemap(sitemap_url):
    print("🗺️ جاري جلب خريطة الموقع...")
    try:
        response = requests.get(sitemap_url, timeout=15)
        response.raise_for_status() 
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        valid_urls = [url for url in urls if url.startswith(UNIVERSITY_BASE_URL)]
        
        # فلترة روابط غير مهمة
        skip_keywords = ["tag", "author", "feed", "category", "page"]
        valid_urls = [u for u in valid_urls if not any(s in u for s in skip_keywords)]
        
        print(f"✅ تم العثور على {len(valid_urls)} رابط صالح.")
        return valid_urls
    except Exception as e:
        print(f"❌ خطأ في جلب خريطة الموقع: {e}")
        # قائمة احتياطية
        return [f"{UNIVERSITY_BASE_URL}/", f"{UNIVERSITY_BASE_URL}/aboutiug/", f"{UNIVERSITY_BASE_URL}/facalties/"]

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 بدء بناء قاعدة المعرفة (الإصدار المطور)...")
    start_time = datetime.datetime.now()

    all_documents = []

    # ===== 🌐 تحميل الموقع (أخف سرعة لتجنب الحظر) =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    # تقليل العدد للسرعة وعدم الزحح المفرط (يمكنك زيادة الرقم إذا أردت)
    urls = urls[:150] 
    
    if urls:
      print("📥 جاري تحميل المحتوى من الويب (قد يستغرق وقتاً)...")
      web_loader = WebBaseLoader(
          urls,
          continue_on_failure=True,
          requests_per_second=1, # إبطاء الطلبات
          bs_kwargs={"parse_only": SoupStrainer("body")}
      )
      web_documents = web_loader.load()
      # تنظيف نصوص الويب
      for doc in web_documents:
          doc.page_content = clean_text(doc.page_content)
          doc.metadata["source"] = "Website"
      
      print(f"✅ تم تحميل {len(web_documents)} صفحة ويب.")
      all_documents.extend(web_documents)

    # ===== 📄 تحميل PDF (الأولوية القصوى) =====
    if os.path.exists(DATA_PATH):
        print("📥 تحميل ومعالجة ملفات PDF...")
        pdf_loader = DirectoryLoader(DATA_PATH, loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()

        for doc in pdf_docs:
            doc.page_content = clean_text(doc.page_content)
            # التأكد من حفظ المصدر كاسم ملف فقط وليس المسار الكامل
            filename = os.path.basename(doc.metadata.get('source', 'Unknown'))
            doc.metadata["source"] = filename
            doc.metadata["type"] = "pdf" 

        print(f"✅ تم تحميل {len(pdf_docs)} صفحة من ملفات PDF.")
        all_documents.extend(pdf_docs)
    else:
        print("⚠️ مجلد data/pdfs غير موجود.")

    if not all_documents:
        print("❌ لا توجد بيانات للمعالجة!")
        return

    # ===== ✂️ تقسيم (مطوّر لحجم أكبر) =====
    print("✂️ تقسيم النصوص إلى أجزاء ذكية...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # زيادة الحجم للحصول على سياق أفضل
        chunk_overlap=200,     # تداخل لعدم فقدان السياق
        separators=["\n\n", "\n", ".", "،", " "]
    )

    chunks = splitter.split_documents(all_documents)
    
    # إضافة Hash للتتبع
    for chunk in chunks:
        chunk.metadata["hash"] = hashlib.md5(chunk.page_content.encode()).hexdigest()

    print(f"✅ عدد الأجزاء النهائية: {len(chunks)}")

    # ===== 🧠 Embeddings =====
    print("🧠 تحميل نموذج التضمين (HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={'normalize_embeddings': True} # تطبيع التضمينات لدقة البحث
    )

    # ===== 💾 بناء DB =====
    if os.path.exists(DB_PATH):
        print("🧹 تنظيف قاعدة البيانات القديمة...")
        shutil.rmtree(DB_PATH)

    print("💾 جاري بناء قاعدة البيانات المتجهية... (استغرق وقتاً أطول بسبب تحسين الدقة)")
    
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    # ===== 🕒 حفظ الوقت =====
    with open("last_update.txt", "w", encoding="utf-8") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"🎉 تم بناء قاعدة البيانات بنجاح في {duration:.2f} ثانية!")

if __name__ == "__main__":
    build_database()
