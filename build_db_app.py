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
import random

# ========= إعدادات =========
load_dotenv()

DATA_PATH = "data/pdfs"
DB_PATH = "university_db_app"
SITEMAP_URL = "https://www.iugaza.edu.ps/wp-sitemap.xml"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ========= جلب كل روابط السايت ماب =========
def get_all_urls_from_sitemap(sitemap_url):
    print("🗺️ قراءة sitemap...")

    all_urls = set()

    def parse_sitemap(url):
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "xml")

            # لو فيه sitemaps فرعية
            sitemap_tags = soup.find_all("sitemap")
            if sitemap_tags:
                for sm in sitemap_tags:
                    loc = sm.find("loc").text
                    parse_sitemap(loc)

            # لو فيه روابط صفحات
            url_tags = soup.find_all("url")
            for u in url_tags:
                loc = u.find("loc").text
                all_urls.add(loc)

        except Exception as e:
            print(f"❌ خطأ في {url}: {e}")

    parse_sitemap(sitemap_url)

    print(f"✅ تم جمع {len(all_urls)} رابط")
    return list(all_urls)

# ========= فلترة الروابط =========
def filter_urls(urls):
    skip_keywords = ["tag", "author", "feed", "comment"]
    filtered = [u for u in urls if not any(s in u for s in skip_keywords)]
    print(f"🔍 بعد الفلترة: {len(filtered)} رابط")
    return filtered

def get_website_urls_from_sitemap(sitemap_url):
    """
    دالة لجلب كل الروابط من ملف sitemap.xml الخاص بالموقع.
    """
    print("🗺️ جاري جلب خريطة الموقع...")
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status() # التأكد من أن الطلب ناجح
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        
        # فلترة الروابط للتأكد من أنها تابعة للموقع الرئيسي
        valid_urls = [url for url in urls if url.startswith(UNIVERSITY_BASE_URL)]
        
        print(f"✅ تم العثور على {len(valid_urls)} رابط صالح في خريطة الموقع.")
        return valid_urls
    except Exception as e:
        print(f"❌ خطأ في جلب خريطة الموقع: {e}")
        print("🔄 الرجوع إلى قائمة روابط يدوية...")
        # قائمة احتياطية في حال فشل قراءة الـ Sitemap
        return [
            f"{UNIVERSITY_BASE_URL}/",
            f"{UNIVERSITY_BASE_URL}/aboutiug/",
            f"{UNIVERSITY_BASE_URL}/facalties/",
            f"{UNIVERSITY_BASE_URL}/division/",
            f"{UNIVERSITY_BASE_URL}/e3lan/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/newstd/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/أخبار-الجامعة/"
        ]


# ========= hash =========
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 بدء بناء قاعدة المعرفة...")

    all_documents = []

    # ===== 🌐 تحميل الموقع =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    #  urls = filter_urls(urls)

    # 🔥 تقليل الحجم (اختياري)
    #if len(urls) > 300:
     #   urls = random.sample(urls, 300)
      #3  print("⚡ تم اختيار 300 رابط عشوائي لتسريع البناء")

    #print("📥 تحميل صفحات الموقع...")

    #web_loader = WebBaseLoader(
     #   urls,
      #  requests_per_second=2,
       # continue_on_failure=True,
        #bs_kwargs={
         #   "parse_only": SoupStrainer(["p", "h1", "h2", "h3", "li"])
        #}
    #)

    #web_docs = web_loader.load()

    #for doc in web_docs:
     #   doc.metadata["source"] = "website" 
    if urls:
      print("📥 جاري تحميل المحتوى من صفحات الويب...")
      # WebBaseLoader يمكنه التعامل مع قائمة من الروابط
      # لاحظ أننا نستخدم bs4.Strategy لاستخراج النصوص فقط
      web_loader = WebBaseLoader(
          urls,
          continue_on_failure=True, # استمرار التحميل حتى لو فشلت بعض الصفحات
          requests_per_second=1,   # إبطاء الطلبات لتجنب حظر السيرفر
          bs_kwargs={"parse_only": SoupStrainer("body")} # تحليل جزء الـ body فقط لتسريع العملية
      )
      web_documents = web_loader.load()
      print(f"✅ تم تحميل {len(web_documents)} وثيقة من الموقع الإلكتروني.")
      all_documents.extend(web_documents)

    #print(f"✅ تم تحميل {len(web_docs)} صفحة من الموقع")
    #all_documents.extend(web_docs)

    # ===== 📄 تحميل PDF =====
    if os.path.exists(DATA_PATH):
        print("📥 تحميل ملفات PDF...")
        pdf_loader = DirectoryLoader(DATA_PATH, loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()

        for doc in pdf_docs:
            doc.metadata["source"] = "pdf"

        print(f"✅ تم تحميل {len(pdf_docs)} ملف PDF")
        all_documents.extend(pdf_docs)

    if not all_documents:
        print("❌ لا يوجد بيانات!")
        return

    # ===== ✂️ تقسيم =====
    print("✂️ تقسيم النصوص...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(all_documents)
    print(f"✅ عدد الأجزاء: {len(chunks)}")

    # ===== 🧠 Embeddings =====
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # ===== 💾 بناء DB =====
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    print("💾 بناء قاعدة البيانات...")

    for chunk in chunks:
        chunk.metadata["hash"] = get_hash(chunk.page_content)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    # ===== 🕒 حفظ آخر تحديث =====
    with open("last_update.txt", "w", encoding="utf-8") as f:
        f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    print("🎉 تم بناء قاعدة البيانات بنجاح!")

# ========= تشغيل =========
if __name__ == "__main__":
    build_database()



