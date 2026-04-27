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
import re

# ========= إعدادات =========
load_dotenv()

DATA_PATH = "data/pdfs"
DB_PATH = "university_db_app"
SITEMAP_URL = "https://www.iugaza.edu.ps/wp-sitemap.xml"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# ========= دالة لتنظيف النص من القوائم والتذييلات =========
def clean_text(text):
    # إزالة الأسطر الفارغة المتكررة والمسافات الزائدة
    text = re.sub(r'\s+', ' ', text).strip()
    # إزالة النصوص الشائعة غير المرغوبة (قوائم وتذييلات)
    noise_phrases = [
        "الرئيسية", "اتصل بنا", "خريطة الموقع", "الرؤية", "الرسالة", 
        "Copyright", "All Rights Reserved", "Facebook", "Twitter", "Instagram",
        "القائمة", "بحث", "تسجيل الدخول", "English"
    ]
    # إذا كان السطر يحتوي على جملة قصيرة من العبارات أعلاه، قد نحتاج لتصفيتها بذكاء
    # هنا سنكتفي بإزالة التكرار المفرط للمسافات والأسطر
    return text

# ========= جلب وتصفية الروابط =========
def get_website_urls_from_sitemap(sitemap_url):
    print("🗺️ جاري جلب خريطة الموقع...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
        response = requests.get(sitemap_url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        
        # 🛑 فلترة الروابط لاستبعاد الأخبار والأرشيف القديم
        exclude_keywords = ["/news/", "/events/", "/category/", "/tag/", "/author/", "/page/", "/2020/", "/2021/", "/2022/"]
        filtered_urls = []
        for url in urls:
            if url.startswith(UNIVERSITY_BASE_URL) and not any(k in url for k in exclude_keywords):
                filtered_urls.append(url)
        
        print(f"✅ تم العثور على {len(filtered_urls)} رابط صالح بعد الاستبعاد.")
        return filtered_urls
    except Exception as e:
        print(f"❌ خطأ في جلب خريطة الموقع: {e}")
        # قائمة احتياطية لأهم الصفحات فقط
        return [
            f"{UNIVERSITY_BASE_URL}/aboutiug/",
            f"{UNIVERSITY_BASE_URL}/facalties/",
            f"{UNIVERSITY_BASE_URL}/division/",
            f"{UNIVERSITY_BASE_URL}/e3lan/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/newstd/",
        ]

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 بدء بناء قاعدة المعرفة (الموقع والـ PDF)...")

    all_documents = []

    # ===== 🌐 تحميل الموقع =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    
    # تحديد عدد الروابط: نأخذ أول 50 رابط بعد الفلترة لضمان الجودة على الكمية
    urls = urls[:100] 

    if urls:
        print(f"📥 جاري تحميل المحتوى من {len(urls)} صفحة ويب...")
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        }

        try:
            # نستخدم bs_kwargs لمحاولة جلب المحتوى الرئيسي فقط (مثل <main> أو <article>)
            # لكن إذا فشل، سيعود لتحميل الصفحة كاملة، لذا سنقوم بالتنظيف لاحقاً
            web_loader = WebBaseLoader(
                urls,
                continue_on_failure=True, 
                requests_per_second=1, # تخفيف الحمل على السيرفر
                requests_kwargs={"headers": headers}, 
                bs_kwargs={"parse_only": SoupStrainer(["main", "article", "div", "body"])} 
            )
            web_documents = web_loader.load()
            
            # تنظيف النصوص المستخرجة
            for doc in web_documents:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["source"] = "website"
                
            all_documents.extend(web_documents)
            print(f"✅ تم تحميل وتنظيف {len(web_documents)} وثيقة من الموقع.")
        except Exception as e:
            print(f"❌ خطأ أثناء تحميل الموقع: {e}")

    # ===== 📄 تحميل PDF =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_path = os.path.join(current_dir, DATA_PATH)

    if os.path.exists(absolute_data_path):
        print("📥 جاري تحميل ملفات PDF...")
        try:
            pdf_loader = DirectoryLoader(absolute_data_path, loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()

            for doc in pdf_docs:
                doc.metadata["source"] = "pdf"
                # تنظيف الـ PDF أيضاً
                doc.page_content = clean_text(doc.page_content)

            all_documents.extend(pdf_docs)
            print(f"✅ تم تحميل {len(pdf_docs)} صفحة من ملفات PDF")
        except Exception as e:
            print(f"❌ خطأ أثناء قراءة ملفات PDF: {e}")

    if not all_documents:
        print("❌ لا توجد مستندات لمعالجتها!")
        return

    # ===== ✂️ تقسيم (زيادة الحجم لتحسين السياق) =====
    print("✂️ تقسيم النصوص...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # زيادة الحجم من 500 إلى 1000 للحصول على سياق أفضل
        chunk_overlap=200
    )
    chunks = splitter.split_documents(all_documents)
    print(f"✅ عدد الأجزاء النهائية: {len(chunks)}")

    # ===== 🧠 Embeddings =====
    print(f"🧠 تحميل نموذج التضمين: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # ===== 💾 بناء DB =====
    print("💾 بناء وحفظ قاعدة البيانات...")
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        with open("last_update.txt", "w", encoding="utf-8") as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
        print("🎉 تم تحديث قاعدة البيانات بنجاح!")
    except Exception as e:
        print(f"❌ خطأ أثناء حفظ قاعدة البيانات: {e}")

if __name__ == "__main__":
    build_database()
