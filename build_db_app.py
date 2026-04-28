
import os
import requests
from bs4 import BeautifulSoup, SoupStrainer
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import datetime
import time

# --- إعدادات ---
load_dotenv()
# مسارات الملفات
DATA_PATH = "data/pdfs"
CHROMA_DB_PATH = "university_db_app"
# إعدادات النماذج
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# إعدادات الموقع
UNIVERSITY_SITEMAP_URL = "https://www.iugaza.edu.ps/wp-sitemap.xml"
UNIVERSITY_BASE_URL = "https://www.iugaza.edu.ps"

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
            f"{UNIVERSITY_BASE_URL}/faculties/",
            f"{UNIVERSITY_BASE_URL}/division/",
            f"{UNIVERSITY_BASE_URL}/centers/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/newstd/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/eservices/"
        ]

def build_database():
    """
    دالة رئيسية لقراءة ملفات PDF والموقع الإلكتروني وبناء قاعدة البيانات المتجهية.
    """
    print("🚀 بدء بناء قاعدة المعرفة الشاملة...")

    all_documents = []

    # --- الجزء الأول: تحميل المحتوى من الموقع الإلكتروني ---
    print("\n--- بدء معالجة المحتوى من الموقع الإلكتروني ---")
    website_urls = get_website_urls_from_sitemap(UNIVERSITY_SITEMAP_URL)
    
    if website_urls:
        print("📥 جاري تحميل المحتوى من صفحات الويب...")
        # WebBaseLoader يمكنه التعامل مع قائمة من الروابط
        # لاحظ أننا نستخدم bs4.Strategy لاستخراج النصوص فقط
        web_loader = WebBaseLoader(
            website_urls,
            continue_on_failure=True, # استمرار التحميل حتى لو فشلت بعض الصفحات
            requests_per_second=1,   # إبطاء الطلبات لتجنب حظر السيرفر
            bs_kwargs={"parse_only": SoupStrainer("body")} # تحليل جزء الـ body فقط لتسريع العملية
        )
        web_documents = web_loader.load()
        print(f"✅ تم تحميل {len(web_documents)} وثيقة من الموقع الإلكتروني.")
        all_documents.extend(web_documents)

    # --- الجزء الثاني: تحميل ملفات PDF ---
    print("\n--- بدء معالجة ملفات PDF ---")
    if os.path.exists(DATA_PATH):
        print("📥 جاري تحميل ملفات PDF...")
        pdf_loader = DirectoryLoader(
            DATA_PATH,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            recursive=True,
            show_progress=True
        )
        pdf_documents = pdf_loader.load()
        print(f"✅ تم تحميل {len(pdf_documents)} وثيقة من ملفات PDF.")
        all_documents.extend(pdf_documents)
    else:
        print(f"⚠️ مجلد البيانات '{DATA_PATH}' غير موجود. سيتم تجاهل ملفات PDF.")

    if not all_documents:
        print("❌ لم يتم العثور على أي وثائق من الموقع أو ملفات PDF. لا يمكن بناء قاعدة البيانات.")
        return

    # --- الجزء الثالث: تقسيم ومعالجة كل الوثائق ---
    print(f"\n✂️ جاري تقسيم إجمالي {len(all_documents)} وثيقة...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_documents)
    print(f"✅ تم تقسيم الوثائق إلى {len(chunks)} جزء.")

    # --- الجزء الرابع: بناء قاعدة البيانات المتجهية ---
    print("🧠 جاري تحميل نموذج التضمين...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    print("💾 جاري بناء قاعدة البيانات المتجهية الموحدة...")
    if os.path.exists(CHROMA_DB_PATH):
        import shutil
        shutil.rmtree(CHROMA_DB_PATH)
        
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
    print(f"✅ تم بناء قاعدة المعرفة الشاملة بنجاح في مجلد '{CHROMA_DB_PATH}'.")

    with open("last_update.txt", "w", encoding="utf-8") as f:
        f.write(f"آخر تحديث لقاعدة البيانات: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("✅ تم تحديث ملف آخر تحديث.")

if __name__ == "__main__":
    build_database()
