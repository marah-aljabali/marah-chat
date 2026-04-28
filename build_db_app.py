import os
import requests
import re
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

# ========= دالة تنظيف الويب (مرنة أكثر) =========
def clean_html_text(text):
    """
    تنظيف خفيف جداً لضمان عدم حذف المعلومات المهمة.
    """
    if not text:
        return ""
    
    # إزالة الأسطر الفارغة المتكررة فقط للحفاظ على المعلومات
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # سنعتمد على Splitter لاحقاً لتنظيف الباقي، هنا فقط ننظّم النص
    return text.strip()

# ========= جلب الروابط (مع قائمة احتياطية أغنى) =========
def get_website_urls_from_sitemap(sitemap_url):
    print("🗺️ Fetching Sitemap...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(sitemap_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        
        valid_urls = [url for url in urls if url.startswith(UNIVERSITY_BASE_URL)]
        
        # فلترة الروابط لتجنب الميديا
        valid_urls = [url for url in valid_urls if not any(x in url.lower() for x in ['.jpg', '.png', '.pdf', 'video', 'attachment'])]
        
        print(f"✅ Found {len(valid_urls)} valid URLs from Sitemap.")
        return valid_urls
    except Exception as e:
        print(f"❌ Sitemap Error: {e}")
        # قائمة احتياطية أغنى بروابط ذات محتوى نصي مرجح
        return [
            f"{UNIVERSITY_BASE_URL}/",
            f"{UNIVERSITY_BASE_URL}/aboutiug/",
            f"{UNIVERSITY_BASE_URL}/facalties/",
            f"{UNIVERSITY_BASE_URL}/division/",
            f"{UNIVERSITY_BASE_URL}/e3lan/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/newstd/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/أخبار-الجامعة",
        ]

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 Starting Database Construction (Web + PDF)...")

    all_documents = []

    # ===== 🌐 تحميل الموقع =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    
    # سنأخذ الروابط كما هي لنجرب جميعها
    print(f"📥 Loading content from {len(urls)} web pages...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    }

    if urls:
        try:
            web_loader = WebBaseLoader(
                urls,
                continue_on_failure=True, 
                requests_per_second=1,
                requests_kwargs={"headers": headers},
                bs_kwargs={"parse_only": SoupStrainer("body")}
            )
            web_documents = web_loader.load()
            
            print(f"⚠️ Raw documents fetched: {len(web_documents)}")
            
            # ✨ عملية التنظيف والتشخيص
            cleaned_web_docs = []
            for i, doc in enumerate(web_documents):
                raw_text = doc.page_content
                cleaned_text = clean_html_text(raw_text)
                
                raw_len = len(raw_text)
                clean_len = len(cleaned_text)
                url = doc.metadata.get('source', 'unknown')

                # 🔥 طباعة التشخيص (هام جداً لمعرفة ما يحدث)
                print(f"🔍 [DEBUG {i+1}] URL: {url} | Raw: {raw_len} chars -> Cleaned: {clean_len} chars")
                
                # خفضنا الحد الأدنى من 100 إلى 50 للتأكد من عدم ضياع الصفحات القصيرة
                if clean_len > 50: 
                    doc.page_content = cleaned_text
                    doc.metadata["source"] = "website"
                    cleaned_web_docs.append(doc)
                else:
                    # طباعة عينة من النص إذا تم حذفه
                    print(f"⚠️ Dropped (Short). Content: {cleaned_text[:100]}")
            
            print(f"✅ Final Cleaned Web Documents: {len(cleaned_web_docs)}")
            all_documents.extend(cleaned_web_docs)
        except Exception as e:
            print(f"❌ Web Loading Error: {e}")

    # ===== 📄 تحميل PDF =====
    current_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_data_path = os.path.join(current_dir, DATA_PATH)

    if os.path.exists(absolute_data_path):
        print("📥 Loading PDF files...")
        try:
            pdf_loader = DirectoryLoader(absolute_data_path, loader_cls=PyPDFLoader, silent_errors=True)
            pdf_docs = pdf_loader.load()
            print(f"✅ Loaded {len(pdf_docs)} pages from PDF files.")
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"❌ PDF Loading Error: {e}")
    else:
        print("⚠️ Warning: data/pdfs directory not found.")

    if not all_documents:
        print("❌ No documents to process!")
        return

    # ===== ✂️ تقسيم النصوص =====
    print("✂️ Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ". ", "，", ""]
    )
    chunks = splitter.split_documents(all_documents)
    print(f"✅ Final chunks count: {len(chunks)}")

    # ===== 🧠 Embeddings & Save =====
    print(f"🧠 Loading Embedding Model: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("💾 Building Vector Database...")
    
    if os.path.exists(DB_PATH):
        print(f"🧹 Cleaning old database at {DB_PATH}...")
        shutil.rmtree(DB_PATH)

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        
        update_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("last_update.txt", "w", encoding="utf-8") as f:
            f.write(update_time)
            
        print(f"🎉 Database Updated Successfully! ({update_time})")
    except Exception as e:
        print(f"❌ Error saving database: {e}")

def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

if __name__ == "__main__":
    build_database()
