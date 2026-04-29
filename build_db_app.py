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
SKIP_URL_KEYWORDS = ["tag", "author", "feed", "comment", "embed", "/page/"]
SKIP_URL_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".zip", ".rar", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".mp4", ".mp3"
)

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
    filtered = [
        u for u in urls
        if u.startswith(UNIVERSITY_BASE_URL)
        and not any(s in u for s in SKIP_URL_KEYWORDS)
        and not u.lower().endswith(SKIP_URL_SUFFIXES)
    ]
    print(f"🔍 بعد الفلترة: {len(filtered)} رابط")
    return filtered

def get_website_urls_from_sitemap(sitemap_url):
    print("🗺️ جاري جلب خريطة الموقع...")
    try:
        urls = get_all_urls_from_sitemap(sitemap_url)
        valid_urls = filter_urls(urls)
        print(f"✅ تم العثور على {len(valid_urls)} رابط صالح في خريطة الموقع.")
        return valid_urls
    except Exception as e:
        print(f"❌ خطأ في جلب خريطة الموقع: {e}")
        print("🔄 الرجوع إلى قائمة روابط يدوية...")
        return [
            f"{UNIVERSITY_BASE_URL}/",
            f"{UNIVERSITY_BASE_URL}/aboutiug/",
            f"{UNIVERSITY_BASE_URL}/facalties/",
            f"{UNIVERSITY_BASE_URL}/division/",
            f"{UNIVERSITY_BASE_URL}/e3lan/",
            f"{UNIVERSITY_BASE_URL}/eservices/",
            f"{UNIVERSITY_BASE_URL}/newstd/",
            f"{UNIVERSITY_BASE_URL}/أخبار-الجامعة/"
        ]


def normalize_text(text):
    cleaned = " ".join((text or "").split())
    return cleaned.strip()


def enrich_document_metadata(doc, source_type):
    doc.metadata["source_type"] = source_type
    doc.metadata["source_label"] = os.path.basename(doc.metadata.get("source", "")) or source_type
    doc.page_content = normalize_text(doc.page_content)
    return doc


# ========= hash =========
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# ========= بناء قاعدة البيانات =========
def build_database():
    print("🚀 بدء بناء قاعدة المعرفة...")

    all_documents = []

    # ===== 🌐 تحميل الموقع =====
    urls = get_website_urls_from_sitemap(SITEMAP_URL)
    if urls:
        print("📥 جاري تحميل المحتوى من صفحات الويب...")
        web_loader = WebBaseLoader(
            urls,
            continue_on_failure=True,
            requests_per_second=1,
            bs_kwargs={"parse_only": SoupStrainer(["main", "article", "body"])}
        )
        web_documents = web_loader.load()
        web_documents = [
            enrich_document_metadata(doc, "website")
            for doc in web_documents
            if normalize_text(doc.page_content)
        ]
        print(f"✅ تم تحميل {len(web_documents)} وثيقة من الموقع الإلكتروني.")
        all_documents.extend(web_documents)

    # ===== 📄 تحميل PDF =====
    if os.path.exists(DATA_PATH):
        print("📥 تحميل ملفات PDF...")
        pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        pdf_docs = [
            enrich_document_metadata(doc, "pdf")
            for doc in pdf_docs
            if normalize_text(doc.page_content)
        ]

        print(f"✅ تم تحميل {len(pdf_docs)} ملف PDF")
        all_documents.extend(pdf_docs)

    if not all_documents:
        print("❌ لا يوجد بيانات!")
        return

    # ===== ✂️ تقسيم =====
    print("✂️ تقسيم النصوص...")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "###", "##", ".", "؟", "!", "؛", "،", " ", ""],
        chunk_size=900,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(all_documents)
    unique_chunks = []
    seen_hashes = set()
    for chunk in chunks:
        chunk.page_content = normalize_text(chunk.page_content)
        if len(chunk.page_content) < 80:
            continue
        chunk_hash = get_hash(chunk.page_content)
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)
        chunk.metadata["hash"] = chunk_hash
        unique_chunks.append(chunk)
    chunks = unique_chunks
    print(f"✅ عدد الأجزاء: {len(chunks)}")

    # ===== 🧠 Embeddings =====
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

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

    print("🎉 تم بناء قاعدة البيانات بنجاح!")

# ========= تشغيل =========
if __name__ == "__main__":
    build_database()



