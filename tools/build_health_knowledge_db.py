import os
import json
import base64
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ====== 設定 ======
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

WEB_CSV = "/home/shambhala/Agent/Health_Report_Agent/web.csv"

def load_web_urls(csv_path: str):
    urls = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for line in f:
            url = line.strip()
            if url:
                urls.append(url)
    # 重複を除去しつつ順序を保持
    seen = set()
    return [u for u in urls if not (u in seen or seen.add(u))]

WEB_URLS = load_web_urls(WEB_CSV)

PNG_DIR = "RAG_PNG"
DB_DIR = "chroma_db_健康資料"
COLLECTION_NAME = "web_png_rag"

# ====== モデル初期化 ======
vision_model = init_chat_model("openai:gpt-4o", api_key=OPENAI_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# ====== ユーティリティ関数 ======
def fetch_web_text(url: str) -> str:
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # 1) 不要な構造を削除
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    # 2) 本文領域のみを抽出（main / article を優先）
    main = soup.find("main") or soup.find("article")
    if main:
        text = main.get_text("\n")
    else:
        text = soup.get_text("\n")

    # 3) 空行と短いノイズ行を除去
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= 1:
            continue
        lines.append(line)

    return "\n".join(lines)

def encode_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ocr_png_with_gpt4o(path: str) -> str:
    b64 = encode_image_b64(path)
    message = HumanMessage(content=[
        {
            "type": "text",
            "text": "この画像から文字をOCRとして抽出してください。出力はテキストのみ。"
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}
        }
    ])
    response = vision_model.invoke([message])
    return response.content.strip()

def build_documents():
    docs = []

    # 1) Web
    for url in WEB_URLS:
        try:
            text = fetch_web_text(url)
            docs.append(Document(
                page_content=text,
                metadata={"source": url, "type": "web"}
            ))
        except Exception as e:
            print(f"[Web error] {url}: {e}")

    # 2) PNG OCR
    png_paths = sorted(Path(PNG_DIR).glob("*.png"))

    for path in png_paths:
        try:
            text = ocr_png_with_gpt4o(str(path))
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={"source": str(path), "type": "png_ocr"}
                ))
        except Exception as e:
            print(f"[PNG error] {path}: {e}")
        print(text)

    return docs

def build_chroma(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"✅ 書き込み完了: {DB_DIR}")
    return db

def test_query(db, query: str, k: int = 3):
    results = db.similarity_search(query, k=k)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Hit {i} ---")
        print("source:", doc.metadata.get("source"))
        print("type:", doc.metadata.get("type"))
        print(doc.page_content)

if __name__ == "__main__":
    os.makedirs(DB_DIR, exist_ok=True)
    docs = build_documents()
    print(f"ドキュメント数: {len(docs)}")
    db = build_chroma(docs)

    # 検索テスト
    test_query(db, "BMIと基準はいくらですか、もし高いなら、どのような健康対策を取るべきですか？")