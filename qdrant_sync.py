"""
qdrant_sync.py — 將 Notion（PAPER_DB / NEWS_DB）中尚未向量化的資料同步至 Qdrant
"""
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("qdrant_sync")

# ── 沿用 notion_sync.py 的 ID 格式化邏輯，確保兩邊讀到同一個 DB ──
def _format_id(notion_id: str) -> str:
    if not notion_id:
        return ""
    s = notion_id.split('#')[0].strip().replace("-", "")
    if len(s) != 32:
        return s
    return f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"

# ── 環境變數：沿用 notion_sync.py 已經設定好的同一組 DB ID ──
NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "").strip()
PAPER_DB_ID  = _format_id(os.environ.get("PAPER_DB_ID", ""))
NEWS_DB_ID   = _format_id(os.environ.get("NEWS_DB_ID", ""))
QDRANT_URL       = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY   = os.environ.get("QDRANT_API_KEY", "").strip()

COLLECTION_NAME  = "research-agent-rag"
VECTOR_SIZE      = 384

qdrant   = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

def _check_env() -> None:
    missing = [n for n, v in [
        ("NOTION_TOKEN", NOTION_TOKEN), ("PAPER_DB_ID", PAPER_DB_ID),
        ("NEWS_DB_ID", NEWS_DB_ID), ("QDRANT_URL", QDRANT_URL),
        ("QDRANT_API_KEY", QDRANT_API_KEY),
    ] if not v]
    if missing:
        raise ValueError(f"缺少環境變數: {missing}")

def _query_db(db_id: str, tag: str) -> List[Dict]:
    """與 notion_sync.py 的 _query_db 邏輯一致，直接用 httpx 避免 SDK 相容性問題"""
    results, cursor = [], None
    url = f"https://api.notion.com/v1/databases/{db_id}/query"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }
    while True:
        payload: Dict[str, Any] = {"page_size": 100}
        if cursor:
            payload["start_cursor"] = cursor
        try:
            r = httpx.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code != 200:
                logger.error("❌ 查詢失敗 [%s]: %s %s", tag, r.status_code, r.text[:200])
                break
            data = r.json()
            results.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            cursor = data.get("next_cursor")
        except Exception as exc:
            logger.error("❌ 查詢異常 [%s]: %s", tag, exc)
            break
    logger.info("📋 查詢 [%s] 完成，共 %d 筆", tag, len(results))
    return results

def _extract_text_props(page: Dict, title_key: str, summary_key: str, url_key: str = "原文連結"):
    """對應 notion_sync.py 實際寫入的中文欄位名稱"""
    props = page.get("properties", {})
    title = "".join(x.get("plain_text", "") for x in props.get(title_key, {}).get("title", []))
    summary_parts = props.get(summary_key, {}).get("rich_text", [])
    summary = "".join(x.get("plain_text", "") for x in summary_parts)
    url = props.get(url_key, {}).get("url", "")
    tags = [t.get("name", "") for t in props.get("標籤", {}).get("multi_select", [])]
    return title, summary, url, tags

def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info("✅ 已建立 Collection: %s", COLLECTION_NAME)

def get_synced_urls() -> set:
    """掃描 Qdrant 中已存在的 notion_url，作為去重依據，取代不可靠的時間篩選"""
    synced = set()
    offset = None
    while True:
        records, offset = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=200,
            offset=offset,
            with_payload=["notion_url"],
            with_vectors=False,
        )
        for r in records:
            url = r.payload.get("notion_url")
            if url:
                synced.add(url)
        if offset is None:
            break
    logger.info("🔍 Qdrant 中已有 %d 筆向量資料", len(synced))
    return synced

def sync_source(db_id: str, source_label: str, title_key: str, summary_key: str,
                 synced_urls: set) -> List[PointStruct]:
    """處理單一資料庫（PAPER_DB 或 NEWS_DB），只向量化尚未同步過的項目"""
    pages = _query_db(db_id, f"qdrant_{source_label}")
    points = []
    for page in pages:
        title, summary, url, tags = _extract_text_props(page, title_key, summary_key)
        if not url or url in synced_urls:
            continue  # 已同步過，跳過
        full_text = f"{title}\n{summary}"
        if not full_text.strip():
            continue
        chunks = splitter.split_text(full_text)
        for idx, chunk in enumerate(chunks):
            vector = embedder.encode(chunk).tolist()
            # 修正：用 uuid5 將字串轉為符合 Qdrant 規範的 UUID
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}_{idx}"))
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text": chunk,
                    "title": title,
                    "notion_url": url,
                    "source_type": source_label,
                    "tags": tags,
                    "synced_at": datetime.now(timezone.utc).isoformat(),
                }
            ))
        time.sleep(0.4)  # Notion API 速率限制保護（此處主要保護寫入端，讀取已分頁處理）
    return points

def run_qdrant_sync():
    _check_env()
    ensure_collection()
    synced_urls = get_synced_urls()
    
    all_points = []
    all_points += sync_source(PAPER_DB_ID, "Paper", "標題", "中文摘要", synced_urls)
    all_points += sync_source(NEWS_DB_ID, "News", "標題", "中文摘要", synced_urls)
    
    if all_points:
        # 批次寫入，每批 100 筆避免單次 payload 過大
        for i in range(0, len(all_points), 100):
            batch = all_points[i:i + 100]
            qdrant.upsert(collection_name=COLLECTION_NAME, points=batch)
        logger.info("✅ 已寫入 %d 筆向量到 Qdrant", len(all_points))
    else:
        logger.info(" 無新資料需要同步")
    logger.info("🏁 Qdrant 同步完成！")

if __name__ == "__main__":
    run_qdrant_sync()
