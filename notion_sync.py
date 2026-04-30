"""
notion_sync.py — 將 final_report.json 同步至 Notion 的五大模組
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import json
import logging
import os
import time
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ── .env 自動載入 ─────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Third-Party ───────────────────────────────────────────────────────────────
from notion_client import Client
from notion_client.errors import APIResponseError

# ─────────────────────────────────────────────────────────────────────────────
# Logging 設定
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("notion_sync")

# ─────────────────────────────────────────────────────────────────────────────
# 工具：ID 格式化
# ─────────────────────────────────────────────────────────────────────────────
def _format_id(notion_id: str) -> str:
    """將 32 碼 ID 轉換為帶連字號的標準 UUID 格式 (8-4-4-4-12)。"""
    if not notion_id:
        return ""
    # 先清理空格與註解
    s = notion_id.split('#')[0].strip().replace("-", "")
    if len(s) != 32:
        return s
    return f"{s[:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:]}"

# ─────────────────────────────────────────────────────────────────────────────
# 環境變數
# ─────────────────────────────────────────────────────────────────────────────
NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "").strip()
DASHBOARD_PAGE_ID  = _format_id(os.environ.get("DASHBOARD_PAGE_ID", ""))
PAPER_DB_ID        = _format_id(os.environ.get("PAPER_DB_ID", ""))
NEWS_DB_ID         = _format_id(os.environ.get("NEWS_DB_ID", ""))
STATS_DB_ID        = _format_id(os.environ.get("STATS_DB_ID", ""))
TREND_DB_ID        = _format_id(os.environ.get("TREND_DB_ID", ""))

INPUT_JSON = "final_report.json"
RICH_TEXT_LIMIT = 1900

# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def _check_env() -> None:
    """驗證所有必要的環境變數。"""
    import notion_client
    logger.info("📦 notion-client 庫版本: %s", getattr(notion_client, "__version__", "未知"))

    missing = []
    for name, val in [
        ("NOTION_TOKEN", NOTION_TOKEN),
        ("DASHBOARD_PAGE_ID", DASHBOARD_PAGE_ID),
        ("PAPER_DB_ID", PAPER_DB_ID),
        ("NEWS_DB_ID", NEWS_DB_ID),
        ("STATS_DB_ID", STATS_DB_ID),
        ("TREND_DB_ID", TREND_DB_ID),
    ]:
        if not val or val.startswith("x" * 10):
            missing.append(name)
    if missing:
        raise ValueError(f"缺少環境變數: {missing}")

def _rt(text: Optional[str], limit: int = RICH_TEXT_LIMIT) -> List[Dict]:
    if not text: return [{"type": "text", "text": {"content": ""}}]
    text = str(text)
    chunks = [text[i:i + limit] for i in range(0, len(text), limit)]
    return [{"type": "text", "text": {"content": chunk}} for chunk in chunks[:10]]

def _multi_select(tags: List[str]) -> List[Dict]:
    return [{"name": str(t)[:99]} for t in tags if t]

def _date_str(raw: Optional[str]) -> Optional[str]:
    """將各種日期格式轉為 Notion 要求的 YYYY-MM-DD。"""
    if not raw:
        return None
    raw = str(raw).strip()
    # 已經是 ISO 格式 (2026-04-29...)
    if len(raw) >= 10 and raw[4] == '-':
        return raw[:10]
    # RFC 2822 格式 (Wed, 29 Apr 2026 ...)
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(raw)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    # Unix timestamp
    try:
        ts = float(raw)
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        pass
    return None

def _api_call(fn, tag: str = "", retries: int = 3) -> Optional[Any]:
    for attempt in range(retries):
        try:
            return fn()
        except APIResponseError as exc:
            if exc.status == 403:
                logger.error("❌ Notion 403 禁止存取 [%s]: 請檢查 Connection。", tag)
                return None
            logger.error("❌ Notion API 錯誤 [%s]: %s", tag, exc.body)
            time.sleep(2 ** attempt)
        except Exception as exc:
            logger.error("❌ 未預期錯誤 [%s]: %s", tag, exc)
            time.sleep(2 ** attempt)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# 去重與查詢
# ─────────────────────────────────────────────────────────────────────────────

def _query_db(notion: Client, db_id: str, tag: str) -> List[Dict]:
    """通用資料庫查詢 — 使用 httpx 直接發送 POST 請求，繞過庫的相容性問題。"""
    import httpx
    results = []
    cursor = None
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
    logger.info("📋 查詢 [%s] 完成，共 %d 筆現有資料", tag, len(results))
    return results

def _get_existing_titles(notion: Client, db_id: str) -> Dict[str, str]:
    existing = {}
    for page in _query_db(notion, db_id, "query_titles"):
        props = page.get("properties", {})
        for p in props.values():
            if p.get("type") == "title":
                t = "".join(x.get("plain_text", "") for x in p.get("title", []))
                if t: existing[t.strip()] = page["id"]
                break
    return existing

def _get_existing_urls(notion: Client, db_id: str, url_prop: str = "原文連結") -> Dict[str, str]:
    existing = {}
    for page in _query_db(notion, db_id, "query_urls"):
        u = page.get("properties", {}).get(url_prop, {}).get("url")
        if u: existing[u.strip()] = page["id"]
    return existing

# ─────────────────────────────────────────────────────────────────────────────
# 同步模組
# ─────────────────────────────────────────────────────────────────────────────

def sync_dashboard(notion: Client, summary: str, trend_data: Dict) -> None:
    logger.info("📌 同步 Dashboard 摘要...")
    today = trend_data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    text = f"📅 {today} | {summary}"
    resp = _api_call(lambda: notion.blocks.children.list(block_id=DASHBOARD_PAGE_ID), "dash_list")
    if not resp: return
    callout_id = next((b["id"] for b in resp.get("results", []) if b["type"] == "callout"), None)
    if callout_id:
        _api_call(lambda: notion.blocks.update(callout_id, callout={"rich_text": _rt(text)}), "dash_upd")
    else:
        _api_call(lambda: notion.blocks.children.append(DASHBOARD_PAGE_ID, children=[{
            "object":"block","type":"callout","callout":{"rich_text":_rt(text),"icon":{"type":"emoji","emoji":"🤖"}}
        }]), "dash_add")

def sync_papers(notion: Client, items: List[Dict]) -> None:
    papers = [i for i in items if i.get("source_type") == "Paper"]
    if not papers: return
    logger.info("📚 同步論文資料庫...")
    existing = _get_existing_urls(notion, PAPER_DB_ID)
    for p in papers:
        if p.get("url") in existing: continue
        props = {
            "標題": {"title": _rt(p.get("title"))},
            "相關度評分": {"number": p.get("relevance_score")},
            "中文摘要": {"rich_text": _rt(p.get("chinese_summary") or p.get("score_reason"))},
            "標籤": {"multi_select": _multi_select(p.get("tags", []) + p.get("categories", []))},
            "原文連結": {"url": p.get("url")},
            "分析類型": {"select": {"name": p.get("analysis_type", "scored_only")}},
        }
        if p.get("published_at"): props["發表日期"] = {"date": {"start": _date_str(p["published_at"])}}
        _api_call(lambda: notion.pages.create(parent={"database_id": PAPER_DB_ID}, properties=props), "paper_add")

def sync_news(notion: Client, items: List[Dict]) -> None:
    news = [i for i in items if i.get("source_type") == "News"]
    if not news: return
    logger.info("📰 同步科技新聞資料庫...")
    existing = _get_existing_urls(notion, NEWS_DB_ID)
    for n in news:
        if n.get("url") in existing: continue
        props = {
            "標題": {"title": _rt(n.get("title"))},
            "來源": {"select": {"name": n.get("source_name", "未知")[:99]}},
            "中文摘要": {"rich_text": _rt(n.get("chinese_summary"))},
            "標籤": {"multi_select": _multi_select(n.get("tags", []))},
            "原文連結": {"url": n.get("url")},
        }
        if n.get("published_at"): props["發布時間"] = {"date": {"start": _date_str(n["published_at"])}}
        _api_call(lambda: notion.pages.create(parent={"database_id": NEWS_DB_ID}, properties=props), "news_add")

def sync_keyword_stats(notion: Client, stats: Dict[str, int]) -> None:
    if not stats: return
    logger.info("🔑 同步關鍵字熱度...")
    existing = _get_existing_titles(notion, STATS_DB_ID)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for kw, count in stats.items():
        props = {"出現次數": {"number": count}, "最後更新日期": {"date": {"start": today}}}
        if kw in existing:
            _api_call(lambda: notion.pages.update(existing[kw], properties=props), "stat_upd")
        else:
            props["關鍵字"] = {"title": _rt(kw)}
            _api_call(lambda: notion.pages.create(parent={"database_id": STATS_DB_ID}, properties=props), "stat_add")

def sync_trend_data(notion: Client, trend: Dict) -> None:
    if not trend: return
    date_str = trend.get("date", "")
    logger.info("📈 同步趨勢記錄 (%s)...", date_str)
    existing = _get_existing_titles(notion, TREND_DB_ID)
    props = {
        "論文總數": {"number": trend.get("total_papers", 0)},
        "新聞總數": {"number": trend.get("total_news", 0)},
        "深度分析篇數": {"number": trend.get("deep_analyzed_papers", 0)},
    }
    if date_str in existing:
        _api_call(lambda: notion.pages.update(existing[date_str], properties=props), "trend_upd")
    else:
        props["日期"] = {"title": _rt(date_str)}
        _api_call(lambda: notion.pages.create(parent={"database_id": TREND_DB_ID}, properties=props), "trend_add")

def run_notion_sync():
    _check_env()
    notion = Client(auth=NOTION_TOKEN)
    try:
        with open(INPUT_JSON, encoding="utf-8") as f: report = json.load(f)
    except Exception: return
    sync_dashboard(notion, report.get("dashboard_summary", ""), report.get("trend_data", {}))
    sync_papers(notion, report.get("analyzed_items", []))
    sync_news(notion, report.get("analyzed_items", []))
    sync_keyword_stats(notion, report.get("keyword_stats", {}))
    sync_trend_data(notion, report.get("trend_data", {}))
    logger.info("🏁 同步完成！")

if __name__ == "__main__":
    run_notion_sync()
