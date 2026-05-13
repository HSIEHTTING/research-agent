"""
fetcher_engine.py — 論文自動化系統的數據抓取中樞

Dependencies (pip install):
    pip install arxiv feedparser requests tenacity

Usage:
    from fetcher_engine import run_all_fetchers
    results = run_all_fetchers()
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import logging
import time
import json
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Optional

# ── Third-Party ───────────────────────────────────────────────────────────────
# pip install arxiv feedparser requests tenacity
import arxiv
import feedparser
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging 設定
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("fetcher_engine")

# ─────────────────────────────────────────────────────────────────────────────
# 常數設定
# ─────────────────────────────────────────────────────────────────────────────
ARXIV_CATEGORIES: List[str] = ["cat:cs.LG", "cat:cs.AI"]
ARXIV_MAX_RESULTS: int = 100          # 每個分類最多抓取筆數
ARXIV_LOOKBACK_HOURS: int = 24        # 僅抓取過去 N 小時的論文（24 小時）
ARXIV_TOTAL_MAX: int = 20             # 所有分類去重後的論文總上限

NEWS_LOOKBACK_HOURS: int = 24         # 新聞日期過濾器：僅保留過去 N 小時內的新聞

RSS_TIMEOUT: int = 15                 # RSS / HTTP 請求逾時（秒）
RETRY_ATTEMPTS: int = 3              # 最大重試次數
RETRY_MIN_WAIT: int = 2              # 重試最小等待秒數
RETRY_MAX_WAIT: int = 10             # 重試最大等待秒數

HACKER_NEWS_TOP_URL: str = "https://hacker-news.firebaseio.com/v0/topstories.json"
HACKER_NEWS_ITEM_URL: str = "https://hacker-news.firebaseio.com/v0/item/{id}.json"
HACKER_NEWS_FETCH_COUNT: int = 30    # 從 Top Stories 取前 N 篇篩選

NEWS_SOURCES: List[Dict[str, str]] = [
    {
        "name": "TechCrunch AI",
        "type": "rss",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
    },
    {
        "name": "Google Research Blog",
        "type": "rss",
        "url": "https://blog.research.google/feeds/posts/default",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    """回傳帶有時區資訊的 UTC 當前時間。"""
    return datetime.now(timezone.utc)


def _truncate(text: Optional[str], max_len: int = 500) -> str:
    """截斷過長的文字，保留乾淨的摘要長度。"""
    if not text:
        return ""
    text = text.strip().replace("\n", " ")
    return text[:max_len] + "…" if len(text) > max_len else text


def _parse_news_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """
    將各種新聞日期字串解析為帶時區的 datetime。
    支援格式：
      - RFC 2822（RSS 標準，如 "Tue, 29 Apr 2026 00:40:16 +0000"）
      - ISO-8601（HN API，如 "2026-04-29T04:11:00+00:00"）
    無法解析時回傳 None（該筆資料將被保留，不過濾）。
    """
    if not date_str:
        return None
    # 嘗試 RFC 2822（feedparser 標準格式）
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # 嘗試 ISO-8601
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    return None


def _make_paper_record(
    title: str,
    authors: List[str],
    pdf_url: str,
    abstract: str,
    published: datetime,
    categories: List[str],
) -> Dict[str, Any]:
    return {
        "source_type": "Paper",
        "title": title.strip(),
        "authors": authors,
        "pdf_url": pdf_url,
        "abstract": _truncate(abstract),
        "published_at": published.isoformat(),
        "categories": categories,
        "fetched_at": _utcnow().isoformat(),
    }


def _make_news_record(
    source_name: str,
    title: str,
    url: str,
    summary: str,
    published: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "source_type": "News",
        "source_name": source_name,
        "title": title.strip(),
        "url": url,
        "summary": _truncate(summary),
        "published_at": published or "",
        "fetched_at": _utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ArXiv 論文抓取
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_arxiv_category(category: str) -> List[Dict[str, Any]]:
    """
    針對單一 ArXiv 分類（如 cat:cs.LG）抓取過去 ARXIV_LOOKBACK_HOURS 小時的論文。
    使用 arxiv 官方 Python SDK（v2.x），以 SortCriterion.SubmittedDate 排序。
    """
    cutoff = _utcnow() - timedelta(hours=ARXIV_LOOKBACK_HOURS)
    records: List[Dict[str, Any]] = []

    logger.info("📄 抓取 ArXiv 分類: %s（截止時間: %s）", category, cutoff.isoformat())

    try:
        client = arxiv.Client(
            page_size=50,
            delay_seconds=3,
            num_retries=RETRY_ATTEMPTS,
        )
        search = arxiv.Search(
            query=category,
            max_results=ARXIV_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        for result in client.results(search):
            pub_dt = result.published
            # 確保 datetime 帶有時區資訊
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)

            if pub_dt < cutoff:
                # 結果已按提交日期降序排列，超出時間窗口後可提前終止
                logger.debug("超出時間窗口，停止分類 %s 的抓取", category)
                break

            pdf_url = next(
                (link.href for link in result.links if link.title == "pdf"),
                result.pdf_url or "",
            )
            record = _make_paper_record(
                title=result.title,
                authors=[a.name for a in result.authors],
                pdf_url=pdf_url,
                abstract=result.summary,
                published=pub_dt,
                categories=result.categories,
            )
            records.append(record)

        logger.info("✅ ArXiv [%s] 共抓取 %d 篇論文", category, len(records))

    except Exception as exc:  # noqa: BLE001
        logger.error("❌ ArXiv [%s] 抓取失敗: %s", category, exc, exc_info=True)

    return records


def fetch_arxiv() -> List[Dict[str, Any]]:
    """
    平行抓取所有 ArXiv 分類，合併結果並依發表日期去重（以 ArXiv ID 為主鍵）。
    """
    all_records: Dict[str, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=len(ARXIV_CATEGORIES), thread_name_prefix="arxiv") as pool:
        futures = {
            pool.submit(_fetch_arxiv_category, cat): cat
            for cat in ARXIV_CATEGORIES
        }
        for future in as_completed(futures):
            cat = futures[future]
            try:
                for record in future.result():
                    # 以 PDF URL 中的 ArXiv ID 去重
                    key = record["pdf_url"] or record["title"]
                    all_records.setdefault(key, record)
            except Exception as exc:  # noqa: BLE001
                logger.error("❌ 彙整 ArXiv [%s] 結果時發生錯誤: %s", cat, exc)

    result_list = sorted(
        all_records.values(),
        key=lambda r: r["published_at"],
        reverse=True,
    )
    total_before = len(result_list)
    result_list = result_list[:ARXIV_TOTAL_MAX]   # 截取最新 N 篇
    logger.info(
        "📚 ArXiv 去重後共 %d 篇，限制20 篇後回傳 %d 篇（依發表日期由新至舊）",
        total_before, len(result_list),
    )
    return result_list


# ─────────────────────────────────────────────────────────────────────────────
# RSS 新聞抓取（含 Retry）
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError, OSError)),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _fetch_rss_with_retry(url: str) -> feedparser.FeedParserDict:
    """以 Retry 機制抓取 RSS Feed，逾時或連線錯誤時自動重試。"""
    response = requests.get(url, timeout=RSS_TIMEOUT, headers={"User-Agent": "fetcher-engine/1.0"})
    response.raise_for_status()
    return feedparser.parse(response.content)


def _fetch_single_rss(source: Dict[str, str]) -> List[Dict[str, Any]]:
    """抓取單一 RSS 來源並轉換為標準格式，僅保留 NEWS_LOOKBACK_HOURS 小時內的新聞。"""
    name = source["name"]
    url = source["url"]
    records: List[Dict[str, Any]] = []
    cutoff = _utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)

    logger.info("📰 抓取 RSS 來源: %s (%s)（截止: %s UTC）", name, url, cutoff.strftime("%Y-%m-%d %H:%M"))

    try:
        feed = _fetch_rss_with_retry(url)

        if feed.bozo and not feed.entries:
            logger.warning("⚠️  RSS [%s] 解析警告: %s", name, feed.bozo_exception)

        skipped = 0
        for entry in feed.entries:
            title = entry.get("title", "（無標題）")
            link = entry.get("link", "")
            summary = entry.get("summary", entry.get("description", ""))
            published_raw = entry.get("published", entry.get("updated", ""))

            # ── 日期過濾器 ──────────────────────────────────────
            pub_dt = _parse_news_datetime(published_raw)
            if pub_dt is not None and pub_dt < cutoff:
                skipped += 1
                continue   # 超出時間窗口，跳過
            # ────────────────────────────────────────────────────

            records.append(_make_news_record(
                source_name=name,
                title=title,
                url=link,
                summary=summary,
                published=published_raw,
            ))

        logger.info(
            "✅ RSS [%s] 保留 %d 則（過濾掉 %d 則超過 %d 小時的舊聞）",
            name, len(records), skipped, NEWS_LOOKBACK_HOURS,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("❌ RSS [%s] 抓取失敗（已跳過）: %s", name, exc, exc_info=True)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Hacker News API 抓取（含 Retry）
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def _hn_get(url: str) -> Any:
    """帶有 Retry 機制的 Hacker News API GET 請求。"""
    response = requests.get(url, timeout=RSS_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _fetch_hacker_news() -> List[Dict[str, Any]]:
    """
    從 Hacker News Firebase API 抓取 Top Stories，
    取前 HACKER_NEWS_FETCH_COUNT 篇，平行獲取各篇詳情。
    僅保留 NEWS_LOOKBACK_HOURS 小時內的新聞。
    """
    records: List[Dict[str, Any]] = []
    cutoff = _utcnow() - timedelta(hours=NEWS_LOOKBACK_HOURS)
    cutoff_ts = int(cutoff.timestamp())  # 轉為 Unix timestamp 以利比對
    logger.info(
        "📰 抓取 Hacker News Top Stories（前 %d 篇，截止: %s UTC）",
        HACKER_NEWS_FETCH_COUNT, cutoff.strftime("%Y-%m-%d %H:%M"),
    )

    try:
        story_ids: List[int] = _hn_get(HACKER_NEWS_TOP_URL)
        top_ids = story_ids[:HACKER_NEWS_FETCH_COUNT]
    except Exception as exc:  # noqa: BLE001
        logger.error("❌ Hacker News Top Stories 清單抓取失敗（已跳過）: %s", exc)
        return records

    def _fetch_item(story_id: int) -> Optional[Dict[str, Any]]:
        try:
            item = _hn_get(HACKER_NEWS_ITEM_URL.format(id=story_id))
            if not item or item.get("type") != "story":
                return None
            title = item.get("title", "")
            url = item.get("url") or f"https://news.ycombinator.com/item?id={story_id}"
            text = item.get("text", "")  # 部分純討論文章有 text 欄位
            score = item.get("score", 0)
            published_ts = item.get("time")

            # ── 日期過濾器 ──────────────────────────────────────
            if published_ts and published_ts < cutoff_ts:
                return None   # 超出時間窗口，丟棄
            # ────────────────────────────────────────────────────

            published_str = (
                datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat()
                if published_ts else ""
            )
            summary = _truncate(text) if text else f"HN score: {score}"
            return _make_news_record(
                source_name="Hacker News",
                title=title,
                url=url,
                summary=summary,
                published=published_str,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("⚠️  HN item %s 抓取失敗（已跳過）: %s", story_id, exc)
            return None

    with ThreadPoolExecutor(max_workers=10, thread_name_prefix="hn") as pool:
        futures = [pool.submit(_fetch_item, sid) for sid in top_ids]
        for future in as_completed(futures):
            result = future.result()
            if result:
                records.append(result)

    logger.info("✅ Hacker News 保留 %d 則（%d 小時內）", len(records), NEWS_LOOKBACK_HOURS)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 新聞總入口
# ─────────────────────────────────────────────────────────────────────────────

def fetch_news() -> List[Dict[str, Any]]:
    """
    平行抓取所有新聞來源（Hacker News + RSS），回傳統一格式的新聞列表。
    套用 NEWS_LOOKBACK_HOURS 日期過濾器，僅保留近期新聞。
    任一來源失敗不影響其他來源。
    """
    all_news: List[Dict[str, Any]] = []

    tasks = []

    with ThreadPoolExecutor(max_workers=5, thread_name_prefix="news") as pool:
        # Hacker News
        tasks.append(pool.submit(_fetch_hacker_news))

        # RSS 來源
        for source in NEWS_SOURCES:
            tasks.append(pool.submit(_fetch_single_rss, source))

        for future in as_completed(tasks):
            try:
                all_news.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                logger.error("❌ 彙整新聞結果時發生未預期錯誤: %s", exc)

    logger.info("📰 新聞總計 %d 則", len(all_news))
    return all_news


# ─────────────────────────────────────────────────────────────────────────────
# 主要入口函式
# ─────────────────────────────────────────────────────────────────────────────

def run_all_fetchers() -> List[Dict[str, Any]]:
    """
    同時啟動論文與新聞抓取，合併後回傳統一的 List[Dict]。

    每筆資料結構：
    {
        "source_type": "Paper" | "News",
        "title": str,
        "authors": List[str],           # 僅 Paper
        "pdf_url": str,                 # 僅 Paper
        "abstract": str,                # 僅 Paper
        "categories": List[str],        # 僅 Paper
        "source_name": str,             # 僅 News
        "url": str,                     # 僅 News
        "summary": str,                 # 僅 News
        "published_at": str,            # ISO-8601
        "fetched_at": str,              # ISO-8601
    }
    """
    start = time.perf_counter()
    logger.info("🚀 啟動 fetcher_engine — %s", _utcnow().isoformat())

    all_data: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="engine") as pool:
        future_papers = pool.submit(fetch_arxiv)
        future_news   = pool.submit(fetch_news)

        try:
            papers = future_papers.result()
            all_data.extend(papers)
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ 論文抓取模組發生嚴重錯誤: %s", exc, exc_info=True)

        try:
            news = future_news.result()
            all_data.extend(news)
        except Exception as exc:  # noqa: BLE001
            logger.error("❌ 新聞抓取模組發生嚴重錯誤: %s", exc, exc_info=True)

    elapsed = time.perf_counter() - start
    logger.info(
        "🏁 抓取完成｜論文: %d 篇｜新聞: %d 則｜耗時: %.2f 秒",
        sum(1 for d in all_data if d["source_type"] == "Paper"),
        sum(1 for d in all_data if d["source_type"] == "News"),
        elapsed,
    )
    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# CLI 快速測試入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all_fetchers()

    print("\n" + "=" * 60)
    print(f"  共取得 {len(results)} 筆資料")
    print("=" * 60)

    papers = [r for r in results if r["source_type"] == "Paper"]
    news   = [r for r in results if r["source_type"] == "News"]

    print(f"\n📄 論文（{len(papers)} 篇）— 前 3 篇預覽：")
    for p in papers[:3]:
        print(f"  [{p['published_at'][:10]}] {p['title']}")
        print(f"    作者: {', '.join(p['authors'][:3])}{'...' if len(p['authors']) > 3 else ''}")
        print(f"    PDF : {p['pdf_url']}")
        print()

    print(f"📰 新聞（{len(news)} 則）— 前 3 則預覽：")
    for n in news[:3]:
        print(f"  [{n['source_name']}] {n['title']}")
        print(f"    URL : {n['url']}")
        print()

    # 輸出完整 JSON（可選）
    output_path = "fetcher_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 完整結果已寫入 {output_path}")
