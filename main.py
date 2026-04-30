"""
main.py — 論文與科技新聞自動化系統的啟動進入點

執行流程：
    1. fetcher_engine.py  →  產出 fetcher_output.json
    2. analyzer.py        →  產出 final_report.json
    3. notion_sync.py     →  同步至 Notion

Usage:
    python main.py
"""

import logging
import os
import sys
import time

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# ─────────────────────────────────────────────────────────────────────────────
# 常數
# ─────────────────────────────────────────────────────────────────────────────
FETCHER_OUTPUT = "fetcher_output.json"
ANALYZER_OUTPUT = "final_report.json"

# 必要環境變數清單
REQUIRED_ENV = [
    "GEMINI_API_KEY",
    "NOTION_TOKEN",
    "DASHBOARD_PAGE_ID",
    "PAPER_DB_ID",
    "NEWS_DB_ID",
    "STATS_DB_ID",
    "TREND_DB_ID",
]


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────

def _check_env() -> bool:
    """檢查所有必要環境變數是否已設定。"""
    # 自動載入 .env（若存在）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        logger.error("❌ 缺少以下環境變數，請在 .env 中設定：")
        for v in missing:
            logger.error("   • %s", v)
        return False

    logger.info("✅ 環境變數檢查通過（%d/%d）", len(REQUIRED_ENV), len(REQUIRED_ENV))
    return True


def _file_exists(path: str) -> bool:
    """檢查檔案是否存在且非空。"""
    return os.path.isfile(path) and os.path.getsize(path) > 0


def _fmt_elapsed(seconds: float) -> str:
    """格式化耗時為可讀字串。"""
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    m, s = divmod(int(seconds), 60)
    return f"{m} 分 {s} 秒"


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    total_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("🚀 論文與科技新聞自動化系統啟動")
    logger.info("=" * 60)

    # ── 0. 環境變數檢查 ───────────────────────────────────────────────────────
    if not _check_env():
        sys.exit(1)

    # ── 1. 資料抓取 ───────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("📡 [階段 1/3] 開始抓取論文與新聞...")
    step_start = time.perf_counter()

    try:
        from fetcher_engine import run_all_fetchers
        import json as _json
        results = run_all_fetchers()
        with open(FETCHER_OUTPUT, "w", encoding="utf-8") as f:
            _json.dump(results, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.error("❌ 資料抓取失敗: %s", exc, exc_info=True)
        sys.exit(1)

    if not _file_exists(FETCHER_OUTPUT):
        logger.error("❌ %s 未生成或為空，流程中止", FETCHER_OUTPUT)
        sys.exit(1)

    step_elapsed = time.perf_counter() - step_start
    logger.info("✅ [階段 1/3] 資料抓取完成（%s）→ %s", _fmt_elapsed(step_elapsed), FETCHER_OUTPUT)

    # ── 2. AI 分析 ────────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("🧠 [階段 2/3] 開始 AI 分析與評分...")
    step_start = time.perf_counter()

    try:
        from analyzer import run_analyzer
        run_analyzer(input_path=FETCHER_OUTPUT, output_path=ANALYZER_OUTPUT)
    except Exception as exc:
        logger.error("❌ AI 分析失敗: %s", exc, exc_info=True)
        sys.exit(1)

    if not _file_exists(ANALYZER_OUTPUT):
        logger.error("❌ %s 未生成或為空，流程中止", ANALYZER_OUTPUT)
        sys.exit(1)

    step_elapsed = time.perf_counter() - step_start
    logger.info("✅ [階段 2/3] AI 分析完成（%s）→ %s", _fmt_elapsed(step_elapsed), ANALYZER_OUTPUT)

    # ── 3. Notion 同步 ────────────────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("📤 [階段 3/3] 開始同步至 Notion...")
    step_start = time.perf_counter()

    try:
        from notion_sync import run_notion_sync
        run_notion_sync()
    except Exception as exc:
        logger.error("❌ Notion 同步失敗: %s", exc, exc_info=True)
        sys.exit(1)

    step_elapsed = time.perf_counter() - step_start
    logger.info("✅ [階段 3/3] Notion 同步完成（%s）", _fmt_elapsed(step_elapsed))

    # ── 完成 ──────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info("🏁 全流程執行完畢！總耗時: %s", _fmt_elapsed(total_elapsed))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
