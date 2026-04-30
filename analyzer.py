"""
analyzer.py — 論文自動化系統的智能分析中樞

Dependencies:
    pip install google-genai requests python-dotenv

Usage:
    python analyzer.py
    # 或從其他模組呼叫：
    from analyzer import run_analyzer
    report = run_analyzer()
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import json
import logging
import os
import re
import tempfile
import time

# ── .env 自動載入（需安裝 python-dotenv）─────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()  # 自動尋找並載入同目錄下的 .env 檔
except ImportError:
    pass  # 若未安裝 python-dotenv，仍可透過系統環境變數提供 API Key
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# ── Third-Party ───────────────────────────────────────────────────────────────
# pip install google-genai requests python-dotenv
import requests
from google import genai
from google.genai import types as genai_types

# ─────────────────────────────────────────────────────────────────────────────
# Logging 設定
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("analyzer")

# ─────────────────────────────────────────────────────────────────────────────
# 常數設定
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL: str = "gemini-2.5-flash-lite"  # 配額獨立；可改為 gemini-2.5-flash（品質更高） / gemini-2.5-flash

INPUT_JSON: str = "fetcher_output.json"
OUTPUT_JSON: str = "final_report.json"

TOP_PAPER_COUNT: int = 3                         # 進行全文深度分析的論文篇數
PDF_TIMEOUT: int = 30                            # PDF 下載逾時（秒）
PDF_MAX_MB: int = 20                             # PDF 大小上限（MB），超過跳過
CALL_INTERVAL: float = 4.0                       # 每次 Gemini API 呼叫之間的間隔（秒）
MAX_RETRY: int = 4                               # 最大重試次數
SCORE_BATCH_SIZE: int = 5                        # 評分每批最大篇數，防止單次 Token 超限

# 關鍵領域與對應權重（用於評分 prompt 參考）
FOCUS_AREAS: List[str] = [
    "Machine Learning", "Deep Learning",
    "Explainable AI (XAI)", "Interpretability",
    "Model Optimization", "Model Compression",
    "Business Intelligence (BI)", "Data Analytics",
    "Large Language Models (LLM)", "Retrieval-Augmented Generation (RAG)",
    "Computer Vision", "Natural Language Processing",
]

# 用於關鍵字統計的常見標籤（正則比對）
KEYWORD_PATTERNS: Dict[str, List[str]] = {
    "LLM":           [r"\bLLM\b", r"large language model", r"language model"],
    "XAI":           [r"\bXAI\b", r"explainable", r"interpretab"],
    "RAG":           [r"\bRAG\b", r"retrieval.augmented"],
    "Transformer":   [r"transformer", r"attention mechanism", r"\bViT\b"],
    "RL":            [r"reinforcement learning", r"\bRLHF\b", r"\bRLVR\b"],
    "BI":            [r"business intelligence", r"\bBI\b", r"data analytics"],
    "Model Opt.":    [r"model compression", r"quantization", r"pruning", r"distillation"],
    "Multi-Agent":   [r"multi.agent", r"agentic", r"agent framework"],
    "CV":            [r"computer vision", r"object detection", r"image segmentation"],
    "NLP":           [r"natural language", r"\bNLP\b", r"text classification"],
    "Graph ML":      [r"graph neural", r"\bGNN\b", r"graph learning"],
    "Federated":     [r"federated learning"],
    "Diffusion":     [r"diffusion model", r"denoising"],
    "Cloud/Edge":    [r"edge computing", r"cloud computing", r"\bMEC\b"],
    "Security/AI":   [r"adversarial", r"jailbreak", r"AI safety", r"alignment"],
}

# 關鍵字停用詞：這些詞太過通用、無法提供有效的分析洞察
STOP_WORDS: set = {
    # 學術通用詞
    "analysis", "method", "methods", "paper", "papers", "result", "results",
    "using", "based", "approach", "approaches", "study", "research",
    "model", "models", "system", "systems", "data", "dataset", "datasets",
    "learning", "training", "evaluation", "performance", "framework",
    "proposed", "propose", "novel", "new", "improved", "better",
    "task", "tasks", "problem", "problems", "solution", "solutions",
    "network", "networks", "algorithm", "algorithms",
    "experiment", "experiments", "experimental", "benchmark", "benchmarks",
    "feature", "features", "input", "output", "process", "processing",
    "application", "applications", "technique", "techniques",
    "existing", "previous", "different", "various", "multiple",
    "show", "shows", "achieve", "achieves", "state", "art",
    # 常見功能詞
    "the", "and", "for", "with", "from", "that", "this", "are", "was",
    "can", "will", "has", "have", "been", "its", "our", "their",
    "also", "more", "than", "other", "each", "well", "use", "used",
    # 中文通用詞
    "分析", "方法", "研究", "結果", "模型", "系統", "論文", "使用",
    "提出", "實驗", "應用", "技術", "問題", "數據", "基於",
}

MIN_KEYWORD_LENGTH: int = 3  # 最短有效關鍵字長度

# ─────────────────────────────────────────────────────────────────────────────
# Gemini 客戶端初始化（使用新版 google-genai SDK）
# ─────────────────────────────────────────────────────────────────────────────

def _init_gemini() -> genai.Client:
    if not GEMINI_API_KEY:
        raise ValueError(
            "找不到 GEMINI_API_KEY！\n"
            "請在專案目錄下的 .env 檔案中設定：\n"
            "  GEMINI_API_KEY=your_api_key_here"
        )
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini 客戶端初始化完成，模型: %s", GEMINI_MODEL)
    return client


# ─────────────────────────────────────────────────────────────────────────────
# 工具函式
# ─────────────────────────────────────────────────────────────────────────────

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _safe_generate(client: genai.Client, prompt: str, tag: str = "") -> str:
    """
    呼叫 Gemini（新版 google-genai SDK），包含智能 Retry：
    - 429 配額錯誤：從錯誤訊息萃取 retry_delay 秒數，等待後再重試
    - 其他錯誤：指數退避等待
    - 每次呼叫前先等待 CALL_INTERVAL 秒防止觸發速率限制
    """
    for attempt in range(MAX_RETRY):
        time.sleep(CALL_INTERVAL)   # 主動限速，降低 429 機率
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as exc:
            exc_str = str(exc)
            wait = _extract_retry_delay(exc_str)
            if wait is None:
                wait = min(2 ** attempt * 5, 60)  # 一般錯誤： 5、15、30、60 秒
            logger.warning(
                "⚠️  Gemini 呼叫失敗 [%s] (嘗試 %d/%d)，等待 %.0f 秒再重試...錯誤: %s",
                tag, attempt + 1, MAX_RETRY, wait, exc,
            )
            time.sleep(wait)
    logger.error("❌ Gemini 呼叫 [%s] %d 次均失敗，回傳空字串", tag, MAX_RETRY)
    return ""


def _extract_retry_delay(exc_str: str) -> Optional[float]:
    """
    從 Gemini 429 錯誤訊息中萌取建議等待秒數。
    支援格式： 'retry in 58.75s' 或 'retry_delay { seconds: 58 }'
    """
    # 格式 1: retry in 58.75s
    m = re.search(r"retry in ([\d.]+)s", exc_str, re.IGNORECASE)
    if m:
        return float(m.group(1)) + 2  # 加 2 秒緩衝
    # 格式 2: retry_delay { seconds: 58 }
    m = re.search(r"seconds:\s*(\d+)", exc_str)
    if m:
        return float(m.group(1)) + 2
    # 格式 3: Please retry in 58
    m = re.search(r"[Rr]etry.*?(\d+\.?\d*)\s*s", exc_str)
    if m:
        return float(m.group(1)) + 2
    return None  # 無法解析，回傳 None 讓呼叫端使用預設策略


def _extract_json_block(text: str) -> str:
    """從 Gemini 回應中萃取 JSON 區塊（去除 markdown code fence）。"""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        return match.group(1).strip()
    # 嘗試直接找最外層的 { } 或 [ ]
    match = re.search(r"(\{[\s\S]+\}|\[[\s\S]+\])", text)
    return match.group(1).strip() if match else text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# PDF 下載與上傳
# ─────────────────────────────────────────────────────────────────────────────

def _download_pdf(pdf_url: str) -> Optional[bytes]:
    """下載 PDF，失敗或超過大小上限時回傳 None。"""
    try:
        logger.info("⬇️  下載 PDF: %s", pdf_url)
        resp = requests.get(pdf_url, timeout=PDF_TIMEOUT, stream=True,
                            headers={"User-Agent": "analyzer/1.0"})
        resp.raise_for_status()

        # 檢查 Content-Length
        content_length = int(resp.headers.get("Content-Length", 0))
        if content_length > PDF_MAX_MB * 1024 * 1024:
            logger.warning("⚠️  PDF 過大（%d MB），跳過全文分析", content_length // 1048576)
            return None

        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > PDF_MAX_MB * 1024 * 1024:
                logger.warning("⚠️  PDF 超過 %d MB 上限，截止下載", PDF_MAX_MB)
                return None
            chunks.append(chunk)

        logger.info("✅ PDF 下載完成（%.1f MB）", total / 1048576)
        return b"".join(chunks)

    except Exception as exc:
        logger.error("❌ PDF 下載失敗（已跳過）: %s", exc)
        return None


def _upload_pdf_to_gemini(
    client: genai.Client, pdf_bytes: bytes, display_name: str
) -> Optional[Any]:
    """將 PDF bytes 上傳至 Gemini Files API（新版 SDK），回傳 file 物件。"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        uploaded = client.files.upload(
            file=tmp_path,
            config=genai_types.UploadFileConfig(
                display_name=display_name,
                mime_type="application/pdf",
            ),
        )
        # 等待處理完成（新 SDK file state）
        for _ in range(10):
            file_info = client.files.get(name=uploaded.name)
            if file_info.state.name == "ACTIVE":
                break
            time.sleep(2)

        logger.info("✅ PDF 已上傳至 Gemini: %s", display_name)
        return uploaded

    except Exception as exc:
        logger.error("❌ PDF 上傳 Gemini 失敗（已跳過）: %s", exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 論文相關性評分
# ─────────────────────────────────────────────────────────────────────────────

def score_papers(
    client: genai.Client,
    papers: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    以 Gemini 對論文分批評分（0-10），每批 SCORE_BATCH_SIZE 篇，
    避免單次 Token 過多觸發 429。
    回傳原始 paper dict，新增 'relevance_score' 與 'score_reason' 欄位。
    """
    if not papers:
        return []

    focus_str = "、".join(FOCUS_AREAS)
    score_map: Dict[int, Any] = {}

    for batch_start in range(0, len(papers), SCORE_BATCH_SIZE):
        batch = papers[batch_start: batch_start + SCORE_BATCH_SIZE]

        items_text = "\n".join(
            f"[{batch_start + j}] 標題: {p['title']}\n"
            f"    摘要: {p.get('abstract','')[:250]}"
            for j, p in enumerate(batch)
        )
        batch_end_idx = batch_start + len(batch) - 1

        prompt = f"""你是一位 AI/ML 研究評審，請對以下 {len(batch)} 篇論文進行相關性評分。

評分標準（0-10分）：
- 10分：與以下領域高度相關且有重大突破：{focus_str}
- 7-9分：明確相關，有實用價值
- 4-6分：部分相關
- 0-3分：幾乎不相關

論文列表：
{items_text}

請輸出嚴格的 JSON 陣列，格式如下（不要有任何額外文字）：
[
  {{"index": {batch_start}, "score": 8, "reason": "一句話說明評分原因（中文）"}},
  ...
]"""

        logger.info("📊 評分第 %d-%d 篇...", batch_start + 1, batch_end_idx + 1)
        raw = _safe_generate(client, prompt, tag=f"score_{batch_start}")

        try:
            batch_scores = json.loads(_extract_json_block(raw))
            for item in batch_scores:
                score_map[item["index"]] = item
        except Exception as exc:
            logger.error("❌ 評分批次 %d-%d JSON 解析失敗: %s",
                         batch_start, batch_end_idx, exc)

    result = []
    for i, paper in enumerate(papers):
        info = score_map.get(i, {})
        paper = dict(paper)
        paper["relevance_score"] = info.get("score", 0)
        paper["score_reason"] = info.get("reason", "無法取得評分原因")
        result.append(paper)

    result.sort(key=lambda x: x["relevance_score"], reverse=True)
    logger.info("📊 論文評分完成，最高分: %.1f（%s）",
                result[0]["relevance_score"] if result else 0,
                result[0]["title"][:40] if result else "")
    return result



# ─────────────────────────────────────────────────────────────────────────────
# 論文深度摘要（含 PDF 全文）
# ─────────────────────────────────────────────────────────────────────────────

def deep_summarize_paper(
    client: genai.Client,
    paper: Dict[str, Any],
) -> Dict[str, Any]:
    """
    下載 PDF 並以 Gemini 進行深度分析。
    若 PDF 下載/上傳失敗，退回使用摘要文字分析。
    """
    title = paper["title"]
    logger.info("🔬 深度分析論文: %s", title[:60])

    pdf_part = None
    pdf_bytes = _download_pdf(paper.get("pdf_url", ""))
    if pdf_bytes:
        pdf_part = _upload_pdf_to_gemini(client, pdf_bytes, display_name=title[:80])

    # 構建 prompt
    base_info = f"""論文標題：{title}
作者：{', '.join(paper.get('authors', [])[:5])}
發表日期：{paper.get('published_at', '')[:10]}
摘要：{paper.get('abstract', '')}"""

    prompt = f"""請對以下論文進行深度學術分析，並以**繁體中文**輸出結果。

{base_info}

請輸出嚴格的 JSON 物件（不要有任何額外文字）：
{{
  "core_contribution": "核心貢獻（2-3句）",
  "innovation": "創新點：與現有方法的區別（2-3句）",
  "experiment_results": "主要實驗結果與數據（2-3句）",
  "bi_insight": "對商業智能/資料分析實務的啟發與應用建議（2-3句）",
  "chinese_summary": "整體中文摘要（100-150字）",
  "tags": ["標籤1", "標籤2", "標籤3"]
}}"""

    if pdf_part:
        raw = _safe_generate_with_file(client, prompt, pdf_part, tag=f"deep_{title[:20]}")
        # 分析完成後清理上傳的檔案
        try:
            client.files.delete(name=pdf_part.name)
        except Exception:
            pass
    else:
        logger.warning("⚠️  使用摘要文字進行分析（無 PDF）: %s", title[:40])
        raw = _safe_generate(client, prompt, tag=f"abstract_{title[:20]}")

    try:
        analysis = json.loads(_extract_json_block(raw))
    except Exception as exc:
        logger.error("❌ 深度摘要 JSON 解析失敗: %s", exc)
        analysis = {
            "core_contribution": "解析失敗",
            "innovation": "解析失敗",
            "experiment_results": "解析失敗",
            "bi_insight": "解析失敗",
            "chinese_summary": raw[:300] if raw else "無法取得摘要",
            "tags": [],
        }

    return {
        **paper,
        "analysis_type": "deep_with_pdf" if pdf_part else "abstract_only",
        **analysis,
    }


def _safe_generate_with_file(
    client: genai.Client,
    prompt: str,
    file_obj: Any,
    tag: str = "",
) -> str:
    """以 PDF file 物件呼叫 Gemini（新版 SDK），含智能 Retry。"""
    for attempt in range(MAX_RETRY):
        time.sleep(CALL_INTERVAL)
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[file_obj, prompt],
            )
            return response.text.strip()
        except Exception as exc:
            exc_str = str(exc)
            wait = _extract_retry_delay(exc_str)
            if wait is None:
                wait = min(2 ** attempt * 5, 60)
            logger.warning(
                "⚠️  Gemini PDF 分析失敗 [%s] (嘗試 %d/%d)，等待 %.0f 秒...錯誤: %s",
                tag, attempt + 1, MAX_RETRY, wait, exc,
            )
            time.sleep(wait)
    # 退回純文字
    logger.warning("⚠️  PDF 分析%d次失敗，改用純文字 prompt", MAX_RETRY)
    return _safe_generate(client, prompt, tag=tag)


# ─────────────────────────────────────────────────────────────────────────────
# 新聞摘要
# ─────────────────────────────────────────────────────────────────────────────

def summarize_news_batch(
    client: genai.Client,
    news_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """批次對新聞進行中文摘要（減少 API 呼叫次數）。"""
    if not news_items:
        return []

    items_text = "\n".join(
        f"[{i}] 來源: {n.get('source_name','')} | 標題: {n['title']}\n"
        f"    摘要: {n.get('summary','')[:300]}"
        for i, n in enumerate(news_items)
    )

    prompt = f"""你是一位科技新聞分析師，請對以下 {len(news_items)} 則新聞進行分析。

新聞列表：
{items_text}

請輸出嚴格的 JSON 陣列（不要有任何額外文字）：
[
  {{
    "index": 0,
    "chinese_summary": "繁體中文摘要（50-80字）",
    "trend_meaning": "趨勢意義（一句話）",
    "tags": ["標籤1", "標籤2"]
  }},
  ...
]"""

    raw = _safe_generate(client, prompt, tag="news_batch")

    try:
        analyses = json.loads(_extract_json_block(raw))
        analysis_map = {item["index"]: item for item in analyses}
    except Exception as exc:
        logger.error("❌ 新聞批次摘要 JSON 解析失敗: %s", exc)
        analysis_map = {}

    result = []
    for i, news in enumerate(news_items):
        info = analysis_map.get(i, {})
        result.append({
            **news,
            "chinese_summary": info.get("chinese_summary", "摘要解析失敗"),
            "trend_meaning": info.get("trend_meaning", ""),
            "tags": info.get("tags", []),
            "relevance_score": None,
        })

    logger.info("✅ 新聞摘要完成，共 %d 則", len(result))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 關鍵字統計
# ─────────────────────────────────────────────────────────────────────────────

def _clean_tag(tag: str) -> Optional[str]:
    """清洗單一標籤：轉小寫、去除頭尾空白，過濾停用詞與過短字串。"""
    t = tag.strip().lower()
    if len(t) < MIN_KEYWORD_LENGTH:
        return None
    if t in STOP_WORDS:
        return None
    return t


def extract_keyword_stats(items: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    從所有資料的標題、摘要、tags 中統計關鍵字出現次數。

    統計來源分兩層：
      1. KEYWORD_PATTERNS 正則比對（不分大小寫）
      2. Gemini 回傳的 tags 欄位（經 STOP_WORDS 與長度過濾）
    """
    counter: Counter = Counter()

    for item in items:
        # ── 合併文字欄位 ──────────────────────────────────────────────
        text_blob = " ".join([
            item.get("title", ""),
            item.get("abstract", ""),
            item.get("summary", ""),
            " ".join(item.get("tags", [])),
        ]).lower()

        # ── 第一層：KEYWORD_PATTERNS 正則比對 ─────────────────────────
        for keyword, patterns in KEYWORD_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text_blob, re.IGNORECASE):
                    counter[keyword] += 1
                    break  # 同一 keyword 在同一篇文章只計一次

        # ── 第二層：收集 Gemini 回傳的 tags（經清洗與過濾）────────────
        for raw_tag in item.get("tags", []):
            cleaned = _clean_tag(raw_tag)
            if cleaned:
                counter[cleaned] += 1

    # 最終輸出前，再次過濾整體結果中的停用詞（防禦性過濾）
    filtered = {
        k: v for k, v in counter.items()
        if k.lower() not in STOP_WORDS and len(k) >= MIN_KEYWORD_LENGTH
    }

    return dict(Counter(filtered).most_common())


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard 總覽生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_dashboard_summary(
    client: genai.Client,
    top_papers: List[Dict[str, Any]],
    news_items: List[Dict[str, Any]],
    keyword_stats: Dict[str, int],
) -> str:
    """生成今日科技動態的一句話總覽（繁體中文）。"""
    top_keywords = list(keyword_stats.keys())[:5]
    paper_titles = [p["title"] for p in top_papers[:3]]
    news_titles = [n["title"] for n in news_items[:3]]

    prompt = f"""請用一句話（50字以內，繁體中文）總結今日 AI/科技動態。

今日精選論文：{paper_titles}
今日科技新聞：{news_titles}
熱門關鍵字：{top_keywords}

輸出格式：只輸出一句話，不要任何其他內容。"""

    summary = _safe_generate(client, prompt, tag="dashboard")
    return summary or "今日 AI 領域持續高速發展，涵蓋 LLM、模型優化與多智能體等核心議題。"


# ─────────────────────────────────────────────────────────────────────────────
# 主分析流程
# ─────────────────────────────────────────────────────────────────────────────

def run_analyzer(
    input_path: str = INPUT_JSON,
    output_path: str = OUTPUT_JSON,
) -> Dict[str, Any]:
    """
    主要入口函式，執行完整分析流程並輸出 final_report.json。

    回傳完整報告 Dict。
    """
    start = time.perf_counter()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("🚀 analyzer 啟動 — %s", today)

    # ── 1. 初始化 ─────────────────────────────────────────────────────────────
    client = _init_gemini()

    # ── 2. 載入資料 ───────────────────────────────────────────────────────────
    logger.info("📂 載入 %s", input_path)
    all_items = _load_json(input_path)
    papers = [d for d in all_items if d.get("source_type") == "Paper"]
    news   = [d for d in all_items if d.get("source_type") == "News"]
    logger.info("📄 論文 %d 篇 | 📰 新聞 %d 則", len(papers), len(news))

    # ── 3. 論文相關性評分 ──────────────────────────────────────────────────────
    logger.info("📊 開始論文相關性評分...")
    scored_papers = score_papers(client, papers)

    # ── 4. Top N 論文深度分析（含 PDF）────────────────────────────────────────
    top_papers = scored_papers[:TOP_PAPER_COUNT]
    deep_analyzed: List[Dict[str, Any]] = []
    for paper in top_papers:
        result = deep_summarize_paper(client, paper)
        deep_analyzed.append(result)

    # ── 5. 其餘論文：僅保留評分結果（不做深度分析）───────────────────────────
    other_papers = []
    for p in scored_papers[TOP_PAPER_COUNT:]:
        other_papers.append({
            **p,
            "analysis_type": "scored_only",
            "chinese_summary": None,
        })

    # ── 6. 新聞批次摘要 ───────────────────────────────────────────────────────
    logger.info("📰 開始新聞批次摘要...")
    analyzed_news = summarize_news_batch(client, news)

    # ── 7. 關鍵字統計 ─────────────────────────────────────────────────────────
    logger.info("🔑 提取關鍵字統計...")
    all_analyzed = deep_analyzed + other_papers + analyzed_news
    keyword_stats = extract_keyword_stats(all_analyzed)

    # ── 8. Dashboard 總覽 ──────────────────────────────────────────────────────
    logger.info("📝 生成 Dashboard 總覽...")
    dashboard = generate_dashboard_summary(client, deep_analyzed, analyzed_news, keyword_stats)

    # ── 9. 組裝 analyzed_items ────────────────────────────────────────────────
    def _build_item(d: Dict[str, Any]) -> Dict[str, Any]:
        """統一輸出結構。"""
        return {
            "source_type":      d.get("source_type"),
            "source_name":      d.get("source_name", "ArXiv"),
            "title":            d.get("title", ""),
            "url":              d.get("pdf_url") or d.get("url", ""),
            "published_at":     d.get("published_at", ""),
            "relevance_score":  d.get("relevance_score"),
            "score_reason":     d.get("score_reason", ""),
            "analysis_type":    d.get("analysis_type", ""),
            "tags":             d.get("tags", []),
            # 論文專屬
            "authors":          d.get("authors", []),
            "categories":       d.get("categories", []),
            "core_contribution":d.get("core_contribution"),
            "innovation":       d.get("innovation"),
            "experiment_results":d.get("experiment_results"),
            "bi_insight":       d.get("bi_insight"),
            # 通用摘要
            "chinese_summary":  d.get("chinese_summary"),
            "trend_meaning":    d.get("trend_meaning"),
        }

    analyzed_items = [_build_item(d) for d in all_analyzed]

    # ── 10. 組裝最終報告 ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - start
    report: Dict[str, Any] = {
        "generated_at": _utcnow_iso(),
        "dashboard_summary": dashboard,
        "trend_data": {
            "date": today,
            "total_papers": len(papers),
            "total_news": len(news),
            "deep_analyzed_papers": len(deep_analyzed),
            "elapsed_seconds": round(elapsed, 1),
        },
        "keyword_stats": keyword_stats,
        "analyzed_items": analyzed_items,
    }

    # ── 11. 寫出 JSON ─────────────────────────────────────────────────────────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "🏁 分析完成｜論文 %d 篇（深度 %d 篇）｜新聞 %d 則｜耗時 %.1f 秒｜輸出: %s",
        len(papers), len(deep_analyzed), len(news), elapsed, output_path,
    )
    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    report = run_analyzer()

    print("\n" + "=" * 60)
    print("  📊 FINAL REPORT 預覽")
    print("=" * 60)
    print(f"\n📌 Dashboard: {report['dashboard_summary']}")
    print(f"\n📈 趨勢統計: {report['trend_data']}")
    print(f"\n🔑 關鍵字 Top 5:")
    for k, v in list(report["keyword_stats"].items())[:5]:
        print(f"    {k}: {v} 次")

    deep = [i for i in report["analyzed_items"] if i.get("analysis_type", "").startswith("deep")]
    print(f"\n🔬 深度分析論文（{len(deep)} 篇）:")
    for p in deep:
        print(f"  ★ [{p.get('relevance_score', '?')}分] {p['title'][:60]}")
        if p.get("core_contribution"):
            print(f"    核心貢獻: {p['core_contribution'][:80]}")

    print(f"\n✅ 完整報告已寫入 final_report.json")
