import re
from typing import Set
from langchain_core.tools import tool
from tools.pdf_report_generator import generate_pdf_report

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None

db = None
health_db = None
_cross_encoder = None

# 多言語対応の Cross-Encoder を使用して、クエリと文書の関連度を再スコアリングする
RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_K = 3
RERANK_CANDIDATE_K = 8


def init_db(db_instance, health_db_instance):
    global db, health_db
    db = db_instance
    health_db = health_db_instance


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    if CrossEncoder is None:
        return None

    try:
        _cross_encoder = CrossEncoder(RERANK_MODEL_NAME)
    except Exception:
        _cross_encoder = None
    return _cross_encoder


# 検索用に文字列を正規化する
def _normalize_text(text: str) -> str:
    return str(text).lower().replace("\u3000", " ").strip()


def _extract_query_keywords(query: str) -> Set[str]:
    """クエリから重み付け用キーワードを抽出する"""
    q = _normalize_text(query)

    # 「AST」など、明示された項目名を優先的に使う
    quoted_terms = re.findall(r"[「\"']([^」\"']+)[」\"']", q)

    tokens = set()
    for t in quoted_terms:
        t = t.strip()
        if len(t) >= 2:
            tokens.add(t)

    # 空白区切り語も補助的に使う（短すぎる語は除外）
    for t in re.split(r"\s+", q):
        t = t.strip(" ,.:;()[]{}")
        if len(t) >= 2:
            tokens.add(t)

    # 項目名が1文字のケースもあるため、引用語がある場合は保持
    for t in quoted_terms:
        t = t.strip()
        if t:
            tokens.add(t)

    return tokens


def _keyword_hit_count(text: str, keywords: Set[str]) -> int:
    if not keywords:
        return 0
    t = _normalize_text(text)
    return sum(1 for kw in keywords if kw and kw in t)


def _rerank_by_keywords(docs: list, query: str, top_k: int) -> list:
    """ベクトル検索結果をキーワード一致数で軽く再ランクする"""
    keywords = _extract_query_keywords(query)
    scored = []
    for idx, doc in enumerate(docs):
        content = getattr(doc, "page_content", "")
        score = _keyword_hit_count(content, keywords)
        scored.append((score, idx, doc))

    # 一致数降順、同点時は元の順位を優先
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in scored[:top_k]]


def _rerank_by_cross_encoder(docs: list, query: str, top_k: int) -> list:
    """Cross-Encoder による意味的reranking"""
    if not docs:
        return []

    reranker = _get_cross_encoder()
    if reranker is None:
        return _rerank_by_keywords(docs, query, top_k)

    pairs = [(query, getattr(doc, "page_content", "")) for doc in docs]
    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True,
    )
    return [doc for _, doc in scored_docs[:top_k]]


@tool
def search_kb(query: str) -> str:
    """健康診断結果に対する評価、評価の基準値および項目の説明を検索するツール

    Args:
        query: 検索したい健康診断項目名
        
    Returns:
        str: マッチした健康診断結果に対する評価（「A異常なし」「B軽度異常」「C要再検査・生活改善」「D要精密検査・治療」のいずれか）、評価の基準値および項目の説明
    """
    
    docs = db.similarity_search(query, k=8)
    ranked_docs = _rerank_by_keywords(docs, query, top_k=3)

    keywords = _extract_query_keywords(query)
    matched_docs = [
        d for d in ranked_docs
        if _keyword_hit_count(getattr(d, "page_content", ""), keywords) > 0
    ]
    
    if not docs or not matched_docs:
        return "【検索結果なし】\n\n該当する判定基準が見つかりませんでした。"
    
    result = "【データベース検索結果】\n\n"
    for i, doc in enumerate(matched_docs):
        result += f"[文書 {i+1} - ページ {doc.metadata.get('page', '?')}]\n"
        result += f"{doc.page_content}\n\n---\n\n"
    
    return result


@tool
def search_health_info(query: str) -> str:
    """健康情報データベースから改善策や対策を検索するツール
    
    Args:
        query: 検索クエリ（例：「尿酸 改善 食事」）
    
    Returns:
        str: 健康改善に関する情報
    """
    docs = health_db.similarity_search(query, k=RERANK_CANDIDATE_K)
    ranked_docs = _rerank_by_cross_encoder(docs, query, top_k=RERANK_TOP_K)
    
    if not ranked_docs:
        return "関連情報が見つかりませんでした。"
    
    result = "【健康情報データベース検索結果】\n\n"
    for i, doc in enumerate(ranked_docs, 1):
        source = doc.metadata.get("source", "不明")
        result += f"[参考{i}] (出典: {source})\n{doc.page_content}\n\n"
    
    return result


@tool
def generate_analysis_pdf(json_path: str, pdf_path: str) -> str:
    """分析結果JSONからレポートPDFを生成するツール

    Args:
        json_path: 分析結果JSONファイルパス
        pdf_path: 出力PDFファイルパス

    Returns:
        str: 実行結果メッセージ
    """
    try:
        generate_pdf_report(json_path, pdf_path)
        return f"PDF生成完了: {pdf_path}"
    except Exception as e:
        return f"PDF生成失敗: {str(e)}"
