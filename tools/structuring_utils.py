import json
from typing import Any, Dict
from pydantic import BaseModel, Field


class OCRHealthReport(BaseModel):
    """PDF OCR から構造化した健康診断データモデル"""
    report_metadata: Dict[str, Any] = Field(default_factory=dict)
    patient_info: Dict[str, Any] = Field(default_factory=dict)
    # LLM may return mixed shapes (nested dict or flat key/value), so accept Any then normalize.
    exam_results: Dict[str, Any] = Field(default_factory=dict)
    questionnaire: Dict[str, Any] = Field(default_factory=dict)
    other_sections: Dict[str, Any] = Field(default_factory=dict)


def normalize_exam_results_structure(exam_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """exam_results を必ず {category: {item: value}} の形に正規化する"""
    normalized: Dict[str, Dict[str, Any]] = {}

    if not isinstance(exam_results, dict):
        return {"その他": {"raw_exam_results": exam_results}}

    for key, value in exam_results.items():
        if isinstance(value, dict):
            normalized[key] = value
        else:
            if "その他" not in normalized:
                normalized["その他"] = {}
            normalized["その他"][key] = value

    return normalized


def build_structured_json_from_ocr(
    ocr_pages: list[dict],
    extraction_agent,
    reference_json_path: str,
) -> dict:
    """OCR テキストを参考 JSON 形式で構造化する"""
    with open(reference_json_path, "r", encoding="utf-8") as f:
        reference_json = json.load(f)

    merged_ocr_text = "\n\n".join(
        [f"[PAGE {p['page']}]\n{p['text']}" for p in ocr_pages]
    )

    extraction_query = f"""以下は健康診断PDFのOCR全文です。下記のテンプレートJSONの「-」をすべてOCR全文から抽出した実際の値で埋めてください。

【テンプレートJSON（「-」の箇所をOCR実測値で必ず上書きすること）】
{json.dumps(reference_json, ensure_ascii=False, indent=2)}

【フィールド別記入ルール】
- report_metadata: OCRから実施日・機関名・医師名・タイトルを探して記入
- patient_info: OCRから氏名・生年月日・性別・年齢・所属会社を探して記入
- exam_results:
  - テンプレートに存在する検査項目は必ず値を埋める
  - OCRで読めた追加の検査項目はカテゴリを判断して追記する（テンプレート外もOK）
  - 値は「数値 単位」の形式で正確に記入（例: 7.2 mg/dL）
- questionnaire:
  - テンプレートの各 item の「answer」フィールドにOCRから読み取った回答を記入
  - 選択肢番号と文言の両方が読めた場合は「②いいえ」のように記入
  - 質問文・選択肢・id はテンプレートのものをそのまま保持

【共通ルール】
1. OCRで明確に読めない値は "-" のまま残す（推測・補完禁止）
2. 出力は構造化JSONのみ（説明文・コメント不要）
3. テンプレートに含まれない情報は other_sections に保存

【OCR全文】
{merged_ocr_text}
"""

    response = extraction_agent.invoke(
        {"messages": [{"role": "user", "content": extraction_query}]}
    )
    extracted = response.get("structured_response") or response["messages"][-1].content

    if isinstance(extracted, OCRHealthReport):
        structured = extracted.model_dump()
    elif isinstance(extracted, dict):
        structured = OCRHealthReport.model_validate(extracted).model_dump()
    else:
        structured = OCRHealthReport.model_validate_json(extracted).model_dump()

    structured["exam_results"] = normalize_exam_results_structure(structured.get("exam_results", {}))

    if not structured.get("exam_results"):
        raise ValueError("OCR構造化結果に exam_results が含まれていません。")

    return structured


def build_structured_json_from_text(
    source_text: str,
    extraction_agent,
    reference_json_path: str,
    source_type: str,
) -> dict:
    """任意形式の抽出テキストを参考 JSON 形式で構造化する"""
    with open(reference_json_path, "r", encoding="utf-8") as f:
        reference_json = json.load(f)

    extraction_query = f"""以下は健康診断データの抽出テキスト（{source_type}形式）です。下記テンプレートJSONの「-」をすべて抽出テキストから読み取った実際の値で埋めてください。

【テンプレートJSON（「-」の箇所を実測値で必ず上書きすること）】
{json.dumps(reference_json, ensure_ascii=False, indent=2)}

【フィールド別記入ルール】
- report_metadata: テキストから実施日・機関名・医師名・タイトルを探して記入
- patient_info: テキストから氏名・生年月日・性別・年齢・所属会社を探して記入
- exam_results:
  - テンプレートに存在する検査項目は必ず値を埋める
  - テキストで読めた追加の検査項目はカテゴリを判断して追記する（テンプレート外もOK）
  - 値は「数値 単位」の形式で正確に記入（例: 7.2 mg/dL）
- questionnaire:
  - テンプレートの各 item の「answer」フィールドにテキストから読み取った回答を記入
  - 選択肢番号と文言の両方が読めた場合は「②いいえ」のように記入
  - 質問文・選択肢・id はテンプレートのものをそのまま保持

【共通ルール】
1. テキストで明確に読めない値は "-" のまま残す（推測・補完禁止）
2. 出力は構造化JSONのみ（説明文・コメント不要）
3. テンプレートに含まれない情報は other_sections に保存

【抽出テキスト全文】
{source_text}
"""

    response = extraction_agent.invoke(
        {"messages": [{"role": "user", "content": extraction_query}]}
    )
    extracted = response.get("structured_response") or response["messages"][-1].content

    if isinstance(extracted, OCRHealthReport):
        structured = extracted.model_dump()
    elif isinstance(extracted, dict):
        structured = OCRHealthReport.model_validate(extracted).model_dump()
    else:
        structured = OCRHealthReport.model_validate_json(extracted).model_dump()

    structured["exam_results"] = normalize_exam_results_structure(structured.get("exam_results", {}))

    if not structured.get("exam_results"):
        raise ValueError("構造化結果に exam_results が含まれていません。")

    return structured
