import base64
import json
import os
from typing import Any

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pdf2image import convert_from_path

from structuring_utils import (
    OCRHealthReport,
    build_structured_json_from_ocr,
    build_structured_json_from_text,
)


def encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    """PDF をページごとの画像に変換する"""
    os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(pdf_path, dpi=250)
    image_paths = []
    for i, image in enumerate(images, start=1):
        image_path = os.path.join(output_dir, f"page_{i}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    return image_paths


def ocr_pdf_pages(pdf_path: str, vision_model: Any, temp_dir: str) -> list[dict]:
    """PDF 全ページを OCR し、ページごとのテキストを返す"""
    print("\n📄 PDF を画像化して OCR を開始します...")
    image_paths = pdf_to_images(pdf_path, temp_dir)
    page_texts = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"  - OCR 実行中: page {idx}/{len(image_paths)}")
        b64 = encode_image_b64(image_path)
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "この画像は健康診断結果票です。"
                        "見出し・患者情報・検査項目名・検査値・判定・問診項目と回答を"
                        "できるだけ漏れなくOCR抽出してください。"
                        "出力はテキストのみで、表形式は行を保ってください。"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
                },
            ]
        )
        res = vision_model.invoke([msg])
        page_text = res.content.strip() if isinstance(res.content, str) else str(res.content)
        page_texts.append({"page": idx, "image_path": image_path, "text": page_text})

    print(f"✅ OCR 完了: {len(page_texts)} ページ")
    return page_texts


def create_health_report_json_agent(
    pdf_file_path: str,
    reference_json_path: str,
    temp_image_dir: str,
    ocr_pages_path: str,
    extracted_json_path: str,
    model,
    vision_model,
) -> dict:
    """PDF -> OCR -> 構造化 JSON の一連処理"""
    extraction_agent = create_agent(
        model=model,
        tools=[],
        response_format=OCRHealthReport,
        system_prompt=(
            "あなたは医療文書の構造化抽出の専門家です。"
            "OCRテキストを元に、健康診断情報を漏れなくJSON化してください。"
            "推測はせず、読み取れる情報を優先してください。"
        ),
    )

    ocr_pages = ocr_pdf_pages(pdf_file_path, vision_model, temp_image_dir)

    with open(ocr_pages_path, "w", encoding="utf-8") as f:
        json.dump(ocr_pages, f, ensure_ascii=False, indent=2)
    print(f"✅ OCRページ全文を保存: {ocr_pages_path}")

    json_data = build_structured_json_from_ocr(
        ocr_pages=ocr_pages,
        extraction_agent=extraction_agent,
        reference_json_path=reference_json_path,
    )

    with open(extracted_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"✅ OCR構造化JSONを保存: {extracted_json_path}")

    return json_data


def create_input_reader_agent(vision_model, temp_image_dir: str):
    """PDF 入力のみを読み取り、内容を抽出する Agent"""

    @tool
    def read_pdf_content(file_path: str) -> str:
        """PDF ファイルを読み取り、OCR で全文テキストを抽出する"""
        pages = ocr_pdf_pages(file_path, vision_model, temp_image_dir)
        return "\n\n".join([f"[PAGE {p['page']}]\n{p['text']}" for p in pages])

    return create_agent(
        model=init_chat_model("openai:gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY")),
        tools=[read_pdf_content],
        system_prompt="""あなたは入力ファイル読取エージェントです。

ユーザーが指定したファイルを読み取る際は、必ず read_pdf_content ツールを1回呼び出してください。
このエージェントは PDF ファイル専用です。

出力はツール実行結果の生テキストのみ返してください。""",
    )


def extract_health_report_json_auto(
    input_file_path: str,
    reference_json_path: str,
    temp_image_dir: str,
    extracted_text_path: str,
    extracted_json_path: str,
    model,
    vision_model,
) -> dict:
    """PDF を読み取り、構造化 JSON を生成する"""
    extraction_agent = create_agent(
        model=model,
        tools=[],
        response_format=OCRHealthReport,
        system_prompt=(
            "あなたは医療文書の構造化抽出の専門家です。"
            "PDF から抽出したテキストを元に、健康診断JSONを作成してください。"
            "推測はせず、必ず文書から情報を抽出してください。"
        ),
    )

    if os.path.splitext(input_file_path)[1].lower() != ".pdf":
        raise ValueError("この処理は PDF ファイルのみ対応です。")

    reader_agent = create_input_reader_agent(vision_model=vision_model, temp_image_dir=temp_image_dir)
    read_query = f"ファイルを読み取ってください: {input_file_path}"
    read_response = reader_agent.invoke({"messages": [{"role": "user", "content": read_query}]})
    extracted_text = read_response.get("messages", [])[-1].content if read_response.get("messages") else ""
    if isinstance(extracted_text, list):
        extracted_text = "\n".join([str(x) for x in extracted_text])
    extracted_text = str(extracted_text).strip()

    if not extracted_text:
        raise ValueError("入力ファイルからテキスト抽出に失敗しました。")

    with open(extracted_text_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_file": input_file_path,
                "file_type": os.path.splitext(input_file_path)[1].lower(),
                "extracted_text": extracted_text,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ 入力抽出テキストを保存: {extracted_text_path}")

    json_data = build_structured_json_from_text(
        source_text=extracted_text,
        extraction_agent=extraction_agent,
        reference_json_path=reference_json_path,
        source_type=os.path.splitext(input_file_path)[1].lower(),
    )

    with open(extracted_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"✅ 構造化JSONを保存: {extracted_json_path}")

    return json_data
