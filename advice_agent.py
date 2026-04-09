from typing import Any
import time
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from tools.RAG_search_tools import search_health_info


# 健康アドバイスの構造化出力
class HealthAdvice(BaseModel):
    """健康アドバイスの構造化モデル"""
    meaning: str = Field(description="この検査値の意味")
    risks: str = Field(description="考えられるリスク（問診結果との関連も含む）")
    improvements: str = Field(description="""具体的な改善策。必ず以下の形式で出力してください：
・食事: （食事に関する具体的なアドバイス）
・運動: （運動に関する具体的なアドバイス）
・生活習慣: （生活習慣に関する具体的なアドバイス）""")
    medical_recommendation: str = Field(description="受診の推奨事項")

# 健康アドバイスのagent新規
def create_advice_agent(model):
    """C/D 評価項目の健康アドバイス用 Agent"""
    return create_agent(
        model=model,
        tools=[search_health_info],
        response_format=HealthAdvice,
        system_prompt="""あなたは健康診断の専門アドバイザーです。

【タスク】
患者の検査結果、問診票の回答、健康情報データベースの検索結果を基に、具体的で実践的な健康アドバイスを提供してください。

【出力形式の厳守】
improvements フィールドは必ず以下の形式で出力してください：
・食事: （具体的な内容）
・運動: （具体的な内容）
・生活習慣: （具体的な内容）

各項目は必ず「・」で始め、改行で区切ってください。

【重要】
- search_health_info ツールを使って最新の健康情報を検索してください
- 問診結果と検査値を関連付けて説明してください
- 回答は日本語で、簡潔かつ分かりやすく記載してください
""",
    )


# 問診票の回答を指定構造に整形する関数
def format_questionnaire_for_prompt(questionnaire: dict) -> str:
    if not questionnaire:
        return "問診データなし"

    lines = ["【問診票の回答】"]
    for item in questionnaire.get("items", []):
        q_id = item.get("id")
        question = item.get("question", "")
        answer = item.get("answer", "")
        if "sub_items" in item:
            lines.append(f"質問{q_id}: {question}")
            for sub in item["sub_items"]:
                lines.append(f"  - {sub.get('question')}: {sub.get('answer')}")
        else:
            lines.append(f"質問{q_id}: {question} → {answer}")
    return "\n".join(lines)





# RAGから検査項目の情報を探し、検査項目を評価するエージェント
def generate_cd_advices(
    results: list,
    patient_info: dict,
    gender: str,
    age: Any,
    questionnaire_text: str,
    advice_agent,
) -> list:
    """advice_agent のみで C/D 評価項目のアドバイスを生成する"""
    print("\n" + "=" * 70)
    print("Step 2/2: C/D項目のアドバイス生成を開始")
    print("=" * 70)

    cd_advices = []
    target_results = [
        r for r in results if r.get("evaluation") in ["C要再検査・生活改善", "D要精密検査・治療"]
    ]

    for result_dict in target_results:
        item = result_dict.get("item", "")
        category = result_dict.get("category", "")
        value = result_dict.get("value", "")
        print(f"\n項目: {item} ({category})")
        print("  → 健康アドバイスを生成中...")

        try:
            advice_retrieval_context = search_health_info.invoke(
                {"query": f"{item} 改善 対策"}
            )

            advice_query = f"""以下の患者情報と異常項目について、健康アドバイスを作成してください。

【検索済み健康情報（以下の情報のみを根拠として回答すること）】
{advice_retrieval_context}

【患者情報】
患者名: {patient_info.get('name', '不明')}
性別: {gender}
年齢: {age}歳

{questionnaire_text}

【異常項目】
項目名: {item}
カテゴリ: {category}
検査結果: {value}
判定: {result_dict.get('evaluation', '-')}
基準範囲: {result_dict.get('normal_range', '-')}
説明: {result_dict.get('rationale', '-')}

【タスク】
上記の検索情報のみを根拠として、以下の構成で回答してください：

1. meaning: この検査値の意味を簡潔に説明

2. risks: 考えられるリスク（問診結果との関連も含む）

3. improvements: 具体的な改善策を**必ず以下の形式**で出力してください：
・食事: （具体的な食事改善策）
・運動: （具体的な運動推奨）
・生活習慣: （具体的な生活習慣改善策）

4. medical_recommendation: 受診の推奨事項

【厳守】
improvements は必ず「・食事:」「・運動:」「・生活習慣:」の3行で構成してください。"""

            advice_response = advice_agent.invoke(
                {"messages": [{"role": "user", "content": advice_query}]}
            )
            advice_result = advice_response.get("structured_response") or advice_response["messages"][-1].content

            # モデルの出力がすでに構造化されている場合と、JSON文字列の場合の両方に対応
            if isinstance(advice_result, HealthAdvice):
                advice_dict = advice_result.model_dump()
            elif isinstance(advice_result, dict):
                advice_dict = HealthAdvice.model_validate(advice_result).model_dump()
            else:
                advice_dict = HealthAdvice.model_validate_json(advice_result).model_dump()

            cd_advices.append(
                {
                    "item": item,
                    "category": category,
                    "value": value,
                    "evaluation": result_dict.get("evaluation", "-"),
                    "advice": advice_dict,
                }
            )
            print("  ✅ アドバイス生成完了")
            time.sleep(2)  # TPMレート制限対策
        except Exception as e:
            print(f"  ❌ アドバイス生成エラー: {str(e)}")

    return cd_advices
