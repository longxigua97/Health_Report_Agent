from typing import Any
import time
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from tools.RAG_search_tools import search_kb

# 健康検査診断の構造化出力
class PersonInfo(BaseModel):
    """健康診断の個人データモデル"""
    category: str = Field(description="カテゴリ")
    item: str = Field(description="項目")
    value: str = Field(description="健康診断結果")
    evaluation: str = Field(description="健康診断結果に対する評価")
    normal_range: str = Field(description="評価の基準値")
    rationale: str = Field(description="項目の説明")


# RAGから検査項目の情報を探し、検査項目を評価するエージェント
def evaluate_exam_results(
    json_exam_results: dict,
    gender: str,
    age: Any,
    eval_agent,
) -> list:
    """eval_agent のみで検査結果を評価する"""
    print("\n" + "=" * 70)
    print("Step 1/2: 検査結果の評価を開始")
    print("=" * 70)

    results = []

    for category, items in json_exam_results.items():
        print(f"\n【カテゴリ: {category}】")

        for item, value in items.items():
            print(f"\n項目: {item}")
            print(f"結果: {value}")

            try:
                # カテゴリ名を含めてクエリを作成（「右」「左」など単一文字項目でも正しく検索できるようにする）
                retrieval_context = search_kb.invoke({"query": f"{category} {item}"})

                eval_query = f"""患者の性別は{gender}、年齢は{age}、健康診断項目「{item}」の検査結果が「{value}」です。

【検索済みデータベース情報（以下の情報のみを根拠として回答すること）】
{retrieval_context}

【タスク】
上記の検索情報のみを根拠として、以下を構造化して回答してください：
- category: 「{category}」
- item: 「{item}」
- value: 「{value}」
- evaluation: 健康診断結果に対する評価（「A異常なし」「B軽度異常」「C要再検査・生活改善」「D要精密検査・治療」または「-」）
- normal_range: 評価の基準値（見つからない場合は「-」）
- rationale: 項目の説明（見つからない場合は「-」、100字以内）

【厳守事項】
- 類似項目で代用しない
- 一般的な医学知識で補完しない
- 検索情報に根拠がない場合は evaluation を「-」にする
- 患者の性別と年齢に基づく回答をしてください"""

                response = eval_agent.invoke(
                    {"messages": [{"role": "user", "content": eval_query}]}
                )
                agent_result = response.get("structured_response") or response["messages"][-1].content

                # モデルの出力がすでに構造化されている場合と、JSON文字列の場合の両方に対応
                if isinstance(agent_result, PersonInfo):
                    result_dict = agent_result.model_dump()
                elif isinstance(agent_result, dict):
                    result_dict = PersonInfo.model_validate(agent_result).model_dump()
                else:
                    result_dict = PersonInfo.model_validate_json(agent_result).model_dump()

                print(f"評価: {result_dict['evaluation']}")
                results.append(result_dict)
                print("\n" + "=" * 70)
                time.sleep(2)  # TPMレート制限対策

            except Exception as e:
                print(f"\n【エラー】: {str(e)}")
                results.append(
                    {
                        "category": category,
                        "item": item,
                        "value": value,
                        "normal_range": "-",
                        "evaluation": "-",
                        "rationale": "-",
                    }
                )

    return results


# 健康検査診断のagent新規
def create_eval_agent(model):
    """評定（A/B/C/D）用 Agent"""
    return create_agent(
        model=model,
        tools=[search_kb],
        response_format=PersonInfo,
        system_prompt="""あなたは健康診断の専門家です。

【重要なルール】
1. 必ず search_kb ツールを使用して、患者の性別と年齢に基づいて、健康診断結果に対する最適な評価、評価の基準値および項目の説明を検索してください
2. 検索結果に該当する項目が明確に記載されている場合のみ、その情報を使って回答してください
3. 検索結果に該当する項目が見つからない場合は、evaluation に「-」を出力してください
4. 自分の知識や推測で判定基準を作り出さないでください
5. 検索結果の情報のみを使用し、訓練データの知識は使用しないでください
6. 回答は必ず構造化フォーマットで出力してください
""",
    )

