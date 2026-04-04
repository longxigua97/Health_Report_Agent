from dotenv import load_dotenv
import os
import json
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tools.ocr_utils import extract_health_report_json_auto
from eval_agent import create_eval_agent, evaluate_exam_results
from advice_agent import create_advice_agent, format_questionnaire_for_prompt, generate_cd_advices
from Health_Report_Agent.tools.RAG_search_tools import generate_analysis_pdf, init_db



# 環境変数からAPIキーを読み込む
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# モデルの初期化
model = init_chat_model("openai:gpt-4o-mini", api_key=OPENAI_API_KEY)
vision_model = init_chat_model("openai:gpt-4o", api_key=OPENAI_API_KEY)

# RAG データベースの初期化
db_path = "/home/shambhala/Agent/Health_Report_Agent/Chroma-db-json"
health_db_path = "/home/shambhala/Agent/Health_Report_Agent/chroma_db_健康資料"
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

db = Chroma(
    persist_directory=db_path,
    embedding_function=embeddings
) 
health_db = Chroma(
    persist_directory=health_db_path,
    embedding_function=embeddings,
    collection_name="web_png_rag"
)

init_db(db, health_db)
print("✅ RAG データベース読み込み完了")

# PDF を読み取り、構造化 JSON を生成
input_file_path = "/home/shambhala/Agent/Health_Report_Agent/患者資料/健康診断結果_坂本一郎.pdf"
reference_json_path = "/home/shambhala/Agent/Health_Report_Agent/sample/json_format.json"
temp_image_dir = "/home/shambhala/Agent/Health_Report_Agent/middle_output"
extracted_text_path = "/home/shambhala/Agent/Health_Report_Agent/middle_output/extracted_input_text.json"
extracted_json_path = "/home/shambhala/Agent/Health_Report_Agent/middle_output/extracted_health_report_from_pdf.json"

# agent1 pdfからの健康診断結果を構造化するエージェントを実行
json_data_agent = extract_health_report_json_auto(
    input_file_path=input_file_path,
    reference_json_path=reference_json_path,
    temp_image_dir=temp_image_dir,
    extracted_text_path=extracted_text_path,
    extracted_json_path=extracted_json_path,
    model=model,
    vision_model=vision_model,
)

json_exam_results = json_data_agent.get("exam_results", {})
print(json_exam_results)
patient_info = json_data_agent.get("patient_info", {})
organization_info = json_data_agent.get("report_metadata", {})
questionnaire = json_data_agent.get("questionnaire", {})
print(questionnaire)

# 患者情報の表示
print(f"患者名: {patient_info.get('name','不明')}、性別: {patient_info.get('gender','不明')}、年齢: {patient_info.get('age','不明')}")
gender = patient_info.get('gender','不明')
age = patient_info.get('age','不明')

# agent2 健康診断検査結果を評価するエージェントを実行
eval_agent = create_eval_agent(model)
results = evaluate_exam_results(
    json_exam_results=json_exam_results,
    gender=gender,
    age=age,
    eval_agent=eval_agent,
)

# agent3 健康アドバイスを生成するエージェントを実行
advice_agent = create_advice_agent(model)
cd_advices = generate_cd_advices(
    results=results,
    patient_info=patient_info,
    gender=gender,
    age=age,
    questionnaire_text=format_questionnaire_for_prompt(questionnaire),
    advice_agent=advice_agent,
)

# 最終出力の構造を指定
final_output = {
    "report_metadata": organization_info,
    "patient_info": patient_info,
    "questionnaire": questionnaire,
    "other_sections": json_data_agent.get("other_sections", {}),
    "analysis_results": results,
    "cd_health_advices": cd_advices,
    "ocr_structured_report": json_data_agent
}
    
# 分析結果をJSONファイルに保存し、PDF生成ツールを呼び出す
output_file = '/home/shambhala/Agent/Health_Report_Agent/エージェント生成レポート/health_analysis_rag.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

output_pdf = '/home/shambhala/Agent/Health_Report_Agent/エージェント生成レポート/健康診断生成レポート.pdf'
pdf_tool_result = generate_analysis_pdf.invoke(
    {"json_path": output_file, "pdf_path": output_pdf}
)

print(f"\n✅ 分析結果を {output_file} に保存しました")
print(f"C/D評価項目数: {len(cd_advices)}")
print(f"📄 {pdf_tool_result}")
