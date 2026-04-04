import json
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
def format_numbered_text(text):
    """テキスト内の番号付きリストを改行形式に変換する"""
    import re
    # "1. ", "2. ", "3. " などの前に改行を挿入
    text = re.sub(r'(\d+)\.\s+\*\*', r'<br/>\1. **', text)
    # 太字でないパターンも処理
    text = re.sub(r'(?<!\*\*)(\d+)\.\s+(?!\*\*)', r'<br/>\1. ', text)
    # 先頭の不要な <br/> を削除
    text = text.lstrip('<br/>')
    return text

def format_improvements_text(text):
    """improvements テキスト内の改行を <br/> に変換する"""
    # JSON 内のエスケープされた \n を HTML 改行タグへ置換
    text = text.replace('\\n', '<br/>')
    # 実際の改行文字も同様に処理
    text = text.replace('\n', '<br/>')
    return text

def generate_pdf_report(json_path, pdf_filename="Health_Report.pdf"):
    # 1. JSON データを読み込む
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. PDF ドキュメントを設定
    doc = SimpleDocTemplate(pdf_filename, pagesize=A4, 
                            rightMargin=20, leftMargin=20, 
                            topMargin=30, bottomMargin=30)
    elements = []
    
    # 3. 日本語フォントを登録
    pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3'))
    
    # 4. スタイルを定義
    styles = getSampleStyleSheet()
    
    jp_style = ParagraphStyle(
        name='JP_Normal',
        parent=styles['Normal'],
        fontName='HeiseiMin-W3',
        fontSize=10,
        leading=14,
    )
    
    jp_title_style = ParagraphStyle(
        name='JP_Title',
        parent=styles['Title'],
        fontName='HeiseiMin-W3',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER
    )

    jp_small_style = ParagraphStyle(
        name='JP_Small',
        parent=styles['Normal'],
        fontName='HeiseiMin-W3',
        fontSize=8,
        leading=10,
    )
    
    jp_h2_style = ParagraphStyle(
        name='JP_H2',
        parent=jp_style,
        fontSize=14,
        leading=18,
        spaceAfter=12,
        textColor=colors.HexColor('#2E4053')
    )
    
    jp_h3_style = ParagraphStyle(
        name='JP_H3',
        parent=jp_style,
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=6,
        textColor=colors.HexColor('#5D6D7E')
    )
    
    # 新增：C/D評価の検査値強調スタイル
    jp_critical_value_style = ParagraphStyle(
        name='JP_CriticalValue',
        parent=jp_style,
        fontSize=12,
        leading=16,
        textColor=colors.white,           # 白文字
        backColor=colors.HexColor('#C0392B'),  # 深紅背景
        borderWidth=2,
        borderColor=colors.HexColor('#922B21'),  # 暗紅枠線
        borderPadding=8,
        leftIndent=10,
        rightIndent=10,
        alignment=TA_CENTER  # 中央揃え
    )
    
    # --- レポートタイトル ---
    meta = data.get("report_metadata", {})
    title_text = meta.get("report_title", "健康診断結果報告書")
    elements.append(Paragraph(title_text, jp_title_style))
    elements.append(Spacer(1, 20))
    
    # --- 患者/医療機関情報テーブル ---
    p_info = data.get("patient_info", {})
    
    info_data = [
        [f"実 施 日 : {meta.get('exam_date', '-')}", f"氏　名 : {p_info.get('name', '-')} 様"],
        [f"実施機関 : {meta.get('institution', '-')}", f"性　別 : {p_info.get('gender', '-')}"],
        [f"年 齢 : {p_info.get('age', '-')}歳", f"医師氏名 : {meta.get('doctor_name', '-')}"]
    ]
    

    col_widths = [260, 260]
    info_table = Table(info_data, colWidths=col_widths)
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'HeiseiMin-W3'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 15))
    
    # --- 判定結果一覧 ---
    elements.append(Paragraph("【判定結果一覧】", jp_h2_style))
    
    headers = ['カテゴリ', '検査項目', '結果', '判定', '基準範囲', '解説']
    table_data = [headers]
    
    analysis_results_raw = data.get("analysis_results", [])
    
    analysis_results = [
        item for item in analysis_results_raw 
        if "判定基準が見つかりませんでした" not in item.get('evaluation', '-') and "-" != item.get('evaluation', '-')
    ]
    
    # 判定結果に基づいてソート (D > C > その他)
    def sort_priority(item):
        ev = item.get('evaluation', '')
        if "D要精密検査・治療" in ev: return 0
        if "C要再検査・生活改善" in ev: return 1
        return 2

    analysis_results.sort(key=sort_priority)

    for item in analysis_results:
        category = item.get('category', '')
        name = item.get('item', '')
        value = str(item.get('value', ''))
        evaluation = item.get('evaluation', '-')
        normal_range = item.get('normal_range', '-')
        rationale = item.get('rationale', '')
        
        if "Error code:" in rationale:
            rationale = "※情報の取得に失敗しました"
        elif len(rationale) > 80:
            rationale = rationale[:80] + "..."
            
        row = [
            Paragraph(category, jp_small_style),
            Paragraph(name, jp_style),
            Paragraph(value, jp_style),
            Paragraph(evaluation, jp_style),
            Paragraph(normal_range, jp_small_style),
            Paragraph(rationale, jp_small_style)
        ]
        table_data.append(row)

    res_col_widths = [50, 100, 50, 90, 80, 180]
    result_table = Table(table_data, colWidths=res_col_widths, repeatRows=1)
    
    bs_style = [
        ('FONTNAME', (0,0), (-1,-1), 'HeiseiMin-W3'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0,0), (-1,0), colors.aliceblue),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 4),
    ]

    for i, item in enumerate(analysis_results, start=1):
        evaluation = item.get('evaluation', '')
        
        if evaluation == "A異常なし":
            pass
        elif evaluation in ["B軽度異常"]:
            bs_style.append(('TEXTCOLOR', (3,i), (3,i), colors.orange))
        elif evaluation in ["C要再検査・生活改善", "D要精密検査・治療"]:
            bs_style.append(('TEXTCOLOR', (3,i), (3,i), colors.red))
            bs_style.append(('BACKGROUND', (0,i), (-1,i), colors.lavenderblush))

    result_table.setStyle(TableStyle(bs_style))
    elements.append(result_table)
    elements.append(Spacer(1, 20))
    
    # --- C/D 健康アドバイス ---
    cd_advices = data.get("cd_health_advices", [])
    
    if cd_advices:
        elements.append(PageBreak())
        elements.append(Paragraph("【要注意項目の健康アドバイス】", jp_title_style))
        elements.append(Spacer(1, 15))
        
        for idx, adv in enumerate(cd_advices, 1):
            # 項目見出し
            item_title = f"{idx}. {adv.get('item', '')}（{adv.get('category', '')}）"
            
            # 検査値と判定を強調表示
            eval_label = adv.get('evaluation', '')
            value_text = f"検査値: <b>{adv.get('value', '')}</b>  ｜  判定: <b>{eval_label}</b>"
            
            elements.append(Paragraph(item_title, jp_h2_style))
            elements.append(Paragraph(value_text, jp_critical_value_style))  # 強調スタイル適用
            elements.append(Spacer(1, 12))
            
            advice = adv.get('advice', {})
            
            # 1. この検査値の意味
            elements.append(Paragraph("1. この検査値の意味", jp_h3_style))
            meaning_text = advice.get('meaning', '情報なし')
            elements.append(Paragraph(meaning_text, jp_style))
            elements.append(Spacer(1, 8))
            
            # 2. 考えられるリスク
            elements.append(Paragraph("2. 考えられるリスク", jp_h3_style))
            risks_text = advice.get('risks', '情報なし')
            elements.append(Paragraph(risks_text, jp_style))
            elements.append(Spacer(1, 8))
            
            # 3. 具体的な改善策（改行処理）
            elements.append(Paragraph("3. 具体的な改善策", jp_h3_style))
            improvements_text = advice.get('improvements', '情報なし')
            improvements_text = format_improvements_text(improvements_text)  # エスケープされた \n を処理
            elements.append(Paragraph(improvements_text, jp_style))
            elements.append(Spacer(1, 8))
            
            # 4. 受診の推奨事項
            elements.append(Paragraph("4. 受診の推奨事項", jp_h3_style))
            medical_text = advice.get('medical_recommendation', '情報なし')
            elements.append(Paragraph(medical_text, jp_style))
            
            # 項目間の区切り
            if idx < len(cd_advices):
                elements.append(Spacer(1, 10))
                elements.append(Paragraph("─" * 67, jp_small_style))
                elements.append(Spacer(1, 10))

    # --- Disclaimer ---
    elements.append(Spacer(1, 30))
    disclaimer_text = "*Disclaimer: このプロジェクトは個人の健康管理をサポートするAIアシスタントであり、医師による診断を代替するものではありません。*"
    jp_disclaimer_style = ParagraphStyle(
        name='JP_Disclaimer',
        parent=styles['Normal'],
        fontName='HeiseiMin-W3',
        fontSize=9,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.gray
    )
    elements.append(Paragraph(disclaimer_text, jp_disclaimer_style))

    # 5. ファイルを生成
    doc.build(elements)
    print(f"✅ PDF Generated successfully: {pdf_filename}")

if __name__ == "__main__":
    input_json = '/home/shambhala/Agent/Health_Report_Agent/エージェント生成レポート/health_analysis_rag.json'
    output_pdf = "/home/shambhala/Agent/Health_Report_Agent/エージェント生成レポート/健康診断生成レポート.pdf"
    
    generate_pdf_report(input_json, output_pdf)