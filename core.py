from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROVIDER_PRESETS: Dict[str, Dict[str, str]] = {
    "OpenAI": {"base_url": "", "model": "gpt-5.4"},
    "DeepSeek": {"base_url": "https://api.deepseek.com", "model": "deepseek-v4-pro"},
    "Kimi": {"base_url": "https://api.moonshot.cn/v1", "model": "kimi-k3"},
    "Gemini (OpenAI-compatible)": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "model": "gemini-3.5-flash"},
    "自定义 OpenAI-compatible": {"base_url": "", "model": ""},
}

STYLE_REQUIREMENTS = {
    "保守": "7. 保持传统量表保守的语言风格，避免创新或实验性表达\n8. 保持传统量表的固定句式，句式结构应相对统一，避免频繁变化",
    "标准": "7. 使用常见、自然的量表语言表达，使题项符合一般心理学量表\n8. 使用符合大多数心理学量表的句式结构，句式结构可以适度变化",
    "创造": "7. 在保证心理学含义准确的基础上，使用更加多样、新颖、有启发性的语言表达\n8. 题项表述可以适度突破传统量表的固定句式，但仍需保持清晰、可理解、可作答",
}

PRESETS = {
    "大五人格": {
        "construct_definition": "人格是指个体内部相对稳定、持久的特质、倾向、动机和行为模式的整体组织，它决定了个体在适应环境时的独特思维、情感和行为方式。人格既包括个体内在的心理结构，也体现在跨时间、跨情境的行为一致性上。",
        "dimensions": {
            "神经质": "代表个体体验心理困扰并表现出适应不良反应的倾向，具体反映为焦虑、抑郁、愤怒、担忧和不安全感，其对立极为情绪稳定性。",
            "外向性": "体验积极情感的倾向，以及精力充沛、热情、自信果断的人际风格，具体表现为社交性、合群、健谈和活跃。",
            "开放性": "保持好奇心和想象力，并寻求新颖体验和观念的倾向，表现为富有想象力、创造性、思想开明等。",
            "宜人性": "信任、关怀和宽容的人际风格，表现为礼貌、温和、合作、宽容和富有同情心。",
            "尽责性": "自律、有条理、以成就为导向，并偏好有计划而非即兴行为的倾向，表现为可靠、负责、高效和坚持。",
        },
        "num_items": 180,
        "bottom_up_counts": "4,5,6",
    },
    "人工智能素养": {
        "construct_definition": "人工智能素养是指在实际应用中认识和理解人工智能技术的能力；能够熟练应用和利用人工智能技术完成任务的能力；能够分析、选择和批判性评估人工智能提供的数据和信息的能力，同时培养对自身个人责任的认识以及对相互权利和义务的尊重。",
        "dimensions": {
            "意识": "在使用人工智能相关应用的过程中，识别和理解人工智能技术的能力。",
            "使用": "熟练应用和利用人工智能技术完成任务的能力。",
            "评估": "分析、选择和批判性评估人工智能应用及其结果的能力。",
            "伦理": "意识到与使用人工智能技术相关的责任和风险的能力。",
        },
        "num_items": 144,
        "bottom_up_counts": "3,4,5",
    },
}

@dataclass
class UsageInfo:
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    finish_reason: Optional[str] = None

def dimensions_to_text(dimensions: Dict[str, str]) -> str:
    return "\n".join(f"{name}：{definition}" for name, definition in dimensions.items())

def parse_dimensions(text: str) -> Dict[str, str]:
    dims: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^\s*\d+[\.、\)）]\s*", "", line)
        if "：" in line:
            name, definition = line.split("：", 1)
        elif ":" in line:
            name, definition = line.split(":", 1)
        else:
            raise ValueError(f"维度行缺少冒号：{raw}")
        name, definition = name.strip(), definition.strip()
        if not name or not definition:
            raise ValueError(f"维度名称或定义为空：{raw}")
        dims[name] = definition
    if not dims:
        raise ValueError("至少需要输入一个维度及其定义。")
    return dims

def parse_dimension_counts(text: str) -> List[int]:
    vals = []
    for token in re.split(r"[,，\s]+", text.strip()):
        if not token:
            continue
        n = int(token)
        if n < 2 or n > 12:
            raise ValueError("候选维度数建议设置在 2 到 12 之间。")
        if n not in vals:
            vals.append(n)
    if not vals:
        raise ValueError("请至少输入一个候选维度数。")
    return vals

def build_generation_prompt(construct_name, construct_definition, framework, style, num_items, dimensions=None):
    base_req = "1. 每个题目都严格、准确地测量目标构念\n2. 题目应覆盖目标构念的核心内容和典型表现\n3. 题目表述应清晰、无歧义，避免隐喻化或过度修辞化的表达\n4. 避免双重问题、复合含义或可能引起不同解释的表述\n5. 不要让多个题项仅通过替换主语、程度副词或同义词形成重复，题目之间要有区分度\n6. 请确保每个题项都可以被被试通过 Likert 量表进行同意程度评分\n"
    if framework == "自上而下":
        if not dimensions:
            raise ValueError("自上而下框架需要提供预设维度。")
        dim_lines = "\n".join(f"{i}. {k}：{v}" for i, (k, v) in enumerate(dimensions.items(), 1))
        task = f"请根据以下构念内容生成{num_items}个‘{construct_name}’量表条目，并尽量使各维度候选条目数量均衡。\n目标构念：{construct_name}\n构念解释：{construct_definition}\n核心维度：\n{dim_lines}\n"
    else:
        task = f"请根据以下构念内容生成{num_items}个‘{construct_name}’量表条目。不要预设或输出固定分维度，而应尽可能扩大候选内容空间，为后续语义归纳和维度探索保留多样性。\n目标构念：{construct_name}\n构念解释：{construct_definition}\n"
    ending = f"采用5点李克特量表（1=非常不同意，5=非常同意）。\n目标上约70%条目为正向计分、30%为反向计分，并在题目末尾标注（正向）或（反向）。\n请直接输出{num_items}个题目，不添加解释，每行一个，格式为‘1. 题目内容（正向/反向）’。"
    return f"作为一名经验丰富且严谨的心理测量学家，{task}生成要求：\n{base_req}{STYLE_REQUIREMENTS[style]}\n{ending}"

def parse_generated_items(result_text: str, max_items: Optional[int] = None) -> pd.DataFrame:
    records = []
    for line in result_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^\s*[-*#]+\s*", "", line)
        cleaned = re.sub(r"^\s*\d+[\.、\)）:]\s*", "", cleaned)
        cleaned = re.sub(r"^题目\s*\d+\s*[：:]\s*", "", cleaned)
        if len(cleaned) < 3:
            continue
        direction = "反向" if re.search(r"[（(]\s*反向\s*[）)]", cleaned) else "正向" if re.search(r"[（(]\s*正向\s*[）)]", cleaned) else "未标注"
        item = re.sub(r"[（(]\s*(正向|反向)\s*[）)]", "", cleaned).strip()
        if item:
            records.append({"题号": len(records)+1, "题目": item, "方向": direction})
        if max_items and len(records) >= max_items:
            break
    return pd.DataFrame(records, columns=["题号", "题目", "方向"])

def build_topdown_filter_prompt(construct_name, dimensions, raw_df):
    dim_def = "\n".join(f"{i}. {name}：{definition}" for i, (name, definition) in enumerate(dimensions.items(), 1))
    items_text = "\n".join(f"{i+1}. {row['题目']}（{row['方向']}）" for i, (_, row) in enumerate(raw_df.iterrows()))
    return f"""你是一位心理测量学专家。请依据“{construct_name}”的预设维度定义，对候选题目进行分类并筛选。\n\n筛选要求：\n1. 与分维度定义高度匹配、表述清晰、无歧义。\n2. 避免语义重复。\n3. 每个维度筛选 3–9 道最合适的题目；各维度数量可以不同。\n4. 避免选择同时测量多个维度的题目。\n5. 尽量覆盖每个维度的不同表现形式，并适合正式心理学研究使用。\n6. 保留原题目的正向/反向方向；若原题未标注，可根据语义判断。\n\n维度定义：\n{dim_def}\n\n输出格式必须严格为：\n维度名称：\n1. 题目内容（正向/反向）\n2. 题目内容（正向/反向）\n下一维度名称：\n1. ...\n\n只输出上述结构，不要解释。\n\n候选题目：\n{items_text}"""

def parse_topdown_filter(content: str, known_dimensions: Sequence[str]) -> pd.DataFrame:
    normalized = content.replace("**", "").replace("##", "").replace("#", "")
    current_dim = None
    rows = []
    for raw in normalized.splitlines():
        line = raw.strip()
        if not line:
            continue
        heading = re.sub(r"^维度[一二三四五六七八九十\d]+\s*[：:]?\s*", "", line.rstrip("：:").strip())
        if heading in known_dimensions and (line.endswith("：") or line.endswith(":") or line == heading):
            current_dim = heading
            continue
        if current_dim and re.match(r"^\s*\d+[\.、\)）]", line):
            item_full = re.sub(r"^\s*\d+[\.、\)）]\s*", "", line).strip()
            direction = "反向" if "反向" in item_full else "正向" if "正向" in item_full else "未标注"
            item = re.sub(r"[（(][^）)]*(正向|反向)[^）)]*[）)]", "", item_full).strip()
            rows.append({"维度": current_dim, "题目": item, "方向": direction})
    return pd.DataFrame(rows, columns=["维度", "题目", "方向"])

def build_bottomup_analysis_prompt(construct_name, raw_df, num_dims):
    items_text = "\n".join(f"{i+1}. {row['题目']}（{row['方向']}）" for i, (_, row) in enumerate(raw_df.iterrows()))
    return f"""请基于“{construct_name}”的所有候选题项进行语义归纳、内容聚类和构念凝练。根据题项共同反映的心理含义、行为表现、认知过程、情绪体验、使用情境或价值取向，自然归纳出该主构念可能包含的 {num_dims} 个分维度（必须恰好 {num_dims} 个）。\n\n请为每个分维度：\n- 命名并定义；\n- 说明高分表现和低分表现；\n- 说明它为何属于主构念；\n- 从候选题目中筛选 3–9 道最合适的题目，各维度筛选数量相同；\n- 题目应与该维度高度匹配、清晰、避免重复、避免同时测量多个维度，并覆盖不同表现形式。\n\n请严格按以下结构输出：\n# {construct_name}分类体系：[体系名称]\n\n## 维度一：[名称]\n- 定义：……\n- 高分特征：……\n- 低分特征：……\n- 归属说明：……\n- 筛选题项：\n  1. 题目内容（正向/反向，维度名称）\n\n## 最终题目列表\n1. 题目内容（正向/反向，维度名称）\n\n不要添加结构之外的解释。\n\n候选题项：\n{items_text}"""

def parse_bottomup_analysis(content: str) -> pd.DataFrame:
    text = content.replace("**", "")
    marker = re.search(r"最终题目列表\s*[:：]?", text)
    block = text[marker.end():] if marker else text
    rows = []
    for raw in block.splitlines():
        line = raw.strip()
        if not re.match(r"^\s*\d+[\.、\)）]", line):
            continue
        value = re.sub(r"^\s*\d+[\.、\)）]\s*", "", line).strip()
        if len(value) < 3:
            continue
        parens = re.findall(r"[（(]([^）)]*)[）)]", value)
        direction, dimension = "未标注", "未知"
        if parens:
            inside = parens[-1]
            direction = "反向" if "反向" in inside else "正向" if "正向" in inside else "未标注"
            parts = [p.strip() for p in re.split(r"[,，]", inside) if p.strip()]
            candidates = [p for p in parts if p not in {"正向", "反向"}]
            if candidates:
                dimension = candidates[-1]
        item = re.sub(r"[（(][^）)]*[）)]", "", value).strip()
        rows.append({"维度": dimension, "题目": item, "方向": direction})
    return pd.DataFrame(rows, columns=["维度", "题目", "方向"])

def _usage_from_response(response, elapsed):
    usage = getattr(response, "usage", None)
    choice0 = response.choices[0] if getattr(response, "choices", None) else None
    return UsageInfo(getattr(usage,"prompt_tokens",None), getattr(usage,"completion_tokens",None), getattr(usage,"total_tokens",None), elapsed, getattr(choice0,"finish_reason",None))

def make_client(api_key: str, base_url: str = "") -> Any:
    from openai import OpenAI
    kwargs = {"api_key": api_key}
    if base_url.strip():
        kwargs["base_url"] = base_url.strip()
    return OpenAI(**kwargs)

def chat_complete(client, model, prompt, max_output_tokens=16000, retries=3) -> Tuple[str, UsageInfo]:
    last_error = None
    for attempt in range(retries):
        started = time.time()
        try:
            try:
                response = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], max_completion_tokens=max_output_tokens)
            except Exception as e:
                msg = str(e).lower()
                if "max_completion_tokens" not in msg and "unsupported" not in msg and "unknown" not in msg:
                    raise
                response = client.chat.completions.create(model=model, messages=[{"role":"user","content":prompt}], max_tokens=max_output_tokens)
            elapsed = time.time()-started
            return (response.choices[0].message.content or "").strip(), _usage_from_response(response, elapsed)
        except Exception as e:
            last_error = e
            if attempt < retries-1:
                time.sleep(min(2**attempt,4))
    raise RuntimeError(f"API 调用失败：{last_error}")

def build_persona_rating_prompt(persona_desc, questions):
    q_text = "\n".join(f"{i}. {q}" for i,q in enumerate(questions,1))
    return f"""你是一个真实的人，具有以下特征：\n{persona_desc}\n\n请对下面每个陈述句，根据你的真实感受用 1 到 5 分进行评分：\n1 = 非常不同意\n2 = 不同意\n3 = 中立\n4 = 同意\n5 = 非常同意\n\n严格按照题目顺序，只输出 {len(questions)} 个 1–5 的整数，每行一个分数，不要输出题号、解释或其他文字。\n\n题目列表：\n{q_text}"""

def parse_ratings(content: str, n_questions: int) -> List[int]:
    ratings=[]
    for line in content.splitlines():
        nums=re.findall(r"(?<!\d)([1-5])(?!\d)", line)
        if nums:
            ratings.append(int(nums[0]))
    if len(ratings)<n_questions:
        ratings.extend([3]*(n_questions-len(ratings)))
    return ratings[:n_questions]

def rate_personas(client, model, personas_df, questions, limit, pause_seconds=0.0, max_output_tokens=8000, progress_callback=None):
    if "description" not in personas_df.columns:
        raise ValueError("AP 文件必须包含 description 列。")
    rows=[]
    work=personas_df.head(limit).copy()
    for idx,(_,persona) in enumerate(work.iterrows(),1):
        text,_=chat_complete(client, model, build_persona_rating_prompt(str(persona["description"]), questions), max_output_tokens=max_output_tokens)
        ratings=parse_ratings(text,len(questions))
        pid=persona["id"] if "id" in persona.index else idx
        row={"persona_id":pid}
        row.update({f"q{i}":r for i,r in enumerate(ratings,1)})
        rows.append(row)
        if progress_callback: progress_callback(idx,len(work))
        if pause_seconds: time.sleep(pause_seconds)
    return pd.DataFrame(rows)

def quality_report(df):
    if df is None or df.empty:
        return pd.DataFrame([{"指标":"题目数","结果":0,"说明":"暂无题目"}]), pd.DataFrame(columns=["题目A","题目B","相似度"])
    texts=[str(x).strip() for x in df["题目"].fillna("")]
    n=len(texts)
    directions=df["方向"].fillna("未标注").astype(str) if "方向" in df.columns else pd.Series(["未标注"]*n)
    pos=int((directions=="正向").sum()); neg=int((directions=="反向").sum()); unlabeled=n-pos-neg
    lengths=[len(t) for t in texts if t]
    duplicate_rows=[]; max_sim=0.0
    if n>=2 and any(texts):
        try:
            vec=TfidfVectorizer(analyzer="char",ngram_range=(2,4),min_df=1).fit_transform(texts)
            sims=cosine_similarity(vec)
            for i in range(n):
                for j in range(i+1,n):
                    s=float(sims[i,j]); max_sim=max(max_sim,s)
                    if s>=0.82: duplicate_rows.append({"题目A":texts[i],"题目B":texts[j],"相似度":round(s,3)})
        except ValueError: pass
    rows=[{"指标":"题目数","结果":n,"说明":"当前进入质量报告的题目总数"},{"指标":"正向比例","结果":f"{pos/n*100:.1f}%" if n else "-","说明":f"正向 {pos} 题；原脚本目标约 70%"},{"指标":"反向比例","结果":f"{neg/n*100:.1f}%" if n else "-","说明":f"反向 {neg} 题；未标注 {unlabeled} 题"},{"指标":"平均题长","结果":f"{sum(lengths)/len(lengths):.1f} 字" if lengths else "-","说明":"仅作为语言长度描述，不是效度指标"},{"指标":"最高语义相似度","结果":f"{max_sim:.3f}","说明":"基于字符 n-gram TF-IDF 的启发式重复检查"},{"指标":"高相似题对","结果":len(duplicate_rows),"说明":"相似度 ≥ 0.82，建议人工复核"}]
    if "维度" in df.columns and not df["维度"].isna().all():
        counts=df["维度"].value_counts()
        rows += [{"指标":"维度数","结果":int(df["维度"].nunique()),"说明":"当前筛选结果中的唯一维度数"},{"指标":"维度题量差","结果":int(counts.max()-counts.min()) if len(counts) else None,"说明":"最大维度题数减最小维度题数"}]
    return pd.DataFrame(rows), pd.DataFrame(duplicate_rows, columns=["题目A","题目B","相似度"])

def export_excel_bytes(raw_df=None, filtered_df=None, quality_df=None, analyses=None, bottomup_results=None, metadata=None, ap_ratings=None):
    output=io.BytesIO()
    with pd.ExcelWriter(output,engine="openpyxl") as writer:
        if metadata: pd.DataFrame([metadata]).to_excel(writer,sheet_name="运行信息",index=False)
        if raw_df is not None and not raw_df.empty: raw_df.to_excel(writer,sheet_name="原始条目",index=False)
        if filtered_df is not None and not filtered_df.empty: filtered_df.to_excel(writer,sheet_name="筛选条目",index=False)
        if quality_df is not None and not quality_df.empty: quality_df.to_excel(writer,sheet_name="质量报告",index=False)
        if bottomup_results:
            for k,df in bottomup_results.items():
                if df is not None and not df.empty: df.to_excel(writer,sheet_name=f"自下而上_{k}维",index=False)
        if analyses: pd.DataFrame([{"维度数":k,"完整归纳文本":v} for k,v in analyses.items()]).to_excel(writer,sheet_name="维度归纳说明",index=False)
        if ap_ratings is not None and not ap_ratings.empty: ap_ratings.to_excel(writer,sheet_name="AP模拟评分",index=False)
    output.seek(0); return output.getvalue()
