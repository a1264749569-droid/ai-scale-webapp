from __future__ import annotations

from datetime import datetime
from pathlib import Path
from io import BytesIO

import pandas as pd
import streamlit as st

from core import (
    PRESETS,
    PROVIDER_PRESETS,
    build_bottomup_analysis_prompt,
    build_generation_prompt,
    build_topdown_filter_prompt,
    chat_complete,
    dimensions_to_text,
    export_excel_bytes,
    make_client,
    parse_bottomup_analysis,
    parse_dimension_counts,
    parse_dimensions,
    parse_generated_items,
    parse_topdown_filter,
    quality_report,
    rate_personas,
)

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "demo_data"


def _secret(section: str, key: str, default=""):
    try:
        block = st.secrets.get(section, {})
        if hasattr(block, "get"):
            return block.get(key, default)
    except Exception:
        pass
    return default


CLOUD_API_KEY = str(_secret("api", "api_key", "") or "")
CLOUD_PROVIDER = str(_secret("api", "provider", "OpenAI") or "OpenAI")
CLOUD_BASE_URL = str(_secret("api", "base_url", "") or "")
CLOUD_MODEL_ID = str(_secret("api", "model_id", "") or "")
CLOUD_API_AVAILABLE = bool(CLOUD_API_KEY and CLOUD_MODEL_ID)

st.set_page_config(page_title="AI 心理量表开发实验室", page_icon="🧠", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1280px;}
[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 12px 14px;}
.small-note {opacity:.72; font-size:.92rem;}
.hero {padding: 1rem 1.1rem; border:1px solid rgba(128,128,128,.22); border-radius:18px; margin-bottom:1rem;}
</style>
""",
    unsafe_allow_html=True,
)

for key, default in {
    "raw_df": pd.DataFrame(),
    "filtered_df": pd.DataFrame(),
    "bottomup_results": {},
    "analyses": {},
    "usage": None,
    "metadata": {},
    "ap_ratings": pd.DataFrame(),
    "last_prompt": "",
    "status": "尚未运行",
    "ap_demo_personas": pd.DataFrame(),
    "ap_demo_items": pd.DataFrame(),
    "ap_demo_ratings": pd.DataFrame(),
    "ap_demo_name": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown(
    """
<div class="hero">
<h2 style="margin:0 0 .35rem 0">AI 心理量表开发实验室</h2>
<div class="small-note">将原始脚本中的“生成 → 筛选/归纳 → AP 模拟作答”统一为可交互流程。支持自上而下与自下而上两种框架。</div>
</div>
""",
    unsafe_allow_html=True,
)

tab_generate, tab_results, tab_ap, tab_help = st.tabs(["① 生成量表", "② 结果与质量", "③ AP 模拟作答", "④ 使用说明"])

with tab_generate:
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.subheader("研究条件")
        run_modes = ["Demo 示例（无需 API）"]
        if CLOUD_API_AVAILABLE:
            run_modes.append("云端 API（无需输入 Key）")
        run_modes.append("自带 API Key")
        mode = st.radio("运行模式", run_modes, horizontal=True)
        preset_name = st.selectbox("构念模板", ["大五人格", "人工智能素养", "自定义"])
        if preset_name == "自定义":
            default_construct = ""
            default_definition = ""
            default_dims = ""
            default_n = 60
            default_counts = "3,4,5"
        else:
            p = PRESETS[preset_name]
            default_construct = preset_name
            default_definition = p["construct_definition"]
            default_dims = dimensions_to_text(p["dimensions"])
            default_n = p["num_items"]
            default_counts = p["bottom_up_counts"]

        construct_name = st.text_input("构念名称", value=default_construct, key=f"construct_name_{preset_name}")
        construct_definition = st.text_area("构念整体定义", value=default_definition, height=150, key=f"construct_definition_{preset_name}")
        framework = st.radio("生成框架", ["自上而下", "自下而上"], horizontal=True)
        style = st.radio("提示词风格", ["保守", "标准", "创造"], horizontal=True, index=1)
        num_items = st.number_input("候选条目数", min_value=12, max_value=300, value=int(default_n), step=6, key=f"num_items_{preset_name}")

        dimensions_text = ""
        dimension_counts_text = ""
        if framework == "自上而下":
            dimensions_text = st.text_area(
                "预设维度及定义（每行：维度：定义）",
                value=default_dims,
                height=210,
                key=f"dimensions_{preset_name}",
            )
        else:
            dimension_counts_text = st.text_input("候选维度数（逗号分隔）", value=default_counts, key=f"dimension_counts_{preset_name}")
            st.caption("与原脚本一致：大五人格可尝试 4,5,6；人工智能素养可尝试 3,4,5。")

    with right:
        st.subheader("模型与调用")
        if mode == "云端 API（无需输入 Key）":
            provider = CLOUD_PROVIDER if CLOUD_PROVIDER in PROVIDER_PRESETS else "自定义 OpenAI-compatible"
            pp = PROVIDER_PRESETS.get(provider, {"base_url": "", "model": ""})
            base_url = CLOUD_BASE_URL or pp["base_url"]
            model_id = CLOUD_MODEL_ID
            api_key = CLOUD_API_KEY
            st.success(f"已连接云端模型：{provider} / {model_id}。API Key 不会发送到浏览器。")
        else:
            provider = st.selectbox("模型服务", list(PROVIDER_PRESETS.keys()))
            pp = PROVIDER_PRESETS[provider]
            base_url = st.text_input("Base URL", value=pp["base_url"], help="OpenAI 官方可留空；其他服务可按实际接口修改。", key=f"base_{provider}")
            model_id = st.text_input("Model ID", value=pp["model"], help="模型名可按服务商当前可用模型手动修改。", key=f"model_{provider}")
            api_key = st.text_input("API Key", type="password", disabled=(mode != "自带 API Key"))
            if mode == "Demo 示例（无需 API）":
                st.caption("Demo 模式不调用任何外部模型，也不会产生 API 费用。")
            else:
                st.info("API Key 仅保存在当前网页会话内，不写入源码、Excel 或下载文件。")
        max_tokens = st.number_input("单次最大输出 tokens", min_value=2000, max_value=50000, value=18000, step=1000)

        if construct_name and construct_definition:
            try:
                dims_preview = parse_dimensions(dimensions_text) if framework == "自上而下" else None
                prompt_preview = build_generation_prompt(
                    construct_name, construct_definition, framework, style, int(num_items), dims_preview
                )
                with st.expander("查看本次生成 Prompt"):
                    st.code(prompt_preview, language="text")
            except Exception as e:
                st.warning(str(e))

    run = st.button("开始生成并筛选", type="primary", use_container_width=True)
    if run:
        try:
            if not construct_name.strip() or not construct_definition.strip():
                raise ValueError("请填写构念名称和整体定义。")
            dims = parse_dimensions(dimensions_text) if framework == "自上而下" else None
            dim_counts = parse_dimension_counts(dimension_counts_text) if framework == "自下而上" else []
            st.session_state.filtered_df = pd.DataFrame()
            st.session_state.bottomup_results = {}
            st.session_state.analyses = {}
            st.session_state.ap_ratings = pd.DataFrame()

            metadata = {
                "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "运行模式": mode,
                "模型服务": provider,
                "Model ID": model_id if not mode.startswith("Demo") else "Demo",
                "构念": construct_name,
                "框架": framework,
                "提示风格": style,
                "期望候选条目数": int(num_items),
            }

            if mode.startswith("Demo"):
                if preset_name == "自定义":
                    raise ValueError("Demo 模式提供大五人格与人工智能素养示例；自定义构念请使用真实 API。")
                if framework == "自上而下":
                    raw_file = DEMO / ("big5_topdown_raw.csv" if preset_name == "大五人格" else "ai_topdown_raw.csv")
                    filtered_file = DEMO / ("big5_topdown_filtered.csv" if preset_name == "大五人格" else "ai_topdown_filtered.csv")
                    raw = pd.read_csv(raw_file, encoding="utf-8-sig")
                    filtered = pd.read_csv(filtered_file, encoding="utf-8-sig")
                    if "题号" not in raw.columns:
                        raw.insert(0, "题号", range(1, len(raw) + 1))
                    st.session_state.raw_df = raw[[c for c in ["题号", "题目", "方向"] if c in raw.columns]]
                    st.session_state.filtered_df = filtered[["维度", "题目", "方向"]]
                else:
                    raw_file = DEMO / ("big5_bottomup_raw.csv" if preset_name == "大五人格" else "ai_bottomup_raw.csv")
                    raw = pd.read_csv(raw_file, encoding="utf-8-sig")
                    if "题号" not in raw.columns:
                        raw.insert(0, "题号", range(1, len(raw) + 1))
                    st.session_state.raw_df = raw[[c for c in ["题号", "题目", "方向"] if c in raw.columns]]
                    for k in dim_counts:
                        f = DEMO / f"{('big5' if preset_name == '大五人格' else 'ai')}_bottomup_dims{k}_filtered.csv"
                        a = DEMO / f"{('big5' if preset_name == '大五人格' else 'ai')}_bottomup_dims{k}_analysis.txt"
                        if f.exists():
                            st.session_state.bottomup_results[k] = pd.read_csv(f, encoding="utf-8-sig")[["维度", "题目", "方向"]]
                        if a.exists():
                            st.session_state.analyses[k] = a.read_text(encoding="utf-8")
                    if st.session_state.bottomup_results:
                        primary_k = min(st.session_state.bottomup_results, key=lambda x: abs(x - (5 if preset_name == "大五人格" else 4)))
                        st.session_state.filtered_df = st.session_state.bottomup_results[primary_k].copy()
                st.session_state.status = "Demo 已载入"
            else:
                if not api_key.strip() or not model_id.strip():
                    raise ValueError("API 模式必须具备 API Key 和 Model ID。")
                client = make_client(api_key, base_url)
                gen_prompt = build_generation_prompt(
                    construct_name, construct_definition, framework, style, int(num_items), dims
                )
                st.session_state.last_prompt = gen_prompt
                with st.status("正在调用模型生成候选条目…", expanded=True) as status:
                    text, usage = chat_complete(client, model_id, gen_prompt, int(max_tokens))
                    raw_df = parse_generated_items(text, int(num_items))
                    if raw_df.empty:
                        raise RuntimeError("模型返回内容未能解析出题目。请检查模型输出或调整 Prompt。")
                    st.write(f"已解析 {len(raw_df)} 个候选条目。")
                    st.session_state.raw_df = raw_df
                    st.session_state.usage = usage
                    if framework == "自上而下":
                        st.write("正在按预设维度筛选…")
                        filter_prompt = build_topdown_filter_prompt(construct_name, dims, raw_df)
                        ftext, _ = chat_complete(client, model_id, filter_prompt, int(max_tokens))
                        filtered = parse_topdown_filter(ftext, list(dims.keys()))
                        if filtered.empty:
                            raise RuntimeError("筛选结果未能解析。建议查看模型输出格式并重试。")
                        st.session_state.filtered_df = filtered
                    else:
                        for k in dim_counts:
                            st.write(f"正在归纳 {k} 维结构…")
                            aprompt = build_bottomup_analysis_prompt(construct_name, raw_df, k)
                            atext, _ = chat_complete(client, model_id, aprompt, int(max_tokens))
                            parsed = parse_bottomup_analysis(atext)
                            st.session_state.analyses[k] = atext
                            if not parsed.empty:
                                st.session_state.bottomup_results[k] = parsed
                        if not st.session_state.bottomup_results:
                            raise RuntimeError("所有维度归纳结果均解析失败。")
                        first_k = dim_counts[0]
                        if first_k not in st.session_state.bottomup_results:
                            first_k = next(iter(st.session_state.bottomup_results))
                        st.session_state.filtered_df = st.session_state.bottomup_results[first_k].copy()
                    status.update(label="生成与筛选完成", state="complete")
                st.session_state.status = "真实 API 已完成"
            st.session_state.metadata = metadata
            st.success("已完成。请切换到“② 结果与质量”查看、编辑和下载。")
        except Exception as e:
            st.error(f"运行失败：{e}")

with tab_results:
    st.subheader("结果预览与质量报告")
    raw_df = st.session_state.raw_df
    filtered_df = st.session_state.filtered_df
    if raw_df is None or raw_df.empty:
        st.info("请先在“① 生成量表”中运行 Demo 或真实 API。")
    else:
        usage = st.session_state.usage
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("候选条目", len(raw_df))
        c2.metric("筛选条目", len(filtered_df) if filtered_df is not None else 0)
        c3.metric("维度数", filtered_df["维度"].nunique() if filtered_df is not None and not filtered_df.empty and "维度" in filtered_df.columns else "—")
        c4.metric("状态", st.session_state.status)
        if usage:
            st.caption(f"生成耗时约 {usage.elapsed_seconds:.1f}s；总 tokens：{usage.total_tokens or '未返回'}；finish_reason：{usage.finish_reason or '未返回'}")

        st.markdown("#### 候选条目")
        st.dataframe(raw_df, use_container_width=True, hide_index=True, height=280)

        if st.session_state.bottomup_results:
            options = sorted(st.session_state.bottomup_results.keys())
            chosen_k = st.selectbox("查看哪一种自下而上维度方案", options, index=0)
            filtered_df = st.session_state.bottomup_results[chosen_k]
            st.session_state.filtered_df = filtered_df
            if chosen_k in st.session_state.analyses:
                with st.expander("查看完整维度归纳说明"):
                    st.markdown(st.session_state.analyses[chosen_k])

        if filtered_df is not None and not filtered_df.empty:
            st.markdown("#### 筛选后的正式候选量表（可直接编辑）")
            edited = st.data_editor(
                filtered_df,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="filtered_editor",
            )
            st.session_state.filtered_df = edited

            quality_df, duplicate_df = quality_report(edited)
            st.markdown("#### 自动质量报告")
            st.dataframe(quality_df, use_container_width=True, hide_index=True)
            if not duplicate_df.empty:
                with st.expander(f"查看 {len(duplicate_df)} 对高相似题目"):
                    st.dataframe(duplicate_df, use_container_width=True, hide_index=True)
            st.caption("自动质量报告用于快速检查方向比例、题长、维度均衡和潜在重复，不替代专家内容效度、EFA/CFA 或真实被试验证。")

            excel = export_excel_bytes(
                raw_df=raw_df,
                filtered_df=edited,
                quality_df=quality_df,
                analyses=st.session_state.analyses,
                bottomup_results=st.session_state.bottomup_results,
                metadata=st.session_state.metadata,
                ap_ratings=st.session_state.ap_ratings,
            )
            st.download_button(
                "下载完整 Excel 结果",
                data=excel,
                file_name=f"AI量表_{st.session_state.metadata.get('构念','结果')}_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

with tab_ap:
    st.subheader("AP 模拟作答")
    st.markdown(
        "AP（artificial persona）模拟用于展示：**人设描述 → 对候选量表逐题作答 → 形成响应矩阵 → 检查分布与维度表现**。"
        "完整 Demo 使用项目中已经产生的 AP 与既有模拟评分结果，因此无需 API、不会产生费用。"
    )

    ap_mode = st.radio("AP 运行模式", ["完整 Demo（无需 API）", "真实 AP 模拟（API）"], horizontal=True, key="ap_run_mode")

    if ap_mode == "完整 Demo（无需 API）":
        demo_kind = st.radio(
            "选择完整示例",
            ["大五人格：自下而上 5 维方案", "人工智能素养：自下而上 4 维方案"],
            horizontal=True,
            key="ap_demo_kind",
        )
        prefix = "big5" if demo_kind.startswith("大五") else "ai"
        demo_label = "大五人格" if prefix == "big5" else "人工智能素养"
        personas_path = DEMO / f"ap_demo_{prefix}_personas.csv"
        items_path = DEMO / f"ap_demo_{prefix}_items.csv"
        ratings_path = DEMO / f"ap_demo_{prefix}_ratings.csv"

        if st.button("一键载入完整 AP Demo", type="primary", use_container_width=True):
            try:
                st.session_state.ap_demo_personas = pd.read_csv(personas_path, encoding="utf-8-sig")
                st.session_state.ap_demo_items = pd.read_csv(items_path, encoding="utf-8-sig")
                st.session_state.ap_demo_ratings = pd.read_csv(ratings_path, encoding="utf-8-sig")
                st.session_state.ap_demo_name = demo_label
                st.session_state.filtered_df = st.session_state.ap_demo_items[["维度", "题目", "方向"]].copy()
                st.session_state.ap_ratings = st.session_state.ap_demo_ratings.copy()
                st.success(f"已载入 {demo_label} AP Demo。")
            except Exception as e:
                st.error(f"Demo 数据载入失败：{e}")

        personas_demo = st.session_state.ap_demo_personas
        items_demo = st.session_state.ap_demo_items
        ratings_demo = st.session_state.ap_demo_ratings

        if not personas_demo.empty and not items_demo.empty and not ratings_demo.empty:
            st.markdown("#### ① AP 人设")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("示例 AP 数", len(personas_demo))
            c2.metric("量表条目数", len(items_demo))
            c3.metric("维度数", items_demo["维度"].nunique())
            c4.metric("评分范围", "1–5")

            preview_cols = [c for c in ["id", "age", "education", "gender", "job", "description"] if c in personas_demo.columns]
            st.dataframe(personas_demo[preview_cols], use_container_width=True, hide_index=True, height=260)
            st.caption("Demo 人设来自项目原有 AP 数据；这里只截取 10 个 AP 用于竞赛展示。")

            st.markdown("#### ② 本次作答的候选量表")
            st.dataframe(items_demo[["题号", "维度", "题目", "方向"]], use_container_width=True, hide_index=True, height=300)

            st.markdown("#### ③ AP 响应矩阵")
            st.dataframe(ratings_demo, use_container_width=True, hide_index=True, height=320)

            qcols = [c for c in ratings_demo.columns if str(c).lower().startswith("q")]
            numeric = ratings_demo[qcols].apply(pd.to_numeric, errors="coerce")
            raw_long = numeric.stack().dropna()
            corrected = numeric.copy()
            directions = items_demo["方向"].astype(str).tolist()
            for i, q in enumerate(qcols):
                if i < len(directions) and "反" in directions[i]:
                    corrected[q] = 6 - corrected[q]

            st.markdown("#### ④ 快速统计摘要")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("平均原始评分", f"{raw_long.mean():.2f}")
            m2.metric("评分标准差", f"{raw_long.std():.2f}")
            m3.metric("最低评分", int(raw_long.min()))
            m4.metric("最高评分", int(raw_long.max()))

            left, right = st.columns(2, gap="large")
            with left:
                st.markdown("**1–5 分作答分布**")
                dist = raw_long.value_counts().sort_index()
                st.bar_chart(pd.DataFrame({"评分": dist.index.astype(int), "次数": dist.values}).set_index("评分"))
            with right:
                st.markdown("**各维度平均得分（反向题已校正）**")
                dim_rows = []
                for dim, idxs in items_demo.groupby("维度").groups.items():
                    cols = [qcols[i] for i in idxs if i < len(qcols)]
                    if cols:
                        dim_rows.append({"维度": dim, "平均分": corrected[cols].mean(axis=1).mean()})
                dim_df = pd.DataFrame(dim_rows).set_index("维度") if dim_rows else pd.DataFrame()
                if not dim_df.empty:
                    st.bar_chart(dim_df)

            with st.expander("查看 AP 个体平均分"):
                ids = ratings_demo["persona_id"] if "persona_id" in ratings_demo.columns else range(1, len(ratings_demo) + 1)
                person_summary = pd.DataFrame({"persona_id": ids, "校正后平均分": corrected.mean(axis=1).round(3)})
                st.dataframe(person_summary, use_container_width=True, hide_index=True)

            st.markdown("#### ⑤ 下载 Demo 数据")
            d1, d2, d3 = st.columns(3)
            d1.download_button("下载 AP 人设 CSV", personas_demo.to_csv(index=False).encode("utf-8-sig"), file_name=f"AP_demo_{demo_label}_personas.csv", mime="text/csv", use_container_width=True)
            d2.download_button("下载响应矩阵 CSV", ratings_demo.to_csv(index=False).encode("utf-8-sig"), file_name=f"AP_demo_{demo_label}_ratings.csv", mime="text/csv", use_container_width=True)
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                personas_demo.to_excel(writer, sheet_name="AP人设", index=False)
                items_demo.to_excel(writer, sheet_name="候选量表", index=False)
                ratings_demo.to_excel(writer, sheet_name="响应矩阵", index=False)
                if not dim_df.empty:
                    dim_df.reset_index().to_excel(writer, sheet_name="维度摘要", index=False)
            d3.download_button("下载完整 AP Demo Excel", output.getvalue(), file_name=f"AP_demo_{demo_label}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary", use_container_width=True)

            st.info("说明：此处展示的是项目既有 AP 与既有模拟评分结果的小型截取版，用于说明流程。正式研究仍应按预设模型、提示词、AP 数量与重复条件批量运行，并检验模拟数据的合理性与稳定性。")
        else:
            st.caption("点击上方“一键载入完整 AP Demo”即可在不调用 API 的情况下完整体验 AP 流程。")

    else:
        final_df = st.session_state.filtered_df
        if final_df is None or final_df.empty:
            st.info("真实 AP 模拟需要先在“① 生成量表”中生成或载入一套筛选后的量表。")
        else:
            source_mode = st.radio("AP 来源", ["示例 AP 文件", "上传自己的 AP CSV"], horizontal=True)
            if source_mode == "示例 AP 文件":
                sample_kind = st.radio("示例类型", ["大五人格 AP", "人工智能素养 AP"], horizontal=True)
                sample_path = DEMO / ("personas_big5_sample.csv" if sample_kind.startswith("大五") else "personas_ai_sample.csv")
                personas = pd.read_csv(sample_path, encoding="utf-8-sig")
            else:
                upload = st.file_uploader("上传 CSV（至少含 description 列；建议另有 id 列）", type=["csv"])
                personas = pd.read_csv(upload, encoding="utf-8-sig") if upload else pd.DataFrame()

            if not personas.empty:
                st.markdown("#### AP 人设预览")
                st.dataframe(personas.head(10), use_container_width=True, hide_index=True)
                limit = st.number_input("本次模拟 AP 数量", min_value=1, max_value=min(1000, len(personas)), value=min(5, len(personas)))
                pause = st.number_input("每次调用间隔（秒）", min_value=0.0, max_value=5.0, value=0.2, step=0.1)
                ap_modes = ["自带 API Key"]
                if CLOUD_API_AVAILABLE:
                    ap_modes.insert(0, "云端 API（无需输入 Key）")
                ap_api_mode = st.radio("AP 调用方式", ap_modes, horizontal=True, key="ap_api_mode")
                if ap_api_mode == "云端 API（无需输入 Key）":
                    provider_ap = CLOUD_PROVIDER if CLOUD_PROVIDER in PROVIDER_PRESETS else "自定义 OpenAI-compatible"
                    ap_base, ap_model, ap_key = CLOUD_BASE_URL, CLOUD_MODEL_ID, CLOUD_API_KEY
                    st.caption(f"使用云端模型：{provider_ap} / {ap_model}")
                else:
                    provider_ap = st.selectbox("AP 评分模型服务", list(PROVIDER_PRESETS.keys()), key="ap_provider")
                    ap_p = PROVIDER_PRESETS[provider_ap]
                    ap_base = st.text_input("AP Base URL", value=ap_p["base_url"])
                    ap_model = st.text_input("AP Model ID", value=ap_p["model"])
                    ap_key = st.text_input("AP API Key", type="password")
                st.warning("真实 AP 模拟需要每个 persona 至少调用一次模型。正式 1000 AP 会产生较多 API 调用与费用，建议先用 5–20 个 AP 测试。")
                if st.button("开始真实 AP 模拟评分", type="primary"):
                    try:
                        if not ap_key.strip() or not ap_model.strip():
                            raise ValueError("请填写 AP API Key 和 Model ID。")
                        client = make_client(ap_key, ap_base)
                        questions = final_df["题目"].astype(str).tolist()
                        prog = st.progress(0, text="准备开始…")
                        def cb(done, total):
                            prog.progress(done / total, text=f"AP {done}/{total}")
                        ratings = rate_personas(client, ap_model, personas, questions, int(limit), float(pause), progress_callback=cb)
                        st.session_state.ap_ratings = ratings
                        st.success("AP 模拟评分完成。")
                        st.dataframe(ratings, use_container_width=True, hide_index=True)
                        st.download_button("下载 AP 评分 CSV", ratings.to_csv(index=False).encode("utf-8-sig"), file_name="AP_simulated_ratings.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"AP 评分失败：{e}")
            else:
                st.caption("等待 AP 文件。")

with tab_help:
    st.subheader("这版网页与原项目的对应关系")
    st.markdown(
        """
- **自上而下**：输入整体构念 + 明确维度定义 → 大规模生成候选条目 → 按预设维度筛选。
- **自下而上**：只输入整体构念定义 → 大规模生成候选条目 → 对多个候选维度数进行语义归纳 → 形成不同结构方案。
- **提示词风格**：完整保留“保守 / 标准 / 创造”三类风格要求。
- **计分方向**：沿用原脚本约 70% 正向、30% 反向的生成目标。
- **AP 模拟**：提供“完整 Demo（无需 API）”和“真实 AP 模拟（API）”两条路径。Demo 直接展示 AP 人设、量表、响应矩阵、分布与维度摘要；真实模式沿用 persona description → 逐题 1–5 评分的程序逻辑。
- **多模型**：DeepSeek、Kimi、OpenAI、Gemini 均按原脚本的 OpenAI-compatible 调用方式预设，且 Base URL / Model ID 可以自行修改。
- **导出**：一次下载 Excel，包含运行信息、原始条目、筛选条目、质量报告、自下而上多维度方案与 AP 评分（如已运行）。
        """
    )
    st.info("安全改动：本应用不会把 API Key 写入源码；线上真实 API 请存放在 Streamlit Secrets。")
    st.markdown("#### 在线部署")
    st.markdown("将项目上传到 GitHub 后，可在 Streamlit Community Cloud 中选择 `app.py` 直接部署。只展示 Demo 时不需要配置 API Key；若需要现场真实生成，可在云端 Secrets 中配置 `[api]`。")
    st.code('[api]\nprovider = "OpenAI"\napi_key = "你的云端API密钥"\nbase_url = ""\nmodel_id = "填写当前可用的模型ID"', language="toml")
