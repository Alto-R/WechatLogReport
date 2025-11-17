import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"

CHAT_STATS_PATH = OUTPUT_DIR / "chat_stats.json"
LLM_ANALYSIS_PATH = OUTPUT_DIR / "llm_analysis.json"
INTERACTION_STATS_PATH = OUTPUT_DIR / "interaction_time_stats.json"
REPORT_PATH = OUTPUT_DIR / "final_report.md"


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def top_n_dict(d: Dict[str, Any], n: int) -> List[tuple]:
    return sorted(d.items(), key=lambda x: x[1], reverse=True)[:n]


def seconds_to_human(s: float) -> str:
    if s is None:
        return "无数据"
    s = int(s)
    if s < 60:
        return f"{s} 秒"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m} 分 {sec} 秒"
    h, m = divmod(m, 60)
    return f"{h} 小时 {m} 分"


def format_time(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def format_time_range_text(range_dict: Dict[str, Any]) -> str:
    if not range_dict:
        return "时间未知"
    start = range_dict.get("start")
    end = range_dict.get("end")
    if start and end:
        return f"{format_time(start)} ~ {format_time(end)}"
    if start:
        return f"{format_time(start)} 起"
    if end:
        return f"{format_time(end)} 前"
    return "时间未知"


def build_report() -> str:
    chat_stats = load_json(CHAT_STATS_PATH)
    llm_data = load_json(LLM_ANALYSIS_PATH)
    interaction = load_json(INTERACTION_STATS_PATH)

    total_messages = chat_stats.get("total_messages")
    favorite_hours_readable = chat_stats.get("favorite_hours_readable") or []

    keywords_all = chat_stats.get("keywords_top") or []
    keywords_self = chat_stats.get("keywords_top_self") or []
    keywords_other = chat_stats.get("keywords_top_other") or []

    topic_distribution = llm_data.get("topic_distribution") or {}
    communication_patterns = llm_data.get("communication_patterns") or {}
    overall_summary = llm_data.get("overall_summary") or ""
    chunk_results = llm_data.get("chunk_results") or []
    time_segments = llm_data.get("time_segments") or []
    time_timeline_summary = llm_data.get("time_timeline_summary") or ""

    weekday_hour_counts = interaction.get("weekday_hour_counts") or {}
    interaction_stats = interaction.get("interaction_stats") or {}

    # 找出最活跃的星期几和小时
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    weekday_totals: Dict[str, int] = {}
    for w, hours in weekday_hour_counts.items():
        weekday_totals[w] = sum(int(v) for v in hours.values())
    top_weekdays = top_n_dict(weekday_totals, 3)

    hourly_counts = chat_stats.get("hourly_counts") or {}
    top_hours = top_n_dict({h: int(c) for h, c in hourly_counts.items()}, 3)

    sessions_total = interaction_stats.get("sessions_total")
    sessions_initiator = interaction_stats.get("sessions_initiator") or {}

    rt_self = interaction_stats.get("reply_time_self") or {}
    rt_other = interaction_stats.get("reply_time_other") or {}

    len_self = interaction_stats.get("length_self") or {}
    len_other = interaction_stats.get("length_other") or {}

    lines: List[str] = []

    lines.append("# 微信关系年度终极报告")
    lines.append("")
    lines.append("> 本报告由聊天记录统计 + 大模型分析自动生成，用于个人回顾与自我理解。")
    lines.append("")

    # 基本概览
    lines.append("## 一、基础概览")
    lines.append("")
    lines.append(f"- 总消息数：**{total_messages}** 条")
    if favorite_hours_readable:
        lines.append(f"- 最活跃时间段：**{', '.join(favorite_hours_readable)}**")
    lines.append("")

    lines.append("### 1.1 一天中的高频时段")
    for h, c in top_hours:
        lines.append(f"- **{h}:00-{h}:59**：{c} 条消息")
    lines.append("")

    lines.append("### 1.2 一周中的高频日子")
    for w, c in top_weekdays:
        w_name = weekday_names[int(w)] if w.isdigit() and 0 <= int(w) < 7 else f"星期 {w}"
        lines.append(f"- **{w_name}**：约 {c} 条消息")
    lines.append("")

    lines.append("（可视化参考：`weekday_hour_heatmap.png`）")
    lines.append("")

    # 互动模式
    lines.append("## 二、互动模式与回复习惯")
    lines.append("")
    lines.append(f"- 会话总数（按 30 分钟间隔切分）：**{sessions_total}** 段")
    lines.append(
        f"- 会话发起次数：我 **{sessions_initiator.get('self', 0)}** 段，对方 **{sessions_initiator.get('other', 0)}** 段"
    )
    lines.append("")

    lines.append("### 2.1 回复速度")
    lines.append(f"- 我回复对方：平均 **{seconds_to_human(rt_self.get('avg_seconds'))}**，中位数 **{seconds_to_human(rt_self.get('median_seconds'))}**")
    lines.append(f"- 对方回复我：平均 **{seconds_to_human(rt_other.get('avg_seconds'))}**，中位数 **{seconds_to_human(rt_other.get('median_seconds'))}**")
    lines.append("")
    lines.append("大致分布（次数）：")
    lines.append(f"- 我回复对方：{rt_self.get('histogram')}")
    lines.append(f"- 对方回复我：{rt_other.get('histogram')}")
    lines.append("")

    lines.append("### 2.2 话痨程度（消息长度）")
    lines.append(
        f"- 我：平均字数约 **{len_self.get('avg_chars')}**，中位数 **{len_self.get('median_chars')}**"
    )
    lines.append(
        f"- 对方：平均字数约 **{len_other.get('avg_chars')}**，中位数 **{len_other.get('median_chars')}**"
    )
    lines.append("")

    # 关键词
    lines.append("## 三、我们常说的话")
    lines.append("")

    lines.append("### 3.1 总体高频词（Top 20）")
    for w, c in keywords_all[:20]:
        lines.append(f"- **{w}**：{c} 次")
    lines.append("")

    if keywords_self:
        lines.append("### 3.2 我的高频词（Top 15）")
        for w, c in keywords_self[:15]:
            lines.append(f"- **{w}**：{c} 次")
        lines.append("")

    if keywords_other:
        lines.append("### 3.3 对方的高频词（Top 15）")
        for w, c in keywords_other[:15]:
            lines.append(f"- **{w}**：{c} 次")
        lines.append("")

    lines.append("（可视化参考：`wordcloud_total.png`、`wordcloud_self.png`、`wordcloud_other.png`）")
    lines.append("")

    # 话题与沟通模式
    lines.append("## 四、话题分布与沟通模式")
    lines.append("")

    lines.append("### 4.1 反复出现的话题（Top 15）")
    for t, c in top_n_dict(topic_distribution, 15):
        lines.append(f"- **{t}**：出现在 {c} 个对话片段中")
    lines.append("")

    lines.append("### 4.2 典型沟通模式（Top 10）")
    for p, c in top_n_dict(communication_patterns, 10):
        lines.append(f"- **{p}**：在 {c} 个对话片段中被识别到")
    lines.append("")

    lines.append("（可视化参考：`topics_distribution.png`、`communication_patterns.png`、`emotion_timeline.png`）")
    lines.append("")

    # 时间线展示
    lines.append("## 五、关键阶段时间线")
    lines.append("")
    if time_segments:
        for seg in time_segments:
            idx = seg.get("chunk_index")
            time_text = seg.get("time_text") or format_time_range_text(seg.get("time_range") or {})
            summary = seg.get("summary") or ""
            lines.append(f"- 第 {idx} 段（{time_text}）：{summary}")
        lines.append("")
    elif chunk_results:
        for r in sorted(chunk_results, key=lambda x: x.get("_chunk_index") or 0):
            idx = r.get("_chunk_index")
            time_text = format_time_range_text(r.get("time_range") or {})
            summary = r.get("summary") or ""
            lines.append(f"- 第 {idx} 段（{time_text}）：{summary}")
        lines.append("")
    else:
        lines.append("暂无时间线数据。")
        lines.append("")

    if time_timeline_summary:
        lines.append("**时间维度洞察：**")
        lines.append("")
        lines.append(time_timeline_summary.strip())
        lines.append("")

    # 整体分析总结
    lines.append("## 六、整体关系分析（大模型总结）")
    lines.append("")
    lines.append(overall_summary.strip() or "（尚未生成整体分析。）")
    lines.append("")

    # 若存在人物画像和故事，也一起挂上
    profile_self_path = OUTPUT_DIR / "profile_self.txt"
    profile_other_path = OUTPUT_DIR / "profile_other.txt"
    story_path = OUTPUT_DIR / "story.md"

    if profile_self_path.exists() or profile_other_path.exists():
        lines.append("## 七、人物画像")
        lines.append("")
        if profile_self_path.exists():
            lines.append("### 7.1 我的人物画像")
            lines.append("")
            text_self = profile_self_path.read_text(encoding="utf-8")
            lines.append(text_self.strip())
            lines.append("")
        if profile_other_path.exists():
            lines.append("### 7.2 对方的人物画像")
            lines.append("")
            text_other = profile_other_path.read_text(encoding="utf-8")
            lines.append(text_other.strip())
            lines.append("")

    if story_path.exists():
        lines.append("## 八、年度故事（回忆录版）")
        lines.append("")
        story = story_path.read_text(encoding="utf-8")
        lines.append(story.strip())
        lines.append("")

    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"终极报告已生成：{REPORT_PATH}")


if __name__ == "__main__":
    main()
