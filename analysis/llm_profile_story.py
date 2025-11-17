import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import requests


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
CHAT_PATH = OUTPUT_DIR / "chat.json"
CHAT_STATS_PATH = OUTPUT_DIR / "chat_stats.json"
LLM_ANALYSIS_PATH = OUTPUT_DIR / "llm_analysis.json"

# 直接复用 llm_analysis.py 中的配置与密钥
from llm_analysis import API_KEY, SILICONFLOW_API_URL, SILICONFLOW_MODEL  # type: ignore


def format_time(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def format_time_range(start: str, end: str) -> str:
    if not start and not end:
        return "时间未知"
    if start and end:
        return f"{format_time(start)} ~ {format_time(end)}"
    if start:
        return f"{format_time(start)} 起"
    return f"{format_time(end)} 前"


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.4) -> str:
    if not API_KEY:
        raise RuntimeError("请在 llm_analysis.py 中设置 API_KEY 为你的 SiliconFlow 密钥。")

    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    return message.get("content") or ""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ChatMsg:
    is_self: bool
    sender_name: str
    content: str
    time: str


def load_chat_messages(limit_per_side: int = 50) -> Dict[str, Any]:
    raw = load_json(CHAT_PATH)
    msgs: List[ChatMsg] = []
    for m in raw:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if content.lstrip().startswith("<?xml"):
            continue
        msgs.append(
            ChatMsg(
                is_self=bool(m.get("isSelf")),
                sender_name=m.get("senderName") or "",
                content=content.replace("\n", " "),
                time=m.get("time") or "",
            )
        )

    # 简单抽样：从头到尾均匀抽取若干条
    def sample(side: bool) -> List[ChatMsg]:
        side_msgs = [m for m in msgs if m.is_self == side]
        if len(side_msgs) <= limit_per_side:
            return side_msgs
        step = max(1, len(side_msgs) // limit_per_side)
        return side_msgs[::step][:limit_per_side]

    def span(sample_msgs: List[ChatMsg]) -> str:
        times = [m.time for m in sample_msgs if m.time]
        if not times:
            return "时间未知"
        start = min(times)
        end = max(times)
        return format_time_range(start, end)

    self_samples = sample(True)
    other_samples = sample(False)

    return {
        "self": self_samples,
        "self_span": span(self_samples),
        "other": other_samples,
        "other_span": span(other_samples),
    }


def build_persona_prompt(role: str, keywords: List[List[Any]], topic_distribution: Dict[str, int],
                         communication_patterns: Dict[str, int], sample_msgs: List[ChatMsg],
                         time_span_desc: str) -> str:
    """
    role: "我" 或 "Ta"
    """
    kw_lines = [f"- {w}: {c}" for w, c in keywords]
    topic_lines = [f"- {t}: {c} 段提到" for t, c in topic_distribution.items()]
    comm_lines = [f"- {p}: {c} 段中出现" for p, c in communication_patterns.items()]

    sample_lines = []
    for m in sample_msgs[:40]:
        who = "我" if m.is_self else (m.sender_name or "Ta")
        sample_lines.append(f"{m.time} {who}: {m.content}")

    return (
        f"下面是关于「{role}」在一段长期微信聊天中的统计信息，请你据此为 {role} 写一张“人物画像卡片”。\n\n"
        "【高频用词】（词:出现次数）\n"
        + "\n".join(kw_lines[:40])
        + "\n\n【话题分布】（话题:在多少段对话里出现）\n"
        + "\n".join(topic_lines)
        + "\n\n【沟通模式统计】（模式:出现段数）\n"
        + "\n".join(comm_lines)
        + "\n\n【示例聊天片段】\n"
        + "\n".join(sample_lines)
        + f"\n\n（以上对话片段大致覆盖：{time_span_desc}）"
        + "\n\n请你用中文输出一段结构化的人物分析，包含：\n"
        "1. 性格特征（比如：外向/内向、感性/理性、敏感度等）\n"
        "2. 典型说话风格（习惯用什么词、句式、表达方式）\n"
        "3. 在关系中的角色和倾向（谁更主动、如何表达关心、如何处理冲突）\n"
        "4. 在压力或情绪波动时的表现\n"
        "5. 对对方可能的需求和期待\n\n"
        "尽量具体、生动，可以用小标题或列表，不需要再重复上面的原始数据。"
    )


def generate_personas():
    chat_stats = load_json(CHAT_STATS_PATH)
    llm_data = load_json(LLM_ANALYSIS_PATH)
    samples = load_chat_messages()

    keywords_self = chat_stats.get("keywords_top_self") or []
    keywords_other = chat_stats.get("keywords_top_other") or []
    topic_distribution = llm_data.get("topic_distribution") or {}
    communication_patterns = llm_data.get("communication_patterns") or {}

    system_prompt = (
        "你是一名温和、有共情力的中文关系分析师，擅长根据聊天统计信息"
        "为当事人写人物画像和沟通风格分析。"
    )

    # 我的人物画像
    user_prompt_self = build_persona_prompt(
        "我",
        keywords_self,
        topic_distribution,
        communication_patterns,
        samples["self"],
        samples.get("self_span", "时间未知"),
    )
    text_self = call_llm(system_prompt, user_prompt_self, temperature=0.5)

    # Ta 的人物画像
    user_prompt_other = build_persona_prompt(
        "Ta",
        keywords_other,
        topic_distribution,
        communication_patterns,
        samples["other"],
        samples.get("other_span", "时间未知"),
    )
    text_other = call_llm(system_prompt, user_prompt_other, temperature=0.5)

    profile_self_path = OUTPUT_DIR / "profile_self.txt"
    profile_other_path = OUTPUT_DIR / "profile_other.txt"

    with profile_self_path.open("w", encoding="utf-8") as f:
        f.write(text_self)
    with profile_other_path.open("w", encoding="utf-8") as f:
        f.write(text_other)

    print(f"人物画像已保存到: {profile_self_path} 和 {profile_other_path}")


def generate_story():
    llm_data = load_json(LLM_ANALYSIS_PATH)
    chunk_results: List[Dict[str, Any]] = llm_data.get("chunk_results") or []
    if not chunk_results:
        print("llm_analysis.json 中没有 chunk_results，无法生成故事。")
        return

    pieces: List[str] = []
    for r in sorted(chunk_results, key=lambda x: x.get("_chunk_index") or 0):
        idx = r.get("_chunk_index")
        summary = r.get("summary") or ""
        topics = r.get("topics") or []
        tr = r.get("time_range") or {}
        tr_text = format_time_range(tr.get("start"), tr.get("end"))
        pieces.append(
            f"【第 {idx} 段 | {tr_text}】\n"
            f"话题：{', '.join(topics)}\n"
            f"小结：{summary}\n"
        )

    joined = "\n\n".join(pieces)

    system_prompt = (
        "你是一名擅长讲述亲密关系故事的中文作者。"
    )

    user_prompt = (
        "下面是一段关系在不同时间的聊天小结，请你基于这些小结，"
        "为「我」写一篇第一人称的回忆录式短篇故事。\n\n"
        "要求：\n"
        "1. 语言真实自然，像在对一个亲密朋友倾诉。\n"
        "2. 按时间顺序，大致反映关系从起伏到变化的过程，可以适当补足细节，但不要胡乱制造与小结完全相反的情节。\n"
        "3. 重点写情绪和心理活动，而不是罗列事实。\n"
        "4. 字数大约 800-1500 字。\n"
        "5. 像着积极方面写，两个人是非常相爱的。\n"
        "6. 不要使用小标题，就按自然段讲完这个故事即可。\n\n"
        "以下是聊天小结：\n"
        f"{joined}"
    )

    story = call_llm(system_prompt, user_prompt, temperature=0.7)

    story_path = OUTPUT_DIR / "story.md"
    with story_path.open("w", encoding="utf-8") as f:
        f.write(story)

    print(f"年度叙事版故事已保存到: {story_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_personas()
    generate_story()


if __name__ == "__main__":
    main()
