import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

import requests


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis_output"
CHAT_PATH = OUTPUT_DIR / "chat.json"
OUTPUT_PATH = OUTPUT_DIR / "llm_analysis.json"

# 每段送给大模型的消息数量，视自己 key 和费用调整
CHUNK_SIZE = 1000

# SiliconFlow Chat Completions 接口配置
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2-Exp"

# 在这里填写你的 SiliconFlow API Key
# 示例：API_KEY = "sk-xxxxxxxxxxxxxxxx"
API_KEY = ""


@dataclass
class ChatMessage:
    time: str
    is_self: bool
    sender_name: str
    content: str


def load_messages(path: Path) -> List[ChatMessage]:
    if not path.exists():
        raise FileNotFoundError(f"chat.json not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    messages: List[ChatMessage] = []
    for m in raw:
        # 跳过非文本内容（图片、语音等）
        if m.get("type") != 1:
            continue

        content = (m.get("content") or "").strip()
        if not content:
            continue

        # 跳过 XML 等结构化内容
        if content.lstrip().startswith("<?xml"):
            continue

        messages.append(
            ChatMessage(
                time=m.get("time") or "",
                is_self=bool(m.get("isSelf")),
                sender_name=m.get("senderName") or "",
                content=content.replace("\n", " "),
            )
        )

    return messages


def sort_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    def parse_time(ts: str) -> float:
        try:
            return datetime.fromisoformat(ts).timestamp()
        except Exception:
            return 0.0

    return sorted(messages, key=lambda m: parse_time(m.time))


def chunk_messages(messages: List[ChatMessage], size: int) -> Iterable[List[ChatMessage]]:
    for i in range(0, len(messages), size):
        yield messages[i : i + size]


def format_chunk_for_llm(chunk: List[ChatMessage]) -> str:
    lines: List[str] = []
    for m in chunk:
        who = "我" if m.is_self else (m.sender_name or "Ta")
        lines.append(f"{m.time} {who}: {m.content}")
    return "\n".join(lines)


DEFAULT_SYSTEM_PROMPT = (
    "你是一名细致的中文对话分析师，专门分析两个人的聊天记录，"
    "从中提取话题分布、情绪变化、沟通模式等信息。"
)


def call_llm(
    user_prompt: str,
    temperature: float = 0.3,
    system_prompt: Optional[str] = None,
) -> str:
    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    if not API_KEY:
        raise RuntimeError("请在 llm_analysis.py 中设置 API_KEY 为你的 SiliconFlow 密钥。")

    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "temperature": temperature,
        "stream": False,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    resp = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    choices = data.get("choices") or []
    if not choices:
        return ""

    message = choices[0].get("message") or {}
    return message.get("content") or ""


def analyze_chunk_with_llm(chunk_index: int, text: str) -> Optional[Dict[str, Any]]:
    if not text.strip():
        return None

    prompt = (
        "下面是一段按时间排序的微信聊天记录节选，请你进行结构化分析，"
        "只用 JSON 格式回答，不要出现任何额外说明文字。\n\n"
        "要求 JSON 结构如下（字段名必须一致）：\n"
        "{\n"
        '  "topics": [字符串, ...],                // 本段主要话题标签，2-6 个\n'
        '  "self_emotion": "字符串",               // 对“我”的情绪概括，如：开心/难过/焦虑/愤怒/平静/复杂\n'
        '  "other_emotion": "字符串",              // 对“Ta”的情绪概括\n'
        '  "emotion_trend": "字符串",              // 这一小段内情绪的大致变化\n'
        '  "communication_pattern": [字符串, ...], // 本段体现出的沟通模式标签，2-6 个，比如：\n'
        '                                          // 亲密分享/理性讨论/回避冲突/倾听与安慰/误解与澄清/冷处理 等\n'
        '  "summary": "字符串"                     // 用 2-4 句话概括这一小段发生了什么\n'
        "}\n\n"
        "请务必返回合法 JSON，使用双引号，避免注释。\n\n"
        f"聊天记录如下：\n{text}"
    )

    raw = call_llm(prompt)

    # 尝试直接解析 JSON；若失败，可适当做一点清洗
    try:
        data = json.loads(raw)
        data["_chunk_index"] = chunk_index
        return data
    except Exception:
        # 简单兜底：从第一对大括号中截取
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(raw[start : end + 1])
                data["_chunk_index"] = chunk_index
                return data
            except Exception:
                return None
        return None


def get_chunk_time_range(chunk: List[ChatMessage]) -> Dict[str, Optional[str]]:
    times: List[datetime] = []
    for msg in chunk:
        ts = msg.time
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except Exception:
            continue
        times.append(dt)
    if not times:
        return {"start": None, "end": None}
    start = min(times).isoformat()
    end = max(times).isoformat()
    return {"start": start, "end": end}


def format_time(ts: Optional[str]) -> Optional[str]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


def format_time_range_text(time_range: Optional[Dict[str, Any]]) -> str:
    if not time_range:
        return "时间未知"
    start = format_time(time_range.get("start"))
    end = format_time(time_range.get("end"))
    if start and end:
        return f"{start} ~ {end}"
    if start:
        return f"{start} 起"
    if end:
        return f"{end} 前"
    return "时间未知"


def build_time_analysis_summary(time_segments: List[Dict[str, Any]]) -> str:
    if not time_segments:
        return ""

    segment_desc = []
    for seg in time_segments:
        topics = ", ".join(seg.get("topics") or []) or "（无话题信息）"
        summary = seg.get("summary") or ""
        segment_desc.append(
            f"第 {seg.get('chunk_index')} 段（{seg.get('time_text')}，话题：{topics}）\n"
            f"{summary}"
        )

    joined = "\n\n".join(segment_desc)

    prompt = (
        "下面是按时间顺序整理的聊天片段，请你聚焦“时间变化”给出分析：\n"
        "1. 整个时间线可粗分成哪几个阶段？每个阶段的时间范围和主要主题是什么？\n"
        "2. 情绪和互动模式在这些阶段如何变化？有没有明显的转折或重演？\n"
        "3. 请用要点列出 3-5 条“时间维度”的洞察，例如：某段时间格外高频、某段时间情绪持续低迷等。\n"
        "4. 最后用 1-2 句话概括“这一段关系在时间轴上的整体走势”。\n\n"
        "聊天片段：\n"
        f"{joined}"
    )

    system_prompt = (
        "你是一名擅长做时间线分析的中文对话研究者，善于从时间维度提炼模式。"
    )

    return call_llm(prompt, temperature=0.4, system_prompt=system_prompt)


def save_partial_output(chunk_results: List[Dict[str, Any]]):
    """把已经完成的分段分析立即写盘，防止后续步骤出错导致结果丢失。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_data = {
        "chunk_size": CHUNK_SIZE,
        "chunk_results": chunk_results,
        "partial": True,
        "note": "此文件仅包含分段分析结果，后续步骤完成后会被完整结果覆盖。",
    }
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(partial_data, f, ensure_ascii=False, indent=2)
    print(f"已保存分段分析的中间结果到: {OUTPUT_PATH}")


def load_saved_chunk_results() -> Optional[List[Dict[str, Any]]]:
    """如果存在之前保存的分段分析结果，则直接加载。"""
    if not OUTPUT_PATH.exists():
        return None
    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    chunks = data.get("chunk_results")
    if isinstance(chunks, list) and chunks:
        print(f"检测到已有分段分析结果，共 {len(chunks)} 段，将直接复用。")
        return chunks
    return None


def aggregate_topics(chunk_results: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for r in chunk_results:
        for t in r.get("topics") or []:
            if not isinstance(t, str):
                continue
            t = t.strip()
            if not t:
                continue
            counter[t] = counter.get(t, 0) + 1
    return counter


def build_emotion_timeline(chunk_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 直接按 chunk_index 排序，形成“段落级”情绪时间线
    return sorted(
        [
            {
                "chunk_index": r.get("_chunk_index"),
                "self_emotion": r.get("self_emotion"),
                "other_emotion": r.get("other_emotion"),
                "emotion_trend": r.get("emotion_trend"),
                "time_range": r.get("time_range"),
            }
            for r in chunk_results
        ],
        key=lambda x: x.get("chunk_index") or 0,
    )


def aggregate_communication_patterns(chunk_results: List[Dict[str, Any]]) -> Dict[str, int]:
    counter: Dict[str, int] = {}
    for r in chunk_results:
        for p in r.get("communication_pattern") or []:
            if not isinstance(p, str):
                continue
            p = p.strip()
            if not p:
                continue
            counter[p] = counter.get(p, 0) + 1
    return counter


def build_overall_summary(chunk_results: List[Dict[str, Any]]) -> str:
    # 把每段的小结拼接，再让大模型做一次总总结
    pieces: List[str] = []
    for r in sorted(chunk_results, key=lambda x: x.get("_chunk_index") or 0):
        idx = r.get("_chunk_index")
        summary = r.get("summary") or ""
        topics = r.get("topics") or []
        pieces.append(
            f"【第 {idx} 段】\n"
            f"话题：{', '.join(topics)}\n"
            f"小结：{summary}\n"
        )

    joined = "\n\n".join(pieces)

    prompt = (
        "下面是按时间顺序对多段聊天记录的小结与话题标签，请你给出整体分析，"
        "重点从三个角度回答：\n"
        "1. 话题分布：这段关系里反复出现的核心话题有哪些？大致占比和变化趋势如何？\n"
        "2. 情绪时间线：从开始到结束，双方情绪如何变化？有没有明显的高峰、低谷或转折点？\n"
        "3. 沟通模式：双方在沟通中的典型模式是什么？例如：谁更主动、谁更回避、谁更擅长安慰、"
        "   是偏理性讨论还是情绪表达为主？\n\n"
        "最后，请用温柔、不评判的语气，给出 3-5 条建议，帮助他们在未来更好地沟通。\n\n"
        "以下是各段小结：\n"
        f"{joined}"
    )

    return call_llm(prompt)


def main():
    messages = load_messages(CHAT_PATH)
    messages = sort_messages(messages)

    if not messages:
        print("没有可分析的文本消息。")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 优先尝试复用已经存在的分段分析结果
    chunk_results: List[Dict[str, Any]] = load_saved_chunk_results() or []

    if not chunk_results:
        for idx, chunk in enumerate(chunk_messages(messages, CHUNK_SIZE), start=1):
            text = format_chunk_for_llm(chunk)
            time_range = get_chunk_time_range(chunk)
            print(f"正在分析第 {idx} 段，对话条数：{len(chunk)} ...")
            result = analyze_chunk_with_llm(idx, text)
            if result is not None:
                result["time_range"] = time_range
                chunk_results.append(result)
                print(f"第 {idx} 段分析完成。")
            else:
                print(f"第 {idx} 段分析失败，已跳过。")

        if not chunk_results:
            print("未得到任何有效的分段分析结果。")
            return
        else:
            save_partial_output(chunk_results)
    
    topic_distribution = aggregate_topics(chunk_results)
    emotion_timeline = build_emotion_timeline(chunk_results)
    communication_patterns = aggregate_communication_patterns(chunk_results)
    overall_summary = build_overall_summary(chunk_results)

    # 构建带时间文本的段落列表
    time_segments: List[Dict[str, Any]] = []
    for r in sorted(chunk_results, key=lambda x: x.get("_chunk_index") or 0):
        tr_text = format_time_range_text(r.get("time_range"))
        time_segments.append(
            {
                "chunk_index": r.get("_chunk_index"),
                "time_range": r.get("time_range"),
                "time_text": tr_text,
                "topics": r.get("topics") or [],
                "summary": r.get("summary") or "",
            }
        )

    time_timeline_summary = build_time_analysis_summary(time_segments)

    output: Dict[str, Any] = {
        "chunk_size": CHUNK_SIZE,
        "chunk_results": chunk_results,
        "topic_distribution": topic_distribution,
        "emotion_timeline": emotion_timeline,
        "communication_patterns": communication_patterns,
        "overall_summary": overall_summary,
        "time_segments": time_segments,
        "time_timeline_summary": time_timeline_summary,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"大模型分析完成，结果已保存到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
